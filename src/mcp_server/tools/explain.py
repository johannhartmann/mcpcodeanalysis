"""Explain tool for MCP server."""

from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src.database import get_session_factory, init_database
from src.database.models import Class, File, Function, Module
from src.database.repositories import CodeEntityRepo
from src.logger import get_logger
from src.query.aggregator import CodeAggregator

logger = get_logger(__name__)

# Constants
MIN_CLASS_METHODS = 2
MIN_MODULE_COMPONENTS = 3
MAX_FUNCTION_LIST = 10


class ExplainTool:
    """MCP tool for explaining code entities."""

    def __init__(self) -> None:
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    async def _get_session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get database session factory, initializing if needed."""
        if self._session_factory is None:
            engine = await init_database()
            self._engine = engine
            # sessionmaker returns a sessionmaker[AsyncSession]
            factory = get_session_factory(engine)
            self._session_factory = factory
        if self._session_factory is None:
            raise RuntimeError
        return self._session_factory

    async def explain_code(self, path: str) -> str:
        """
        Explain what a code element does (function, class, module, or package).
        For classes: aggregates explanations of all methods and attributes.
        For packages: provides overview of all modules and main components.

        Args:
            path: Path to code element (e.g., "module.Class.method" or "path/to/module.py")

        Returns:
            Detailed explanation of the code element
        """
        try:
            session_factory = await self._get_session_factory()
            async with session_factory() as session:
                entity_repo = CodeEntityRepo(session)
                # Parse the path to determine entity type and location
                entity_info = await self._parse_code_path(path, session, entity_repo)

                if not entity_info:
                    return f"Could not find code element at path: {path}"

                # Get explanation
                aggregator = CodeAggregator(session)
                explanation = await aggregator.explain_entity(
                    entity_type=entity_info["type"],
                    entity_id=entity_info["id"],
                    _include_code=False,
                )

                # Format explanation as text
                return self._format_explanation(explanation)

        except Exception as e:
            logger.exception("Error in explain_code: %s")
            return f"Error explaining code: {e!s}"

    async def _parse_code_path(
        self, path: str, session: AsyncSession, entity_repo: CodeEntityRepo
    ) -> dict[str, Any] | None:
        """Parse a code path to find the entity."""
        # Check if it's a file path
        if "/" in path or path.endswith(".py"):
            # File path - look for module
            # Extract just the filename from the full path for searching
            file_name = path.split("/")[-1] if "/" in path else path
            result = await session.execute(
                select(File).where(File.path.like(f"%{file_name}")),
            )
            files = result.scalars().all()

            file = None
            for f in files:
                if f.path.endswith(path):
                    file = f
                    break

            if file:
                result = await session.execute(
                    select(Module).where(Module.file_id == file.id).limit(1),
                )
                module = result.scalar_one_or_none()
                if module:
                    return {"type": "module", "id": module.id}

        # Check if it's a qualified name (module.Class.method)
        parts = path.split(".")

        # Try to find by name
        if len(parts) == 1:
            # Single name - could be function, class, or module
            results = await entity_repo.find_by_name(parts[0])
            if results:
                entity = results[0]
                return {"type": entity["type"], "id": entity["entity"].id}

        elif len(parts) == MIN_CLASS_METHODS:
            # Could be module.function or class.method
            # First try module.function
            module_results = await entity_repo.find_by_name(
                parts[0],
                "module",
            )
            if module_results:
                module = module_results[0]["entity"]
                # Look for function in this module
                result = await session.execute(
                    select(Function)
                    .where(
                        and_(
                            Function.module_id == module.id,
                            Function.name == parts[1],
                            Function.class_id.is_(None),
                        )
                    )
                    .limit(1),
                )
                func = result.scalar_one_or_none()
                if func:
                    return {"type": "function", "id": func.id}

            # Try class.method
            class_results = await entity_repo.find_by_name(
                parts[0],
                "class",
            )
            if class_results:
                cls = class_results[0]["entity"]
                # Look for method in this class
                result = await session.execute(
                    select(Function)
                    .where(
                        and_(
                            Function.class_id == cls.id,
                            Function.name == parts[1],
                        )
                    )
                    .limit(1),
                )
                method = result.scalar_one_or_none()
                if method:
                    return {"type": "function", "id": method.id}

        elif len(parts) >= MIN_MODULE_COMPONENTS:
            # module.class.method
            module_results = await entity_repo.find_by_name(
                parts[0],
                "module",
            )
            if module_results:
                module = module_results[0]["entity"]
                # Look for class in module
                result = await session.execute(
                    select(Class)
                    .where(
                        and_(
                            Class.module_id == module.id,
                            Class.name == parts[1],
                        )
                    )
                    .limit(1),
                )
                cls = result.scalar_one_or_none()
                if cls:
                    if len(parts) == MIN_MODULE_COMPONENTS:
                        # Look for method
                        result = await session.execute(
                            select(Function)
                            .where(
                                and_(
                                    Function.class_id == cls.id,
                                    Function.name == parts[2],
                                )
                            )
                            .limit(1),
                        )
                        method = result.scalar_one_or_none()
                        if method:
                            return {"type": "function", "id": method.id}
                    else:
                        # Return the class
                        return {"type": "class", "id": cls.id}

        return None

    def _format_explanation(self, explanation: dict[str, Any]) -> str:
        """Format explanation dictionary as readable text."""
        if "error" in explanation:
            return str(explanation["error"])

        parts = []

        # Header
        entity_type = explanation.get("type", "entity")
        name = explanation.get("name", "unknown")
        qualified_name = explanation.get("qualified_name", name)

        parts.append(f"# {entity_type.capitalize()}: {qualified_name}")
        parts.append("")

        # Location
        location = explanation.get("location", {})
        if location.get("repository"):
            parts.append(f"**Repository**: {location['repository']}")
            parts.append(f"**File**: {location.get('file', 'unknown')}")
            parts.append(
                f"**Lines**: {location.get('start_line', '?')}-{location.get('end_line', '?')}",
            )
            parts.append("")

        # Docstring
        if explanation.get("docstring"):
            parts.append("## Documentation")
            parts.append(explanation["docstring"])
            parts.append("")

        # Type-specific information
        if entity_type in {"function", "method"}:
            # Signature
            parts.append("## Signature")
            parts.append("```python")
            parts.append(explanation.get("signature", f"{name}()"))
            parts.append("```")
            parts.append("")

            # Parameters
            if explanation.get("parameters"):
                parts.append("## Parameters")
                for param in explanation["parameters"]:
                    param_str = f"- **{param['name']}**"
                    if param.get("type"):
                        param_str += f" ({param['type']})"
                    if param.get("default"):
                        param_str += f" = {param['default']}"
                    parts.append(param_str)
                parts.append("")

            # Properties
            props = explanation.get("properties", {})
            special = []
            if props.get("is_async"):
                special.append("async")
            if props.get("is_generator"):
                special.append("generator")
            if props.get("is_property"):
                special.append("property")
            if props.get("is_staticmethod"):
                special.append("static method")
            if props.get("is_classmethod"):
                special.append("class method")

            if special:
                parts.append(f"**Type**: {', '.join(special)}")
                parts.append("")

        elif entity_type == "class":
            # Inheritance
            if explanation.get("base_classes"):
                parts.append("## Inheritance")
                parts.append(f"Inherits from: {', '.join(explanation['base_classes'])}")
                parts.append("")

            # Methods
            for method_type in [
                "methods",
                "properties",
                "class_methods",
                "static_methods",
                "special_methods",
            ]:
                method_list = explanation.get(method_type, [])
                if method_list:
                    title = method_type.replace("_", " ").title()
                    parts.append(f"## {title}")
                    for method in method_list:
                        parts.append(
                            f"- **{method['name']}**: {method.get('signature', method['name'] + '()')}",
                        )
                        if method.get("docstring"):
                            parts.append(f"  {method['docstring'].split('.')[0]}.")
                    parts.append("")

        elif entity_type == "module":
            # Imports
            if explanation.get("imports"):
                parts.append("## Imports")
                for imp in explanation["imports"][:10]:
                    parts.append(f"- {imp['statement']}")
                if len(explanation["imports"]) > MAX_FUNCTION_LIST:
                    parts.append(f"- ... and {len(explanation['imports']) - 10} more")
                parts.append("")

            # Classes
            if explanation.get("classes"):
                parts.append("## Classes")
                for cls in explanation["classes"]:
                    parts.append(f"- **{cls['name']}**")
                    if cls.get("docstring"):
                        parts.append(f"  {cls['docstring'].split('.')[0]}.")
                parts.append("")

            # Functions
            if explanation.get("functions"):
                parts.append("## Functions")
                for func in explanation["functions"]:
                    parts.append(f"- **{func['signature']}**")
                    if func.get("docstring"):
                        parts.append(f"  {func['docstring'].split('.')[0]}.")
                parts.append("")

        # Interpretation
        if explanation.get("interpretation"):
            parts.append("## Summary")
            parts.append(explanation["interpretation"])

        return "\n".join(parts)
