"""Code explanation aggregator for hierarchical code structures."""

from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Class, File, Function, Module, Repository
from src.logger import get_logger
from src.parser.code_extractor import CodeExtractor

logger = get_logger(__name__)

# Display limits
MAX_DISPLAY_ITEMS = 5


class CodeAggregator:
    """Aggregate code information for explanations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.code_extractor = CodeExtractor()

    async def explain_entity(
        self,
        entity_type: str,
        entity_id: int,
        *,
        include_code: bool = False,
    ) -> dict[str, Any]:
        """Generate comprehensive explanation for a code entity."""
        if entity_type == "function":
            return await self._explain_function(entity_id, include_code=include_code)
        if entity_type == "class":
            return await self._explain_class(entity_id, include_code=include_code)
        if entity_type == "module":
            return await self._explain_module(entity_id, include_code=include_code)
        if entity_type == "package":
            return await self._explain_package(entity_id, include_code=include_code)
        msg = f"Unknown entity type: {entity_type}"
        raise ValueError(msg)

    async def _explain_function(
        self,
        function_id: int,
        *,
        include_code: bool,
    ) -> dict[str, Any]:
        """Explain a function."""
        # Load function details
        func = await self.session.get(Function, function_id)
        if not func:
            return {"error": "Function not found"}

        # Load related data
        module = await self.session.get(Module, func.module_id)
        file = await self.session.get(File, module.file_id) if module else None

        # Load repository
        repo = None
        if file:
            repo = await self.session.get(Repository, file.repository_id)

        # Load class if it's a method
        class_info = None
        if func.class_id:
            cls = await self.session.get(Class, func.class_id)
            if cls:
                class_info = {
                    "name": cls.name,
                    "docstring": cls.docstring,
                }

        explanation = {
            "type": "method" if func.class_id else "function",
            "name": func.name,
            "qualified_name": self._build_qualified_name(func, module, class_info),
            "docstring": func.docstring,
            "signature": self._build_function_signature(func),
            "parameters": func.parameters,
            "return_type": func.return_type,
            "properties": {
                "is_async": func.is_async,
                "is_generator": func.is_generator,
                "is_property": func.is_property,
                "is_static": func.is_static,
                "is_classmethod": func.is_classmethod,
            },
            "decorators": func.decorators,
            "complexity": func.complexity,
            "location": {
                "file": file.path if file else "unknown",
                "start_line": func.start_line,
                "end_line": func.end_line,
                "repository": repo.name if repo else "unknown",
            },
        }

        if include_code and file:
            # Extract code content
            repo_path = Path("repositories") / repo.owner / repo.name if repo else None
            if repo_path and repo_path.exists():
                file_path = repo_path / file.path
                if file_path.exists():
                    code = self.code_extractor.extract_function_code(
                        file_path, func.start_line, func.end_line
                    )
                    explanation["code"] = code

        return explanation

    async def _explain_class(
        self,
        class_id: int,
        *,
        include_code: bool,
    ) -> dict[str, Any]:
        """Explain a class."""
        # Load class details
        cls = await self.session.get(Class, class_id)
        if not cls:
            return {"error": "Class not found"}

        # Load related data
        module = await self.session.get(Module, cls.module_id)
        file = await self.session.get(File, module.file_id) if module else None
        repo = None
        if file:
            repo = await self.session.get(Repository, file.repository_id)

        # Load methods
        result = await self.session.execute(
            select(Function)
            .where(Function.class_id == class_id)
            .limit(MAX_DISPLAY_ITEMS)
        )
        methods = result.scalars().all()

        explanation = {
            "type": "class",
            "name": cls.name,
            "qualified_name": f"{module.name}.{cls.name}" if module else cls.name,
            "docstring": cls.docstring,
            "base_classes": cls.base_classes,
            "decorators": cls.decorators,
            "properties": {
                "is_abstract": cls.is_abstract,
            },
            "location": {
                "file": file.path if file else "unknown",
                "start_line": cls.start_line,
                "end_line": cls.end_line,
                "repository": repo.name if repo else "unknown",
            },
            "methods": [
                {
                    "name": method.name,
                    "docstring": method.docstring,
                    "signature": self._build_function_signature(method),
                    "is_property": method.is_property,
                    "is_static": method.is_static,
                    "is_classmethod": method.is_classmethod,
                }
                for method in methods
            ],
        }

        if include_code and file:
            # Extract code content
            repo_path = Path("repositories") / repo.owner / repo.name if repo else None
            if repo_path and repo_path.exists():
                file_path = repo_path / file.path
                if file_path.exists():
                    code = self.code_extractor.extract_class_code(
                        file_path, cls.start_line, cls.end_line
                    )
                    explanation["code"] = code

        return explanation

    async def _explain_module(
        self,
        module_id: int,
        *,
        include_code: bool = False,
    ) -> dict[str, Any]:
        """Explain a module."""
        # Load module details
        module = await self.session.get(Module, module_id)
        if not module:
            return {"error": "Module not found"}

        # Load related data
        file = await self.session.get(File, module.file_id)
        repo = None
        if file:
            repo = await self.session.get(Repository, file.repository_id)

        # Load classes and functions
        result = await self.session.execute(
            select(Class).where(Class.module_id == module_id).limit(MAX_DISPLAY_ITEMS)
        )
        classes = result.scalars().all()

        result = await self.session.execute(
            select(Function)
            .where(
                Function.module_id == module_id,
                Function.class_id.is_(None),  # Only module-level functions
            )
            .limit(MAX_DISPLAY_ITEMS)
        )
        functions = result.scalars().all()

        explanation = {
            "type": "module",
            "name": module.name,
            "qualified_name": module.name,
            "docstring": module.docstring,
            "location": {
                "file": file.path if file else "unknown",
                "repository": repo.name if repo else "unknown",
            },
            "classes": [
                {
                    "name": cls.name,
                    "docstring": cls.docstring,
                    "is_abstract": cls.is_abstract,
                }
                for cls in classes
            ],
            "functions": [
                {
                    "name": func.name,
                    "docstring": func.docstring,
                    "signature": self._build_function_signature(func),
                }
                for func in functions
            ],
        }

        return explanation

    async def _explain_package(
        self,
        package_id: int,
        *,
        include_code: bool = False,
    ) -> dict[str, Any]:
        """Explain a package (placeholder)."""
        return {
            "type": "package",
            "name": f"package_{package_id}",
            "docstring": "Package explanation not implemented yet",
            "modules": [],
        }

    def _build_qualified_name(
        self,
        func: Function,
        module: Module | None,
        class_info: dict[str, Any] | None,
    ) -> str:
        """Build qualified name for a function."""
        parts = []
        if module:
            parts.append(module.name)
        if class_info:
            parts.append(class_info["name"])
        parts.append(func.name)
        return ".".join(parts)

    def _build_function_signature(self, func: Function) -> str:
        """Build function signature string."""
        params = []
        if func.parameters:
            for param in func.parameters:
                param_str = param.get("name", "")
                if param.get("type"):
                    param_str += f": {param['type']}"
                if param.get("default"):
                    param_str += f" = {param['default']}"
                params.append(param_str)

        signature = f"{func.name}({', '.join(params)})"
        if func.return_type:
            signature += f" -> {func.return_type}"

        return signature
