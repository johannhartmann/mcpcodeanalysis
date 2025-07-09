"""Code explanation aggregator for hierarchical code structures."""

from pathlib import Path
from typing import Any

from sqlalchemy import text

from src.database import get_repositories, get_session
from src.database.models import Class, File, Function, Module
from src.parser.code_extractor import CodeExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CodeAggregator:
    """Aggregate code information for explanations."""

    def __init__(self) -> None:
        self.code_extractor = CodeExtractor()

    async def explain_entity(
        self,
        entity_type: str,
        entity_id: int,
        include_code: bool = False,
    ) -> dict[str, Any]:
        """Generate comprehensive explanation for a code entity."""
        async with get_session() as session, get_repositories(session) as repos:
            if entity_type == "function":
                return await self._explain_function(repos, entity_id, include_code)
            if entity_type == "class":
                return await self._explain_class(repos, entity_id, include_code)
            if entity_type == "module":
                return await self._explain_module(repos, entity_id, include_code)
            if entity_type == "package":
                return await self._explain_package(repos, entity_id, include_code)
            raise ValueError(f"Unknown entity type: {entity_type}")

    async def _explain_function(
        self,
        repos: dict[str, Any],
        function_id: int,
        include_code: bool,
    ) -> dict[str, Any]:
        """Explain a function."""
        session = repos["repository"].session

        # Load function details
        func = await session.get(Function, function_id)
        if not func:
            return {"error": "Function not found"}

        # Load related data
        module = await session.get(Module, func.module_id)
        file = await session.get(File, module.file_id) if module else None
        repo = await repos["repository"].get_by_id(file.repository_id) if file else None

        # Load class if it's a method
        class_info = None
        if func.class_id:
            cls = await session.get(Class, func.class_id)
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
                "is_staticmethod": func.is_staticmethod,
                "is_classmethod": func.is_classmethod,
            },
            "decorators": func.decorators,
            "complexity": func.complexity,
            "location": {
                "repository": repo.name if repo else None,
                "repository_url": repo.github_url if repo else None,
                "file": file.path if file else None,
                "start_line": func.start_line,
                "end_line": func.end_line,
            },
        }

        if class_info:
            explanation["class"] = class_info

        if include_code and file:
            # Get code snippet
            file_path = Path("./repositories") / repo.owner / repo.name / file.path
            if file_path.exists():
                raw_code, _ = self.code_extractor.get_entity_content(
                    file_path,
                    "function",
                    func.start_line,
                    func.end_line,
                    include_context=False,
                )
                explanation["code"] = raw_code

        # Generate interpretation
        explanation["interpretation"] = self._interpret_function(func, class_info)

        return explanation

    async def _explain_class(
        self,
        repos: dict[str, Any],
        class_id: int,
        include_code: bool,
    ) -> dict[str, Any]:
        """Explain a class with all its methods."""
        session = repos["repository"].session

        # Load class details
        cls = await session.get(Class, class_id)
        if not cls:
            return {"error": "Class not found"}

        # Load related data
        module = await session.get(Module, cls.module_id)
        file = await session.get(File, module.file_id) if module else None
        repo = await repos["repository"].get_by_id(file.repository_id) if file else None

        # Load all methods
        methods = await session.execute(
            text("SELECT * FROM functions WHERE class_id = :class_id ORDER BY start_line"),
            {"class_id": class_id},
        )
        method_list = list(methods.scalars().all()) if methods else []

        explanation = {
            "type": "class",
            "name": cls.name,
            "qualified_name": self._build_qualified_name(cls, module),
            "docstring": cls.docstring,
            "base_classes": cls.base_classes,
            "is_abstract": cls.is_abstract,
            "decorators": cls.decorators,
            "location": {
                "repository": repo.name if repo else None,
                "repository_url": repo.github_url if repo else None,
                "file": file.path if file else None,
                "start_line": cls.start_line,
                "end_line": cls.end_line,
            },
            "methods": [],
            "properties": [],
            "class_methods": [],
            "static_methods": [],
            "special_methods": [],
        }

        # Categorize methods
        for method in method_list:
            method_info = {
                "name": method.name,
                "docstring": method.docstring,
                "signature": self._build_function_signature(method),
                "is_async": method.is_async,
                "decorators": method.decorators,
            }

            if method.is_property:
                explanation["properties"].append(method_info)
            elif method.is_classmethod:
                explanation["class_methods"].append(method_info)
            elif method.is_staticmethod:
                explanation["static_methods"].append(method_info)
            elif method.name.startswith("__") and method.name.endswith("__"):
                explanation["special_methods"].append(method_info)
            else:
                explanation["methods"].append(method_info)

        if include_code and file:
            # Get code snippet
            file_path = Path("./repositories") / repo.owner / repo.name / file.path
            if file_path.exists():
                raw_code, _ = self.code_extractor.get_entity_content(
                    file_path,
                    "class",
                    cls.start_line,
                    cls.end_line,
                    include_context=False,
                )
                explanation["code"] = raw_code

        # Generate aggregated interpretation
        explanation["interpretation"] = self._interpret_class(cls, method_list)

        return explanation

    async def _explain_module(
        self,
        repos: dict[str, Any],
        module_id: int,
        include_code: bool,
    ) -> dict[str, Any]:
        """Explain a module with all its contents."""
        session = repos["repository"].session

        # Load module details
        module = await session.get(Module, module_id)
        if not module:
            return {"error": "Module not found"}

        # Load related data
        file = await session.get(File, module.file_id)
        repo = await repos["repository"].get_by_id(file.repository_id) if file else None

        # Load classes and functions
        classes = await session.execute(
            text("SELECT * FROM classes WHERE module_id = :module_id ORDER BY start_line"),
            {"module_id": module_id},
        )
        class_list = list(classes.scalars().all()) if classes else []

        functions = await session.execute(
            text("SELECT * FROM functions WHERE module_id = :module_id AND class_id IS NULL ORDER BY start_line"),
            {"module_id": module_id},
        )
        function_list = list(functions.scalars().all()) if functions else []

        # Load imports
        imports = await session.execute(
            text("SELECT * FROM imports WHERE file_id = :file_id ORDER BY line_number"),
            {"file_id": file.id},
        )
        import_list = list(imports.scalars().all()) if imports else []

        explanation = {
            "type": "module",
            "name": module.name,
            "docstring": module.docstring,
            "location": {
                "repository": repo.name if repo else None,
                "repository_url": repo.github_url if repo else None,
                "file": file.path if file else None,
            },
            "imports": [
                {
                    "statement": imp.import_statement,
                    "from": imp.imported_from,
                    "names": imp.imported_names,
                }
                for imp in import_list
            ],
            "classes": [
                {
                    "name": cls.name,
                    "docstring": cls.docstring,
                    "is_abstract": cls.is_abstract,
                    "base_classes": cls.base_classes,
                }
                for cls in class_list
            ],
            "functions": [
                {
                    "name": func.name,
                    "docstring": func.docstring,
                    "signature": self._build_function_signature(func),
                    "is_async": func.is_async,
                }
                for func in function_list
            ],
        }

        # Generate module interpretation
        explanation["interpretation"] = self._interpret_module(
            module,
            class_list,
            function_list,
            import_list,
        )

        return explanation

    async def _explain_package(
        self,
        repos: dict[str, Any],
        package_path: str,
        include_code: bool,
    ) -> dict[str, Any]:
        """Explain a package (directory with multiple modules)."""
        # This would require more complex logic to identify all modules in a package
        # For now, return a placeholder
        return {
            "type": "package",
            "path": package_path,
            "interpretation": "Package explanation not yet implemented",
        }

    def _build_qualified_name(
        self,
        entity: Any,
        module: Any | None = None,
        class_info: dict[str, Any] | None = None,
    ) -> str:
        """Build fully qualified name for an entity."""
        parts = []

        if module:
            parts.append(module.name)

        if class_info:
            parts.append(class_info["name"])
        elif hasattr(entity, "name") and entity.name:
            parts.append(entity.name)

        return ".".join(parts)

    def _build_function_signature(self, func: Function) -> str:
        """Build function signature string."""
        params = []

        for param in func.parameters or []:
            param_str = param["name"]
            if param.get("type"):
                param_str += f": {param['type']}"
            if param.get("default"):
                param_str += f" = {param['default']}"
            params.append(param_str)

        signature = f"{func.name}({', '.join(params)})"

        if func.return_type:
            signature += f" -> {func.return_type}"

        return signature

    def _interpret_function(
        self,
        func: Function,
        class_info: dict[str, Any] | None = None,
    ) -> str:
        """Generate natural language interpretation of a function."""
        parts = []

        # Basic description
        func_type = "method" if class_info else "function"
        if func.is_async:
            func_type = f"async {func_type}"

        parts.append(
            f"This is {'an' if func.is_async else 'a'} {func_type} named '{func.name}'",
        )

        # Purpose from docstring
        if func.docstring:
            parts.append(f"that {func.docstring.split('.')[0].lower()}")

        # Parameters
        if func.parameters:
            param_names = [
                p["name"] for p in func.parameters if p["name"] not in ("self", "cls")
            ]
            if param_names:
                parts.append(
                    f"It takes {len(param_names)} parameter(s): {', '.join(param_names)}",
                )

        # Return value
        if func.return_type and func.return_type != "None":
            parts.append(f"and returns {func.return_type}")
        elif func.is_generator:
            parts.append("and yields values as a generator")

        # Special properties
        if func.is_property:
            parts.append("This is a property accessor")
        elif func.is_staticmethod:
            parts.append("This is a static method that doesn't require an instance")
        elif func.is_classmethod:
            parts.append(
                "This is a class method that receives the class as first argument",
            )

        return ". ".join(parts) + "."

    def _interpret_class(self, cls: Class, methods: list[Function]) -> str:
        """Generate natural language interpretation of a class."""
        parts = []

        # Basic description
        if cls.is_abstract:
            parts.append(f"This is an abstract class named '{cls.name}'")
        else:
            parts.append(f"This is a class named '{cls.name}'")

        # Inheritance
        if cls.base_classes:
            parts.append(f"that inherits from {', '.join(cls.base_classes)}")

        # Purpose from docstring
        if cls.docstring:
            parts.append(f"\n\nPurpose: {cls.docstring}")

        # Method summary
        if methods:
            method_count = len(methods)
            property_count = sum(1 for m in methods if m.is_property)
            special_count = sum(1 for m in methods if m.name.startswith("__"))

            parts.append(f"\n\nThe class has {method_count} method(s)")

            if property_count:
                parts.append(f"including {property_count} properties")
            if special_count:
                parts.append(f"and {special_count} special methods")

        return ". ".join(parts) + "."

    def _interpret_module(
        self,
        module: Module,
        classes: list[Class],
        functions: list[Function],
        imports: list[Any],
    ) -> str:
        """Generate natural language interpretation of a module."""
        parts = []

        parts.append(f"This is a Python module named '{module.name}'")

        # Purpose from docstring
        if module.docstring:
            parts.append(f"that {module.docstring.split('.')[0].lower()}")

        # Content summary
        summaries = []
        if classes:
            summaries.append(f"{len(classes)} class(es)")
        if functions:
            summaries.append(f"{len(functions)} function(s)")

        if summaries:
            parts.append(f"\n\nIt contains: {', '.join(summaries)}")

        # Main components
        if classes:
            class_names = [cls.name for cls in classes[:5]]
            parts.append(f"\n\nMain classes: {', '.join(class_names)}")
            if len(classes) > 5:
                parts.append(f"and {len(classes) - 5} more")

        if functions:
            func_names = [func.name for func in functions[:5]]
            parts.append(f"\n\nMain functions: {', '.join(func_names)}")
            if len(functions) > 5:
                parts.append(f"and {len(functions) - 5} more")

        return ". ".join(parts) + "."
