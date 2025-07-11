"""Analysis tools for MCP server."""

import re
from typing import Any

from sqlalchemy import and_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Class, File, Function, Module
from src.database.repositories import (
    CodeEntityRepo,
    EmbeddingRepo,
    FileRepo,
)
from src.indexer.embeddings import EmbeddingGenerator
from src.logger import get_logger

logger = get_logger(__name__)

# Constants for code quality thresholds
MAX_FUNCTION_LENGTH = 50
MAX_FUNCTION_PARAMS = 5
MAX_CLASS_METHODS = 20
MAX_FILE_LENGTH = 200
HIGH_COMPLEXITY_THRESHOLD = 20


class AnalyzeTool:
    """MCP tools for code analysis."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.embedding_generator = EmbeddingGenerator()
        self.file_repo = FileRepo(session)
        self.entity_repo = CodeEntityRepo(session)
        self.embedding_repo = EmbeddingRepo(session)

    async def analyze_dependencies(self, module_path: str) -> dict[str, Any]:
        """
        Analyze dependencies of a module.

        Args:
            module_path: Path to the module (e.g., "src/utils/logger.py")

        Returns:
            Dictionary containing dependency analysis
        """
        try:
            # Find the file
            from src.database.models import File

            result = await self.session.execute(
                select(File).where(File.path.like(f"%{module_path}")),
            )
            files = result.scalars().all()

            file = None
            for f in files:
                if f.path.endswith(module_path):
                    file = f
                    break

            if not file:
                return {"error": f"Module not found: {module_path}"}

            # Get imports
            from src.database.models import Import

            result = await self.session.execute(
                select(Import).where(Import.file_id == file.id),
            )
            import_list = list(result.scalars().all())

            # Analyze dependencies
            analysis: dict[str, Any] = {
                "module": module_path,
                "direct_dependencies": [],
                "external_dependencies": [],
                "internal_dependencies": [],
                "dependency_graph": {},
                "circular_dependencies": [],
            }

            # Categorize imports
            for imp in import_list:
                dep_info = {
                    "module": imp.imported_from or "direct import",
                    "names": imp.imported_names,
                    "statement": imp.import_statement,
                    "is_relative": imp.is_relative,
                }

                # Categorize
                if imp.imported_from:
                    module_name = imp.imported_from.split(".")[0]
                    if module_name in [
                        "os",
                        "sys",
                        "json",
                        "datetime",
                        "typing",
                        "pathlib",
                    ]:
                        dep_info["type"] = "stdlib"
                        analysis["external_dependencies"].append(dep_info)
                    elif imp.is_relative or module_name == "src":
                        dep_info["type"] = "internal"
                        analysis["internal_dependencies"].append(dep_info)
                    else:
                        dep_info["type"] = "third_party"
                        analysis["external_dependencies"].append(dep_info)

                analysis["direct_dependencies"].append(dep_info)

            # Build dependency graph (simplified)
            analysis["dependency_graph"] = {
                "module": module_path,
                "imports": len(import_list),
                "imported_by": await self._find_importers(module_path),
            }

            # Check for circular dependencies (simplified)
            # This would need more sophisticated analysis in practice
            analysis["circular_dependencies"] = []

            return analysis

        except Exception as e:
            logger.exception("Error in analyze_dependencies: %s")
            return {"error": str(e), "module_path": module_path}

    async def suggest_refactoring(self, code_path: str) -> list[dict[str, Any]]:
        """
        Suggest refactoring improvements.

        Args:
            code_path: Path to code element (function, class, or module)

        Returns:
            List of refactoring suggestions
        """
        try:
            suggestions = []

            # Parse the code path to find entity
            entity_info = await self._find_entity_by_path(code_path)

            if not entity_info:
                return [
                    {
                        "error": f"Entity not found: {code_path}",
                        "suggestions": [],
                    },
                ]

            entity = entity_info["entity"]
            entity_type = entity_info["type"]

            # Analyze based on entity type
            if entity_type == "function":
                suggestions.extend(await self._analyze_function(entity))
            elif entity_type == "class":
                suggestions.extend(await self._analyze_class(entity))
            elif entity_type == "module":
                suggestions.extend(await self._analyze_module(entity))

            return suggestions

        except Exception as e:
            logger.exception("Error in suggest_refactoring: %s")
            return [{"error": str(e), "code_path": code_path}]

    async def find_similar_code(self, code_snippet: str) -> list[dict[str, Any]]:
        """
        Find similar code patterns in the codebase.

        Args:
            code_snippet: Code snippet to find similar patterns for

        Returns:
            List of similar code locations
        """
        try:
            # Generate embedding for the snippet
            snippet_embedding = await self.embedding_generator.generate_embedding(
                code_snippet,
            )

            # Search for similar embeddings
            similar = await self.embedding_repo.search_similar(
                snippet_embedding,
                limit=20,
                threshold=0.8,
                embedding_type="raw",
            )

            results = []
            for embedding, similarity in similar:
                # Get entity details
                entity_data = await self._get_entity_details(
                    embedding.entity_type,
                    embedding.entity_id,
                )

                if entity_data:
                    results.append(
                        {
                            "similarity": round(similarity, 3),
                            "entity": entity_data,
                            "matched_content": embedding.content[:200] + "...",
                        },
                    )

            return results

        except Exception as e:
            logger.exception("Error in find_similar_code: %s")
            return [{"error": str(e), "code_snippet": code_snippet[:100] + "..."}]

    async def get_code_structure(self, path: str) -> dict[str, Any]:
        """
        Get hierarchical structure of a module/package.

        Args:
            path: Path to module or package

        Returns:
            Hierarchical structure information
        """
        try:
            # Check if it's a file or directory
            if path.endswith(".py"):
                # Single module
                return await self._get_module_structure(path)
            # Package/directory
            return await self._get_package_structure(path)

        except Exception as e:
            logger.exception("Error in get_code_structure: %s")
            return {"error": str(e), "path": path}

    async def _find_importers(
        self,
        module_path: str,
    ) -> list[str]:
        """Find modules that import this module."""
        # Use the instance session
        importers = []

        # Extract module name from path
        module_name = module_path.replace("/", ".").replace(".py", "")

        # Find imports

        # Using raw SQL for complex join
        query = text(
            """
            SELECT DISTINCT f.path
            FROM files f
            JOIN imports i ON i.file_id = f.id
            WHERE i.imported_from LIKE :pattern
        """
        )
        result = await self.session.execute(query, {"pattern": f"%{module_name}%"})
        importers = [row[0] for row in result.all()]

        return importers[:10]  # Limit to 10

    async def _find_entity_by_path(
        self,
        path: str,
    ) -> dict[str, Any] | None:
        """Find entity by path."""
        # This is simplified - would need proper path parsing
        parts = path.split(".")

        if len(parts) == 1:
            # Could be function, class, or module
            results = await self.entity_repo.find_by_name(parts[0])
            if results:
                return {"entity": results[0]["entity"], "type": results[0]["type"]}
            return None
            if results:
                return {"entity": results[0]["entity"], "type": results[0]["type"]}

        return None

    async def _analyze_function(self, func: Function) -> list[dict[str, Any]]:
        """Analyze a function for refactoring suggestions."""
        suggestions = []

        # Check function length
        if func.end_line - func.start_line > MAX_FUNCTION_LENGTH:
            suggestions.append(
                {
                    "type": "complexity",
                    "severity": "medium",
                    "message": f"Function '{func.name}' is too long ({func.end_line - func.start_line} lines)",
                    "suggestion": "Consider breaking it into smaller functions",
                },
            )

        # Check parameter count
        param_count = len(func.parameters or [])
        if param_count > MAX_FUNCTION_PARAMS:
            suggestions.append(
                {
                    "type": "complexity",
                    "severity": "medium",
                    "message": f"Function '{func.name}' has too many parameters ({param_count})",
                    "suggestion": "Consider using a configuration object or builder pattern",
                },
            )

        # Check for missing docstring
        if not func.docstring:
            suggestions.append(
                {
                    "type": "documentation",
                    "severity": "low",
                    "message": f"Function '{func.name}' is missing a docstring",
                    "suggestion": "Add a docstring explaining the function's purpose, parameters, and return value",
                },
            )

        # Check naming convention
        if not self._is_snake_case(func.name):
            suggestions.append(
                {
                    "type": "style",
                    "severity": "low",
                    "message": f"Function name '{func.name}' doesn't follow snake_case convention",
                    "suggestion": f"Rename to '{self._to_snake_case(func.name)}'",
                },
            )

        return suggestions

    async def _analyze_class(
        self,
        cls: Class,
    ) -> list[dict[str, Any]]:
        """Analyze a class for refactoring suggestions."""
        suggestions = []
        # Use the instance session

        # Get methods
        result = await self.session.execute(
            select(Function).where(Function.class_id == cls.id),
        )
        method_list = list(result.scalars().all())

        # Check class size
        if cls.end_line - cls.start_line > MAX_FILE_LENGTH:
            suggestions.append(
                {
                    "type": "complexity",
                    "severity": "high",
                    "message": f"Class '{cls.name}' is too large ({cls.end_line - cls.start_line} lines)",
                    "suggestion": "Consider splitting into multiple classes with single responsibilities",
                },
            )

        # Check method count
        if len(method_list) > MAX_CLASS_METHODS:
            suggestions.append(
                {
                    "type": "complexity",
                    "severity": "medium",
                    "message": f"Class '{cls.name}' has too many methods ({len(method_list)})",
                    "suggestion": "Consider extracting related methods into separate classes",
                },
            )

        # Check for missing docstring
        if not cls.docstring:
            suggestions.append(
                {
                    "type": "documentation",
                    "severity": "low",
                    "message": f"Class '{cls.name}' is missing a docstring",
                    "suggestion": "Add a docstring explaining the class's purpose and responsibilities",
                },
            )

        # Check naming convention
        if not self._is_camel_case(cls.name):
            suggestions.append(
                {
                    "type": "style",
                    "severity": "low",
                    "message": f"Class name '{cls.name}' doesn't follow CamelCase convention",
                    "suggestion": f"Rename to '{self._to_camel_case(cls.name)}'",
                },
            )

        return suggestions

    async def _analyze_module(
        self,
        module: Module,
    ) -> list[dict[str, Any]]:
        """Analyze a module for refactoring suggestions."""
        suggestions = []
        # Use the instance session

        # Get imports
        file = await self.session.get(File, module.file_id)
        import_count = 0
        if file:
            from src.database.models import Import

            result = await self.session.execute(
                select(Import).where(Import.file_id == file.id),
            )
            import_count = len(list(result.scalars().all()))

        if import_count > MAX_CLASS_METHODS:
            suggestions.append(
                {
                    "type": "complexity",
                    "severity": "medium",
                    "message": f"Module '{module.name}' has too many imports ({import_count})",
                    "suggestion": "Consider splitting the module or reducing dependencies",
                },
            )

        # Check for missing docstring
        if not module.docstring:
            suggestions.append(
                {
                    "type": "documentation",
                    "severity": "low",
                    "message": f"Module '{module.name}' is missing a module-level docstring",
                    "suggestion": "Add a docstring at the beginning of the file explaining the module's purpose",
                },
            )

        return suggestions

    async def _get_entity_details(
        self,
        entity_type: str,
        entity_id: int,
    ) -> dict[str, Any] | None:
        """Get entity details for similarity results."""
        # Use the instance session

        if entity_type == "function":
            func = await self.session.get(Function, entity_id)
            if func:
                module = await self.session.get(Module, func.module_id)
                file = await self.session.get(File, module.file_id) if module else None
                # Need to get repository - implement using proper repository pattern
                from src.database.repositories import RepositoryRepo

                repo_repo = RepositoryRepo(self.session)
                repo = await repo_repo.get_by_id(file.repository_id) if file else None

                return {
                    "type": "function",
                    "name": func.name,
                    "location": {
                        "repository": repo.name if repo else None,
                        "file": file.path if file else None,
                        "line": func.start_line,
                    },
                }

        elif entity_type == "class":
            cls = await self.session.get(Class, entity_id)
            if cls:
                module = await self.session.get(Module, cls.module_id)
                file = await self.session.get(File, module.file_id) if module else None
                from src.database.repositories import RepositoryRepo

                repo_repo = RepositoryRepo(self.session)
                repo = await repo_repo.get_by_id(file.repository_id) if file else None

                return {
                    "type": "class",
                    "name": cls.name,
                    "location": {
                        "repository": repo.name if repo else None,
                        "file": file.path if file else None,
                        "line": cls.start_line,
                    },
                }
        elif entity_type == "module":
            module = await self.session.get(Module, entity_id)
            if module:
                file = await self.session.get(File, module.file_id)
                from src.database.repositories import RepositoryRepo

                repo_repo = RepositoryRepo(self.session)
                repo = await repo_repo.get_by_id(file.repository_id) if file else None

                return {
                    "type": "module",
                    "name": module.name,
                    "location": {
                        "repository": repo.name if repo else None,
                        "file": file.path if file else None,
                        "line": 1,
                    },
                }

        return None

    async def _get_module_structure(
        self,
        module_path: str,
    ) -> dict[str, Any]:
        """Get structure of a single module."""
        # Use the instance session

        # Find the file
        from src.database.models import File

        result = await self.session.execute(
            select(File).where(File.path.like(f"%{module_path}")),
        )
        files = result.scalars().all()

        file = None
        for f in files:
            if f.path.endswith(module_path):
                file = f
                break

        if not file:
            return {"error": f"Module not found: {module_path}"}

        # Get module
        from src.database.models import Module

        result = await self.session.execute(
            select(Module).where(Module.file_id == file.id),
        )
        module = result.scalar_one_or_none()

        if not module:
            return {"error": f"Module data not found: {module_path}"}

        # Get structure
        structure = {
            "type": "module",
            "name": module.name,
            "path": module_path,
            "docstring": module.docstring,
            "imports": [],
            "classes": [],
            "functions": [],
        }

        # Get imports
        from src.database.models import Import

        result = await self.session.execute(
            select(Import)
            .where(Import.file_id == file.id)
            .order_by(Import.line_number),
        )
        for imp in result.scalars().all():
            structure["imports"].append(
                {
                    "statement": imp.import_statement,
                    "line": imp.line_number,
                },
            )

        # Get classes
        from src.database.models import Class

        result = await self.session.execute(
            select(Class)
            .where(Class.module_id == module.id)
            .order_by(Class.start_line),
        )
        for cls in result.scalars().all():
            # Get methods
            from sqlalchemy.orm import aliased

            from src.database.models import Function

            func_alias = aliased(Function)
            method_stmt = (
                select(func_alias.name)
                .where(func_alias.class_id == cls.id)
                .order_by(func_alias.start_line)
            )
            method_result = await self.session.execute(method_stmt)
            method_names = [name for (name,) in method_result.all()]

            structure["classes"].append(
                {
                    "name": cls.name,
                    "line": cls.start_line,
                    "methods": method_names,
                    "is_abstract": cls.is_abstract,
                },
            )

        # Get functions
        from src.database.models import Function

        func_result = await self.session.execute(
            select(Function)
            .where(and_(Function.module_id == module.id, Function.class_id.is_(None)))
            .order_by(Function.start_line),
        )
        for func in func_result.scalars().all():
            structure["functions"].append(
                {
                    "name": func.name,
                    "line": func.start_line,
                    "is_async": func.is_async,
                },
            )

        return structure

    async def _get_package_structure(
        self,
        package_path: str,
    ) -> dict[str, Any]:
        """Get structure of a package (directory)."""
        # This would need implementation to scan directory structure
        return {
            "type": "package",
            "path": package_path,
            "message": "Package structure analysis not yet implemented",
        }

    def _is_snake_case(self, name: str) -> bool:
        """Check if name follows snake_case convention."""
        return bool(re.match(r"^[a-z][a-z0-9_]*$", name))

    def _is_camel_case(self, name: str) -> bool:
        """Check if name follows CamelCase convention."""
        return bool(re.match(r"^[A-Z][a-zA-Z0-9]*$", name))

    def _to_snake_case(self, name: str) -> str:
        """Convert to snake_case."""
        # Simple conversion
        result = re.sub("([A-Z]+)", r"_\1", name)
        return result.lower().strip("_")

    def _to_camel_case(self, name: str) -> str:
        """Convert to CamelCase."""
        parts = name.split("_")
        return "".join(word.capitalize() for word in parts)
