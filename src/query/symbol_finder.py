"""Symbol finder for locating code definitions."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Class, File, Function, Module
from src.logger import get_logger

logger = get_logger(__name__)


class SymbolFinder:
    """Find symbols (functions, classes, modules) in the codebase."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def find_definitions(
        self,
        name: str,
        file_path: str | None = None,
        entity_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find where a symbol is defined.

        Args:
            name: Symbol name to search for
            file_path: Optional file path to restrict search
            entity_type: Optional entity type (function, class, module)

        Returns:
            List of definition locations
        """
        results = []

        # Search functions
        if not entity_type or entity_type == "function":
            stmt = (
                select(Function, Module, File)
                .join(Module, Function.module_id == Module.id)
                .join(File, Module.file_id == File.id)
            )

            stmt = stmt.where(Function.name == name)

            if file_path:
                stmt = stmt.where(File.path.like(f"%{file_path}%"))

            result = await self.session.execute(stmt)
            for func, module, file in result:
                results.append(
                    {
                        "type": "function",
                        "name": func.name,
                        "file_path": file.path,
                        "module_name": module.name,
                        "start_line": func.start_line,
                        "end_line": func.end_line,
                        "is_async": func.is_async,
                        "parameters": func.parameters,
                        "return_type": func.return_type,
                        "docstring": func.docstring,
                    }
                )

        # Search classes
        if not entity_type or entity_type == "class":
            stmt = (
                select(Class, Module, File)
                .join(Module, Class.module_id == Module.id)
                .join(File, Module.file_id == File.id)
            )

            stmt = stmt.where(Class.name == name)

            if file_path:
                stmt = stmt.where(File.path.like(f"%{file_path}%"))

            result = await self.session.execute(stmt)
            for cls, module, file in result:
                results.append(
                    {
                        "type": "class",
                        "name": cls.name,
                        "file_path": file.path,
                        "module_name": module.name,
                        "start_line": cls.start_line,
                        "end_line": cls.end_line,
                        "base_classes": cls.base_classes,
                        "is_abstract": cls.is_abstract,
                        "docstring": cls.docstring,
                    }
                )

        # Search modules
        if not entity_type or entity_type == "module":
            stmt = select(Module, File).join(File, Module.file_id == File.id)

            stmt = stmt.where(Module.name == name)

            if file_path:
                stmt = stmt.where(File.path.like(f"%{file_path}%"))

            result = await self.session.execute(stmt)
            for module, file in result:
                results.append(
                    {
                        "type": "module",
                        "name": module.name,
                        "file_path": file.path,
                        "module_name": module.name,
                        "start_line": module.start_line,
                        "end_line": module.end_line,
                        "docstring": module.docstring,
                    }
                )

        return results

    async def find_by_partial_name(
        self,
        partial_name: str,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Find symbols by partial name match.

        Args:
            partial_name: Partial name to search for
            entity_type: Optional entity type filter
            limit: Maximum results to return

        Returns:
            List of matching symbols
        """
        results = []
        partial_lower = partial_name.lower()

        # Search functions
        if not entity_type or entity_type == "function":
            stmt = (
                select(Function, Module, File)
                .join(Module, Function.module_id == Module.id)
                .join(File, Module.file_id == File.id)
            )

            stmt = stmt.where(Function.name.ilike(f"%{partial_name}%")).limit(limit)

            result = await self.session.execute(stmt)
            for func, module, file in result:
                results.append(
                    {
                        "type": "function",
                        "name": func.name,
                        "file_path": file.path,
                        "module_name": module.name,
                        "match_score": self._calculate_match_score(
                            func.name, partial_lower
                        ),
                    }
                )

        # Search classes
        if not entity_type or entity_type == "class":
            remaining = limit - len(results)
            if remaining > 0:
                stmt = (
                    select(Class, Module, File)
                    .join(Module, Class.module_id == Module.id)
                    .join(File, Module.file_id == File.id)
                )

                stmt = stmt.where(Class.name.ilike(f"%{partial_name}%")).limit(
                    remaining
                )

                result = await self.session.execute(stmt)
                for cls, module, file in result:
                    results.append(
                        {
                            "type": "class",
                            "name": cls.name,
                            "file_path": file.path,
                            "module_name": module.name,
                            "match_score": self._calculate_match_score(
                                cls.name, partial_lower
                            ),
                        }
                    )

        # Sort by match score
        results.sort(key=lambda x: x["match_score"], reverse=True)

        return results[:limit]

    def _calculate_match_score(self, name: str, query: str) -> float:
        """Calculate match score for ranking results."""
        name_lower = name.lower()

        # Exact match
        if name_lower == query:
            return 1.0

        # Starts with query
        if name_lower.startswith(query):
            return 0.8

        # Contains query
        if query in name_lower:
            return 0.6

        # Default score for partial matches
        return 0.4
