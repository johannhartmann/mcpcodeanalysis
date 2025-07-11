"""Search engine for code queries."""

import time
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database.models import Class, File, Function, Module
from src.indexer.embeddings import EmbeddingGenerator
from src.logger import get_logger

logger = get_logger(__name__)


class SearchEngine:
    """Search engine for code entities using embeddings."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.embedding_generator = EmbeddingGenerator()
        self.query_config = settings.query

    async def search(
        self,
        query: str,
        limit: int | None = None,
        entity_types: list[str] | None = None,
        repository_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for code entities matching a query."""
        start_time = time.time()

        # Set default limit
        if limit is None:
            limit = self.query_config.default_limit
        limit = min(limit, self.query_config.max_limit)

        try:
            # For now, implement keyword-based search until embedding search is fully implemented
            results = await self._keyword_search(
                query, limit, entity_types, repository_filter
            )

            # Log search statistics
            execution_time = (time.time() - start_time) * 1000
            logger.info(
                "Search completed: query='%s', results=%d, time=%.2fms",
                query,
                len(results),
                execution_time,
            )

            return results

        except Exception:
            logger.exception("Search failed")
            return []

    async def _keyword_search(
        self,
        query: str,
        limit: int,
        entity_types: list[str] | None = None,
        repository_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform keyword-based search as fallback."""
        results = []
        query_lower = query.lower()

        # Search functions
        if not entity_types or "function" in entity_types:
            stmt = select(Function, File, Module).select_from(
                Function.__table__.join(
                    Module.__table__, Function.module_id == Module.id
                ).join(File.__table__, Module.file_id == File.id)
            )
            if repository_filter:
                stmt = stmt.where(File.path.like(f"%{repository_filter}%"))

            result = await self.session.execute(stmt.limit(limit))
            functions = result.all()

            for func, file, module in functions:
                if query_lower in func.name.lower() or (
                    func.docstring and query_lower in func.docstring.lower()
                ):
                    results.append(
                        {
                            "entity_type": "function",
                            "entity_id": func.id,
                            "name": func.name,
                            "file_path": file.path,
                            "module_name": module.name,
                            "docstring": func.docstring,
                            "start_line": func.start_line,
                            "end_line": func.end_line,
                            "similarity": 0.8,  # Placeholder similarity score
                        }
                    )

        # Search classes
        if not entity_types or "class" in entity_types:
            stmt = select(Class, File, Module).select_from(
                Class.__table__.join(
                    Module.__table__, Class.module_id == Module.id
                ).join(File.__table__, Module.file_id == File.id)
            )
            if repository_filter:
                stmt = stmt.where(File.path.like(f"%{repository_filter}%"))

            result = await self.session.execute(stmt.limit(limit))
            classes = result.all()

            for cls, file, module in classes:
                if query_lower in cls.name.lower() or (
                    cls.docstring and query_lower in cls.docstring.lower()
                ):
                    results.append(
                        {
                            "entity_type": "class",
                            "entity_id": cls.id,
                            "name": cls.name,
                            "file_path": file.path,
                            "module_name": module.name,
                            "docstring": cls.docstring,
                            "start_line": cls.start_line,
                            "end_line": cls.end_line,
                            "similarity": 0.8,  # Placeholder similarity score
                        }
                    )

        # Search modules
        if not entity_types or "module" in entity_types:
            stmt = select(Module, File).select_from(
                Module.__table__.join(File.__table__, Module.file_id == File.id)
            )
            if repository_filter:
                stmt = stmt.where(File.path.like(f"%{repository_filter}%"))

            result = await self.session.execute(stmt.limit(limit))
            modules = result.all()

            for module, file in modules:
                if query_lower in module.name.lower() or (
                    module.docstring and query_lower in module.docstring.lower()
                ):
                    results.append(
                        {
                            "entity_type": "module",
                            "entity_id": module.id,
                            "name": module.name,
                            "file_path": file.path,
                            "module_name": module.name,
                            "docstring": module.docstring,
                            "similarity": 0.8,  # Placeholder similarity score
                        }
                    )

        # Sort by relevance (for now just by name similarity)
        results.sort(key=lambda x: x["name"].lower().find(query_lower))

        return results[:limit]

    async def search_semantic(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7,
        entity_types: list[str] | None = None,
        repository_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search code entities using semantic similarity with embeddings."""
        # Generate embedding for query
        try:
            query_embedding = await self.embedding_generator.generate_embedding(query)
        except Exception:
            logger.exception("Failed to generate query embedding")
            # Fall back to keyword search
            return await self.search(query, limit, entity_types, repository_filter)

        # Search for similar embeddings in the database
        from sqlalchemy import text

        # Build the query based on entity types
        if not entity_types:
            entity_types = ["module", "class", "function", "method"]

        results = []

        # Query for similar embeddings using pgvector
        sql = text(
            """
            SELECT
                e.entity_type,
                e.entity_id,
                e.file_id,
                1 - (e.embedding <=> cast(:query_embedding as vector)) as similarity
            FROM code_embeddings e
            WHERE
                e.entity_type = ANY(:entity_types)
                AND 1 - (e.embedding <=> cast(:query_embedding as vector)) > :threshold
            ORDER BY similarity DESC
            LIMIT :limit
        """
        )

        result = await self.session.execute(
            sql,
            {
                "query_embedding": str(query_embedding),
                "entity_types": entity_types,
                "threshold": threshold,
                "limit": limit,
            },
        )

        for row in result:
            entity_type = row.entity_type
            entity_id = row.entity_id
            similarity = row.similarity

            # Fetch entity details based on type
            if entity_type in {"function", "method"}:
                entity_result = await self.session.execute(
                    select(Function, Module, File)
                    .select_from(
                        Function.__table__.join(
                            Module.__table__, Function.module_id == Module.id
                        ).join(File.__table__, Module.file_id == File.id)
                    )
                    .where(Function.id == entity_id)
                )
                entity_row = entity_result.first()
                if entity_row:
                    func, module, file = entity_row
                    results.append(
                        {
                            "entity_type": entity_type,
                            "entity_id": func.id,
                            "name": func.name,
                            "file_path": file.path,
                            "module_name": module.name,
                            "docstring": func.docstring,
                            "start_line": func.start_line,
                            "end_line": func.end_line,
                            "similarity": similarity,
                        }
                    )
            elif entity_type == "class":
                entity_result = await self.session.execute(
                    select(Class, Module, File)
                    .select_from(
                        Class.__table__.join(
                            Module.__table__, Class.module_id == Module.id
                        ).join(File.__table__, Module.file_id == File.id)
                    )
                    .where(Class.id == entity_id)
                )
                entity_row = entity_result.first()
                if entity_row:
                    cls, module, file = entity_row
                    results.append(
                        {
                            "entity_type": entity_type,
                            "entity_id": cls.id,
                            "name": cls.name,
                            "file_path": file.path,
                            "module_name": module.name,
                            "docstring": cls.docstring,
                            "start_line": cls.start_line,
                            "end_line": cls.end_line,
                            "similarity": similarity,
                        }
                    )
            elif entity_type == "module":
                entity_result = await self.session.execute(
                    select(Module, File)
                    .select_from(
                        Module.__table__.join(File.__table__, Module.file_id == File.id)
                    )
                    .where(Module.id == entity_id)
                )
                entity_row = entity_result.first()
                if entity_row:
                    module, file = entity_row
                    results.append(
                        {
                            "entity_type": entity_type,
                            "entity_id": module.id,
                            "name": module.name,
                            "file_path": file.path,
                            "module_name": module.name,
                            "docstring": module.docstring,
                            "similarity": similarity,
                        }
                    )

        return results

    async def search_similar_code(
        self,
        code_snippet: str,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Search for similar code patterns."""
        try:
            # Generate embedding for the code snippet
            snippet_embedding = await self.embedding_generator.generate_embedding(
                code_snippet
            )

            # Search for similar embeddings
            query = text(
                """
                SELECT
                    e.entity_type,
                    e.entity_id,
                    e.file_id,
                    e.content,
                    1 - (e.embedding <=> cast(:query_embedding as vector)) as similarity
                FROM code_embeddings e
                WHERE
                    e.embedding_type = 'raw'
                    AND 1 - (e.embedding <=> cast(:query_embedding as vector)) > :threshold
                ORDER BY similarity DESC
                LIMIT :limit
                """
            )

            result = await self.session.execute(
                query,
                {
                    "query_embedding": str(snippet_embedding),
                    "threshold": threshold,
                    "limit": limit,
                },
            )

            similar_code = []
            for row in result:
                # Get entity details
                entity_details = await self._get_entity_details(
                    row.entity_type, row.entity_id
                )

                if entity_details:
                    similar_code.append(
                        {
                            "entity_type": row.entity_type,
                            "entity_id": row.entity_id,
                            "name": entity_details.get("name", "Unknown"),
                            "file_path": entity_details.get("file_path", "Unknown"),
                            "similarity": row.similarity,
                            "content_preview": (
                                row.content[:200] + "..."
                                if len(row.content) > 200
                                else row.content
                            ),
                        }
                    )

            return similar_code

        except Exception:
            logger.exception("Failed to search similar code")
            return []

    async def get_code_context(
        self,
        entity_type: str,
        entity_id: int,
        include_dependencies: bool = False,
    ) -> dict[str, Any]:
        """Get context information for a code entity."""
        try:
            if entity_type == "function":
                return await self._get_function_context(entity_id, include_dependencies)
            if entity_type == "class":
                return await self._get_class_context(entity_id, include_dependencies)
            if entity_type == "module":
                return await self._get_module_context(entity_id, include_dependencies)
            return {"error": f"Unknown entity type: {entity_type}"}
        except Exception as e:
            logger.exception("Failed to get context for %s %d", entity_type, entity_id)
            return {"error": str(e)}

    async def _get_function_context(
        self, function_id: int, _include_dependencies: bool
    ) -> dict[str, Any]:
        """Get context for a function."""
        result = await self.session.execute(
            select(Function, Module, File)
            .join(Module)
            .join(File)
            .where(Function.id == function_id)
        )
        row = result.first()

        if not row:
            return {"error": "Function not found"}

        func, module, file = row
        context = {
            "type": "function",
            "name": func.name,
            "module": module.name,
            "file": file.path,
            "docstring": func.docstring,
            "parameters": func.parameters,
            "return_type": func.return_type,
            "start_line": func.start_line,
            "end_line": func.end_line,
        }

        if _include_dependencies:
            # Add related functions, classes, etc.
            # This would need more complex implementation
            context["dependencies"] = []

        return context

    async def _get_class_context(
        self, class_id: int, _include_dependencies: bool
    ) -> dict[str, Any]:
        """Get context for a class."""
        result = await self.session.execute(
            select(Class, Module, File)
            .join(Module)
            .join(File)
            .where(Class.id == class_id)
        )
        row = result.first()

        if not row:
            return {"error": "Class not found"}

        cls, module, file = row

        # Get methods
        methods_result = await self.session.execute(
            select(Function).where(Function.class_id == class_id)
        )
        methods = methods_result.scalars().all()

        return {
            "type": "class",
            "name": cls.name,
            "module": module.name,
            "file": file.path,
            "docstring": cls.docstring,
            "base_classes": cls.base_classes,
            "start_line": cls.start_line,
            "end_line": cls.end_line,
            "methods": [
                {
                    "name": method.name,
                    "docstring": method.docstring,
                    "parameters": method.parameters,
                    "is_property": method.is_property,
                    "is_static": method.is_static,
                    "is_classmethod": method.is_classmethod,
                }
                for method in methods
            ],
        }

    async def _get_module_context(
        self, module_id: int, _include_dependencies: bool
    ) -> dict[str, Any]:
        """Get context for a module."""
        result = await self.session.execute(
            select(Module, File).join(File).where(Module.id == module_id)
        )
        row = result.first()

        if not row:
            return {"error": "Module not found"}

        module, file = row

        # Get classes and functions
        classes_result = await self.session.execute(
            select(Class).where(Class.module_id == module_id)
        )
        classes = classes_result.scalars().all()

        functions_result = await self.session.execute(
            select(Function).where(
                Function.module_id == module_id,
                Function.class_id.is_(None),  # Only module-level functions
            )
        )
        functions = functions_result.scalars().all()

        return {
            "type": "module",
            "name": module.name,
            "file": file.path,
            "docstring": module.docstring,
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
                    "parameters": func.parameters,
                }
                for func in functions
            ],
        }

    async def _get_entity_details(
        self, entity_type: str, entity_id: int
    ) -> dict[str, Any] | None:
        """Get details for a specific entity."""
        try:
            if entity_type in {"function", "method"}:
                result = await self.session.execute(
                    select(Function, Module, File)
                    .join(Module, Function.module_id == Module.id)
                    .join(File, Module.file_id == File.id)
                    .where(Function.id == entity_id)
                )
                row = result.first()
                if row:
                    func, module, file = row
                    return {
                        "name": func.name,
                        "file_path": file.path,
                        "module_name": module.name,
                        "start_line": func.start_line,
                        "end_line": func.end_line,
                    }
            elif entity_type == "class":
                result = await self.session.execute(
                    select(Class, Module, File)
                    .join(Module, Class.module_id == Module.id)
                    .join(File, Module.file_id == File.id)
                    .where(Class.id == entity_id)
                )
                row = result.first()
                if row:
                    cls, module, file = row
                    return {
                        "name": cls.name,
                        "file_path": file.path,
                        "module_name": module.name,
                        "start_line": cls.start_line,
                        "end_line": cls.end_line,
                    }
            elif entity_type == "module":
                result = await self.session.execute(
                    select(Module, File)
                    .join(File, Module.file_id == File.id)
                    .where(Module.id == entity_id)
                )
                row = result.first()
                if row:
                    module, file = row
                    return {
                        "name": module.name,
                        "file_path": file.path,
                        "module_name": module.name,
                        "start_line": module.start_line,
                        "end_line": module.end_line,
                    }
        except Exception:
            logger.exception("Failed to get entity details")

        return None
