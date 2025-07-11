"""Vector similarity search for code embeddings."""

from enum import Enum
from typing import Any

import numpy as np
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.config import settings
from src.database.models import (
    Class,
    CodeEmbedding,
    File,
    Function,
    Module,
)
from src.logger import get_logger

logger = get_logger(__name__)


class SearchScope(Enum):
    """Scope for vector search."""

    ALL = "all"
    REPOSITORY = "repository"
    FILE = "file"
    FUNCTIONS = "functions"
    CLASSES = "classes"
    MODULES = "modules"


class VectorSearch:
    """Perform vector similarity search on code embeddings."""

    def __init__(
        self,
        db_session: AsyncSession,
    ) -> None:
        """Initialize vector search.

        Args:
            db_session: Database session
        """
        self.db_session = db_session
        # settings imported globally from src.config
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key.get_secret_value(),
            model=settings.embeddings.model,
        )

    async def search(
        self,
        query: str,
        scope: SearchScope = SearchScope.ALL,
        repository_id: int | None = None,
        file_id: int | None = None,
        limit: int = 10,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search for code entities similar to query.

        Args:
            query: Search query text
            scope: Search scope
            repository_id: Optional repository to limit search
            file_id: Optional file to limit search
            limit: Maximum number of results
            threshold: Optional similarity threshold (0-1)

        Returns:
            List of search results with similarity scores
        """
        logger.info(
            "Searching for '%s' in scope %s (repo=%s, file=%s)",
            query,
            scope.value,
            repository_id,
            file_id,
        )

        # Generate query embedding
        query_embedding = await self.embeddings.aembed_query(query)

        # Build search query
        search_query = self._build_search_query(
            query_embedding,
            scope,
            repository_id,
            file_id,
            limit,
            threshold,
        )

        # Execute search
        result = await self.db_session.execute(search_query)
        rows = result.fetchall()

        # Format results
        results = []
        for row in rows:
            embedding_id, similarity, entity_type = row[0], row[1], row[2]

            # Load full embedding with entity
            embedding = await self._load_embedding_with_entity(embedding_id)
            if embedding:
                results.append(
                    {
                        "embedding_id": embedding_id,
                        "similarity": float(similarity),
                        "entity_type": entity_type,
                        "entity": await self._format_entity(embedding),
                        "text": embedding.content,
                        "metadata": embedding.repo_metadata,
                    },
                )

        logger.info("Found %d results for query '%s'", len(results), query)

        return results

    async def search_similar(
        self,
        embedding_id: int,
        *,
        limit: int = 10,
        exclude_same_file: bool = False,
    ) -> list[dict[str, Any]]:
        """Find entities similar to a given embedding.

        Args:
            embedding_id: ID of embedding to find similar to
            limit: Maximum number of results
            exclude_same_file: Whether to exclude results from same file

        Returns:
            List of similar entities
        """
        # Get source embedding
        result = await self.db_session.execute(
            select(CodeEmbedding).where(CodeEmbedding.id == embedding_id),
        )
        source_embedding = result.scalar_one_or_none()

        if not source_embedding:
            msg = f"Embedding {embedding_id} not found"
            raise ValueError(msg)

        # Build similarity query
        query = select(
            CodeEmbedding.id,
            CodeEmbedding.embedding.cosine_distance(source_embedding.embedding).label(
                "distance",
            ),
            CodeEmbedding.entity_type,
        ).where(CodeEmbedding.id != embedding_id)

        if exclude_same_file and source_embedding.file_id:
            query = query.where(CodeEmbedding.file_id != source_embedding.file_id)

        query = query.order_by("distance").limit(limit)

        # Execute search
        result = await self.db_session.execute(query)
        rows = result.fetchall()

        # Format results
        results = []
        for row in rows:
            similar_id, distance, entity_type = row[0], row[1], row[2]
            similarity = 1.0 - float(distance)  # Convert distance to similarity

            embedding = await self._load_embedding_with_entity(similar_id)
            if embedding:
                results.append(
                    {
                        "embedding_id": similar_id,
                        "similarity": similarity,
                        "entity_type": entity_type,
                        "entity": await self._format_entity(embedding),
                        "text": embedding.content,
                        "metadata": embedding.repo_metadata,
                    },
                )

        return results

    async def search_by_code(
        self,
        code_snippet: str,
        scope: SearchScope = SearchScope.ALL,
        repository_id: int | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for entities similar to a code snippet.

        Args:
            code_snippet: Code snippet to search for
            scope: Search scope
            repository_id: Optional repository to limit search
            limit: Maximum number of results

        Returns:
            List of search results
        """
        # Format code snippet as query
        query = f"Code snippet:\n{code_snippet}"

        return await self.search(
            query,
            scope=scope,
            repository_id=repository_id,
            limit=limit,
        )

    def _build_search_query(
        self,
        query_embedding: list[float],
        scope: SearchScope,
        repository_id: int | None,
        file_id: int | None,
        limit: int,
        threshold: float | None,
    ):
        """Build the SQL query for vector search.

        Args:
            query_embedding: Query embedding vector
            scope: Search scope
            repository_id: Optional repository filter
            file_id: Optional file filter
            limit: Result limit
            threshold: Optional similarity threshold

        Returns:
            SQLAlchemy query
        """
        # Convert to numpy array for pgvector
        query_vector = np.array(query_embedding)

        # Base query with cosine similarity
        query = select(
            CodeEmbedding.id,
            (1 - CodeEmbedding.embedding.cosine_distance(query_vector)).label(
                "similarity",
            ),
            CodeEmbedding.entity_type,
        )

        # Apply scope filters
        if scope == SearchScope.FUNCTIONS:
            query = query.where(CodeEmbedding.entity_type == "function")
        elif scope == SearchScope.CLASSES:
            query = query.where(CodeEmbedding.entity_type == "class")
        elif scope == SearchScope.MODULES:
            query = query.where(CodeEmbedding.entity_type == "module")

        # Apply repository filter
        if repository_id:
            query = query.join(File, CodeEmbedding.file_id == File.id).where(
                File.repository_id == repository_id,
            )

        # Apply file filter
        if file_id:
            query = query.where(CodeEmbedding.file_id == file_id)

        # Apply similarity threshold
        if threshold:
            query = query.where(
                (1 - CodeEmbedding.embedding.cosine_distance(query_vector))
                >= threshold,
            )

        # Order by similarity and limit
        return query.order_by(text("similarity DESC")).limit(limit)

    async def _load_embedding_with_entity(
        self,
        embedding_id: int,
    ) -> CodeEmbedding | None:
        """Load embedding with its associated entity.

        Args:
            embedding_id: Embedding ID

        Returns:
            CodeEmbedding with loaded relationships
        """
        result = await self.db_session.execute(
            select(CodeEmbedding)
            .where(CodeEmbedding.id == embedding_id)
            .options(
                selectinload(CodeEmbedding.file).selectinload(File.repository),
            ),
        )
        return result.scalar_one_or_none()

    async def _format_entity(self, embedding: CodeEmbedding) -> dict[str, Any]:
        """Format entity information from embedding.

        Args:
            embedding: CodeEmbedding record

        Returns:
            Formatted entity information
        """
        entity_info = {
            "id": embedding.entity_id,
            "type": embedding.entity_type,
            "file_path": embedding.file.path if embedding.file else None,
            "repository": (
                embedding.file.repository.name
                if embedding.file and embedding.file.repository
                else None
            ),
        }

        # Load specific entity details
        if embedding.entity_type == "function":
            result = await self.db_session.execute(
                select(Function)
                .where(Function.id == embedding.entity_id)
                .options(selectinload(Function.parent_class)),
            )
            func = result.scalar_one_or_none()
            if func:
                entity_info.update(
                    {
                        "name": func.name,
                        "class_name": (
                            func.parent_class.name if func.parent_class else None
                        ),
                        "start_line": func.start_line,
                        "end_line": func.end_line,
                        "is_async": func.is_async,
                        "parameters": func.parameters,
                        "return_type": func.return_type,
                    },
                )

        elif embedding.entity_type == "class":
            result = await self.db_session.execute(
                select(Class).where(Class.id == embedding.entity_id),
            )
            cls = result.scalar_one_or_none()
            if cls:
                entity_info.update(
                    {
                        "name": cls.name,
                        "start_line": cls.start_line,
                        "end_line": cls.end_line,
                        "base_classes": cls.base_classes,
                        "is_abstract": cls.is_abstract,
                    },
                )

        elif embedding.entity_type == "module":
            result = await self.db_session.execute(
                select(Module).where(Module.id == embedding.entity_id),
            )
            module = result.scalar_one_or_none()
            if module:
                entity_info.update(
                    {
                        "name": module.name,
                        "start_line": module.start_line,
                        "end_line": module.end_line,
                    },
                )

        return entity_info

    async def get_repository_stats(self, repository_id: int) -> dict[str, Any]:
        """Get embedding statistics for a repository.

        Args:
            repository_id: Repository ID

        Returns:
            Statistics dictionary
        """
        # Count embeddings by type
        result = await self.db_session.execute(
            select(
                CodeEmbedding.entity_type,
                func.count(CodeEmbedding.id).label("count"),
            )
            .join(File, CodeEmbedding.file_id == File.id)
            .where(File.repository_id == repository_id)
            .group_by(CodeEmbedding.entity_type),
        )

        counts = {row[0]: row[1] for row in result}

        # Get total and file count
        total_result = await self.db_session.execute(
            select(func.count(CodeEmbedding.id))
            .join(File, CodeEmbedding.file_id == File.id)
            .where(File.repository_id == repository_id),
        )
        total = total_result.scalar() or 0

        file_result = await self.db_session.execute(
            select(func.count(func.distinct(CodeEmbedding.file_id)))
            .join(File, CodeEmbedding.file_id == File.id)
            .where(File.repository_id == repository_id),
        )
        file_count = file_result.scalar() or 0

        return {
            "repository_id": repository_id,
            "total_embeddings": total,
            "files_with_embeddings": file_count,
            "embeddings_by_type": {
                "functions": counts.get("function", 0),
                "classes": counts.get("class", 0),
                "modules": counts.get("module", 0),
            },
        }
