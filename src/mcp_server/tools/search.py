"""Search tool for MCP server."""

from typing import Any

from src.database import get_session_factory, init_database
from src.embeddings.vector_search import SearchScope, VectorSearch
from src.logger import get_logger

logger = get_logger(__name__)


class SearchTool:
    """MCP tool for searching code."""

    def __init__(self) -> None:
        self._engine = None
        self._session_factory = None

    async def _get_session_factory(self):
        """Get database session factory, initializing if needed."""
        if self._session_factory is None:
            self._engine = await init_database()
            self._session_factory = get_session_factory(self._engine)
        return self._session_factory

    async def search_code(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Search for code by natural language query.

        Args:
            query: Natural language search query
            limit: Maximum number of results to return (default: 10)

        Returns:
            List of search results with code entities matching the query
        """
        try:
            session_factory = await self._get_session_factory()
            async with session_factory() as session:
                vector_search = VectorSearch(session)

                # Use proper vector search with embeddings
                results = await vector_search.search(
                    query=query,
                    scope=SearchScope.ALL,
                    limit=limit,
                    threshold=0.3,  # Lower threshold for better results
                )

                # Format results for MCP output
                formatted_results = []
                for result in results:
                    entity = result.get("entity", {})
                    formatted_results.append(
                        {
                            "name": entity.get("name", "Unknown"),
                            "type": entity.get("type", result.get("entity_type")),
                            "file_path": entity.get("file_path", ""),
                            "repository": entity.get("repository", ""),
                            "start_line": entity.get("start_line"),
                            "end_line": entity.get("end_line"),
                            "similarity_score": result.get("similarity", 0),
                            "content": result.get("text", ""),
                            "class_name": entity.get("class_name"),
                            "is_async": entity.get("is_async"),
                            "parameters": entity.get("parameters"),
                            "return_type": entity.get("return_type"),
                            "base_classes": entity.get("base_classes"),
                            "is_abstract": entity.get("is_abstract"),
                        }
                    )

                return formatted_results

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.exception("Error in search_code")
            return [{"error": str(e), "query": query}]
