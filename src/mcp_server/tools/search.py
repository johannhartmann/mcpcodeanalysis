"""Search tool for MCP server."""

from typing import Any

from src.database import get_session_factory, init_database
from src.logger import get_logger
from src.query.ranking import ResultRanker
from src.query.search_engine import SearchEngine

logger = get_logger(__name__)


class SearchTool:
    """MCP tool for searching code."""

    def __init__(self) -> None:
        self.result_ranker = ResultRanker()
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
                search_engine = SearchEngine(session)

                # Try semantic search first
                results = await search_engine.search_semantic(query, limit)

                # If no semantic results, fall back to keyword search
                if not results:
                    logger.info(
                        "No semantic search results, falling back to keyword search"
                    )
                    results = await search_engine.search(query, limit)

                # Rank results
                ranked_results = self.result_ranker.rank_results(results, query)

                # Format for output
                return self.result_ranker.format_results(
                    ranked_results,
                    include_scores=False,
                    include_context=True,
                )

        except Exception as e:
            logger.exception("Error in search_code: %s", e)
            return [{"error": str(e), "query": query}]
