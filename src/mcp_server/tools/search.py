"""Search tool for MCP server."""

from typing import Any

from src.query.ranking import ResultRanker
from src.query.search_engine import SearchEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SearchTool:
    """MCP tool for searching code."""

    def __init__(self) -> None:
        self.search_engine = SearchEngine()
        self.result_ranker = ResultRanker()

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
            # Perform search
            results = await self.search_engine.search(query=query, limit=limit)

            # Rank results
            ranked_results = self.result_ranker.rank_results(results, query)

            # Format for output
            return self.result_ranker.format_results(
                ranked_results,
                include_scores=False,
                include_context=True,
            )

        except Exception as e:
            logger.exception("Error in search_code: %s")
            return [{"error": str(e), "query": query}]
