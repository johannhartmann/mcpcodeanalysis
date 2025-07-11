"""Main entry point for the MCP Code Analysis Server."""

from typing import Any

from fastmcp import FastMCP

from src.config import settings
from src.database import get_session_factory, init_database
from src.logger import get_logger, setup_logging

logger = get_logger(__name__)


# Database connection manager
class DatabaseManager:
    """Manages database connections."""

    def __init__(self):
        self._engine = None
        self._session_factory = None

    async def get_session(self):
        """Get database session for tools."""
        if self._session_factory is None:
            self._engine = await init_database()
            self._session_factory = get_session_factory(self._engine)
        return self._session_factory()


_db_manager = DatabaseManager()


async def get_db_session():
    """Get database session for tools."""
    return await _db_manager.get_session()


# Create MCP server
mcp = FastMCP("Code Analysis Server")


# Database will be initialized on first use in get_db_session()


# Register MCP tools


@mcp.tool
async def search_code(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search for code by natural language query."""
    try:
        from src.query.ranking import ResultRanker
        from src.query.search_engine import SearchEngine

        async with await get_db_session() as session:
            search_engine = SearchEngine(session)
            results = await search_engine.search(query, limit)

            result_ranker = ResultRanker()
            ranked_results = result_ranker.rank_results(results, query)

            return result_ranker.format_results(
                ranked_results,
                include_scores=False,
                include_context=True,
            )
    except Exception as e:
        logger.exception("Error in search_code")
        return [{"error": str(e), "query": query}]


@mcp.tool
async def explain_code(path: str) -> str:
    """
    Explain what a code element does (function, class, module, or package).

    Args:
        path: The path to the code element (e.g., "src.utils.helpers.parse_json")

    Returns:
        A hierarchical explanation of the code element.
    """
    try:
        from src.mcp_server.tools.explain import ExplainTool

        tool = ExplainTool()
        return await tool.explain_code(path)
    except Exception as e:
        logger.exception("Error in explain_code")
        return f"Error explaining code: {e!s}"


@mcp.tool
async def find_definition(
    name: str,
    file_path: str | None = None,
    entity_type: str | None = None,
) -> list[dict[str, Any]]:
    """
    Find where a symbol is defined.

    Args:
        name: The name of the symbol to find
        file_path: Optional file path to search within
        entity_type: Optional type of entity to search for (function, class, module)

    Returns:
        List of locations where the symbol is defined.
    """
    try:
        from src.query.symbol_finder import SymbolFinder

        async with await get_db_session() as session:
            finder = SymbolFinder(session)
            return await finder.find_definitions(name, file_path, entity_type)
    except Exception as e:
        logger.exception("Error in find_definition")
        return [{"error": str(e)}]


@mcp.tool
async def get_code_structure(file_path: str) -> dict[str, Any]:
    """
    Get the structure of a code file.

    Args:
        file_path: The path to the file relative to the repository root

    Returns:
        Dictionary containing the file's structure (modules, classes, functions).
    """
    try:
        from sqlalchemy.future import select

        from src.database.models import File

        async with await get_db_session() as session:
            # Get file from database
            result = await session.execute(select(File).where(File.path == file_path))
            file = result.scalar_one_or_none()

            if not file:
                return {"error": f"File not found: {file_path}"}

            # Get code structure using CodeProcessor
            from src.scanner.code_processor import CodeProcessor

            processor = CodeProcessor(session)
            return await processor.get_file_structure(file)

    except Exception as e:
        logger.exception("Error in get_code_structure")
        return {"error": str(e)}


@mcp.tool
async def suggest_refactoring(
    file_path: str, focus_area: str | None = None
) -> list[dict[str, Any]]:
    """Suggest refactoring opportunities."""
    try:
        from src.mcp_server.tools.analyze import AnalyzeTool

        async with await get_db_session() as session:
            tool = AnalyzeTool(session)
            suggestions = await tool.suggest_refactoring(file_path)

            # If focus_area is provided, filter suggestions
            if focus_area and suggestions:
                # Filter suggestions based on focus area
                # For now, just mention the focus area in the response
                return [
                    {
                        "focus_area": focus_area,
                        "suggestions": suggestions,
                        "note": f"Focused on {focus_area} improvements",
                    }
                ]

            return suggestions
    except Exception as e:
        logger.exception("Error in suggest_refactoring")
        return [{"error": str(e)}]


@mcp.tool
async def sync_repository(repository_url: str) -> dict[str, Any]:
    """Manually trigger sync for a specific repository."""
    try:
        from src.mcp_server.tools.repository import RepositoryTool

        async with await get_db_session() as session:
            tool = RepositoryTool(session)
            return await tool.sync_repository(repository_url)
    except Exception as e:
        logger.exception("Error in sync_repository")
        return {"error": str(e)}


def main() -> None:
    """Main entry point."""
    # Set up logging
    setup_logging()

    logger.info("Starting MCP server on %s:%s", settings.mcp.host, settings.mcp.port)

    # Run the MCP server with HTTP transport
    mcp.run(
        transport="http",
        host=settings.mcp.host,
        port=settings.mcp.port,
    )


if __name__ == "__main__":
    main()
