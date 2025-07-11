"""Main entry point for the MCP Code Analysis Server."""

from typing import Any

from fastmcp import FastMCP

from src.config import settings
from src.database import get_session_factory, init_database
from src.logger import get_logger, setup_logging

logger = get_logger(__name__)

# Global database state
_engine = None
_session_factory = None


async def get_db_session():
    """Get database session for tools."""
    global _engine, _session_factory
    if _session_factory is None:
        _engine = await init_database()
        _session_factory = get_session_factory(_engine)
    return _session_factory()


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
        logger.exception("Error in search_code: %s", e)
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
        from src.query.explain import CodeExplainer

        async with await get_db_session() as session:
            explainer = CodeExplainer(session)
            explanation = await explainer.explain(path)

            if not explanation:
                return f"Code element not found: {path}"

            return explainer.format_explanation(explanation)
    except Exception as e:
        logger.exception("Error in explain_code: %s", e)
        return f"Error explaining code: {str(e)}"


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
            results = await finder.find_definition(name, file_path, entity_type)
            return finder.format_results(results)
    except Exception as e:
        logger.exception("Error in find_definition: %s", e)
        return [{"error": str(e)}]


@mcp.tool
async def analyze_dependencies(
    module_path: str,
    include_transitive: bool = False,
) -> dict[str, Any]:
    """
    Analyze dependencies of a module.

    Args:
        module_path: The path to the module (e.g., "src.utils.helpers")
        include_transitive: Whether to include transitive dependencies

    Returns:
        Dictionary containing direct imports and optionally transitive dependencies.
    """
    try:
        from src.query.dependency_analyzer import DependencyAnalyzer

        async with await get_db_session() as session:
            analyzer = DependencyAnalyzer(session)
            dependencies = await analyzer.analyze(module_path, include_transitive)
            return analyzer.format_results(dependencies)
    except Exception as e:
        logger.exception("Error in analyze_dependencies: %s", e)
        return {"error": str(e)}


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
            structure = await processor.get_file_structure(file)

            return structure
    except Exception as e:
        logger.exception("Error in get_code_structure: %s", e)
        return {"error": str(e)}


@mcp.tool
async def find_usage(
    name: str,
    entity_type: str | None = None,
    search_scope: str | None = None,
) -> list[dict[str, Any]]:
    """Find where a symbol is used."""
    try:
        # Placeholder implementation
        return [{"info": f"Usage search for '{name}' not yet implemented"}]
    except Exception as e:
        logger.exception("Error in find_usage: %s", e)
        return [{"error": str(e)}]


@mcp.tool
async def find_similar_code(code_snippet: str, limit: int = 10) -> list[dict[str, Any]]:
    """Find code similar to the given snippet."""
    try:
        # Placeholder implementation
        return [{"info": "Similar code search not yet implemented"}]
    except Exception as e:
        logger.exception("Error in find_similar_code: %s", e)
        return [{"error": str(e)}]


@mcp.tool
async def suggest_refactoring(
    file_path: str, focus_area: str | None = None
) -> list[dict[str, Any]]:
    """Suggest refactoring opportunities."""
    try:
        # Placeholder implementation
        return [
            {"info": f"Refactoring suggestions for '{file_path}' not yet implemented"}
        ]
    except Exception as e:
        logger.exception("Error in suggest_refactoring: %s", e)
        return [{"error": str(e)}]


@mcp.tool
async def list_repositories() -> list[dict[str, Any]]:
    """List all tracked repositories."""
    try:
        from sqlalchemy.future import select

        from src.database.models import Repository

        async with await get_db_session() as session:
            result = await session.execute(select(Repository))
            repositories = result.scalars().all()

            return [
                {
                    "id": repo.id,
                    "name": repo.name,
                    "owner": repo.owner,
                    "url": repo.github_url,
                    "branch": repo.default_branch,
                    "last_synced": (
                        repo.last_synced.isoformat() if repo.last_synced else None
                    ),
                }
                for repo in repositories
            ]
    except Exception as e:
        logger.exception("Error in list_repositories: %s", e)
        return [{"error": str(e)}]


@mcp.tool
async def sync_repository(repository_url: str) -> dict[str, Any]:
    """Manually trigger sync for a specific repository."""
    try:
        # Placeholder implementation
        return {"info": f"Repository sync for '{repository_url}' not yet implemented"}
    except Exception as e:
        logger.exception("Error in sync_repository: %s", e)
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
