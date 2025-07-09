"""Main entry point for the MCP Code Analysis Server."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import FastMCP

from src.database import init_database
from src.mcp_server.config import config, settings
from src.mcp_server.tools.analyze import AnalyzeTool
from src.mcp_server.tools.explain import ExplainTool
from src.mcp_server.tools.find import FindTool
from src.mcp_server.tools.repository import RepositoryTool
from src.mcp_server.tools.search import SearchTool
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


# Create tool instances
search_tool = SearchTool()
explain_tool = ExplainTool()
find_tool = FindTool()
analyze_tool = AnalyzeTool()
repository_tool = RepositoryTool()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting MCP Code Analysis Server")

    # Initialize database
    await init_database()

    yield

    # Shutdown
    logger.info("Shutting down MCP Code Analysis Server")


# Create FastAPI app
app = FastAPI(
    title="MCP Code Analysis Server",
    description="Intelligent code analysis and search capabilities via MCP",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.mcp.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create MCP server
mcp = FastMCP("Code Analysis MCP Server")


# Register MCP tools


@mcp.tool
async def search_code(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search for code by natural language query."""
    return await search_tool.search_code(query, limit)


@mcp.tool
async def explain_code(path: str) -> str:
    """
    Explain what a code element does (function, class, module, or package).
    For classes: aggregates explanations of all methods and attributes.
    For packages: provides overview of all modules and main components.
    """
    return await explain_tool.explain_code(path)


@mcp.tool
async def find_definition(name: str, type: str = "any") -> list[dict[str, Any]]:
    """
    Find where a function/class/module is defined.
    Type can be: 'function', 'class', 'module', or 'any'
    """
    return await find_tool.find_definition(name, type)


@mcp.tool
async def find_usage(
    function_or_class: str,
    repository: str | None = None,
) -> list[dict[str, Any]]:
    """
    Find all places where a function/class is used.
    Can search across all repositories or filter by specific repository.
    """
    return await find_tool.find_usage(function_or_class, repository)


@mcp.tool
async def analyze_dependencies(module_path: str) -> dict[str, Any]:
    """Analyze dependencies of a module."""
    return await analyze_tool.analyze_dependencies(module_path)


@mcp.tool
async def suggest_refactoring(code_path: str) -> list[dict[str, Any]]:
    """Suggest refactoring improvements."""
    return await analyze_tool.suggest_refactoring(code_path)


@mcp.tool
async def find_similar_code(code_snippet: str) -> list[dict[str, Any]]:
    """Find similar code patterns in the codebase."""
    return await analyze_tool.find_similar_code(code_snippet)


@mcp.tool
async def get_code_structure(path: str) -> dict[str, Any]:
    """Get hierarchical structure of a module/package."""
    return await analyze_tool.get_code_structure(path)


@mcp.tool
async def list_repositories() -> list[dict[str, Any]]:
    """List all monitored GitHub repositories with their sync status."""
    return await repository_tool.list_repositories()


@mcp.tool
async def sync_repository(repository_url: str) -> dict[str, Any]:
    """Manually trigger sync for a specific repository."""
    return await repository_tool.sync_repository(repository_url)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "mcp-code-analysis-server",
        "version": "0.1.0",
    }


# Mount MCP server to FastAPI
mcp_asgi = mcp.get_asgi_app()
app.mount("/mcp", mcp_asgi)


def main() -> None:
    """Main entry point."""
    # Set up logging
    setup_logging()

    # Run the server
    import uvicorn

    logger.info("Starting server on %s:%s", settings.mcp_host, settings.mcp_port)

    uvicorn.run(
        "src.mcp_server.main:app",
        host=settings.mcp_host,
        port=settings.mcp_port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
