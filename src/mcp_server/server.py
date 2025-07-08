"""MCP Code Analysis Server implementation."""

import asyncio
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP

from src.database.init_db import get_session_factory, init_database
from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.openai_client import OpenAIClient
from src.mcp_server.config import get_settings
from src.mcp_server.tools.analysis_tools import AnalysisTools
from src.mcp_server.tools.code_analysis import CodeAnalysisTools
from src.mcp_server.tools.code_search import CodeSearchTools
from src.mcp_server.tools.domain_tools import DomainTools
from src.mcp_server.tools.repository_management import RepositoryManagementTools
from src.scanner.repository_scanner import RepositoryScanner
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

logger = get_logger(__name__)


class MCPCodeAnalysisServer:
    """Main MCP server for code analysis."""

    def __init__(self) -> None:
        """Initialize MCP server."""
        self.settings = get_settings()
        self.mcp = FastMCP("Code Analysis Server")
        self.engine: AsyncEngine | None = None
        self.session_factory = None
        self.openai_client: OpenAIClient | None = None
        self._initialized = False

        # Tool instances (initialized on startup)
        self.code_search_tools: CodeSearchTools | None = None
        self.code_analysis_tools: CodeAnalysisTools | None = None
        self.repo_management_tools: RepositoryManagementTools | None = None
        self.domain_tools: DomainTools | None = None
        self.analysis_tools: AnalysisTools | None = None

    async def initialize(self) -> None:
        """Initialize server resources."""
        if self._initialized:
            return

        logger.info("Starting MCP Code Analysis Server")

        try:
            # Initialize database
            logger.info("Initializing database connection")
            self.engine = await init_database()
            self.session_factory = get_session_factory(self.engine)

            # Initialize OpenAI client
            logger.info("Initializing OpenAI client")
            self.openai_client = OpenAIClient()

            # Test OpenAI connection
            if not await self.openai_client.test_connection():
                logger.warning(
                    "OpenAI connection test failed - embeddings will be disabled",
                )
                self.openai_client = None

            # Initialize and register tools
            await self._register_tools()

            # Log configuration
            logger.info(
                f"Server initialized with {len(self.settings.repositories)} repositories",
            )
            logger.info(f"Scanner storage path: {self.settings.scanner.storage_path}")
            logger.info(f"Embedding model: {self.settings.embeddings.model}")

            self._initialized = True
            logger.info("MCP Code Analysis Server started successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize server: {e}")
            raise

    async def _register_tools(self) -> None:
        """Register all MCP tools."""
        logger.info("Registering MCP tools")

        # Get a session for tool initialization
        async with self.session_factory() as session:
            # Initialize code search tools
            self.code_search_tools = CodeSearchTools(
                session, self.openai_client, self.mcp,
            )
            await self.code_search_tools.register_tools()

            # Initialize code analysis tools
            self.code_analysis_tools = CodeAnalysisTools(session, self.mcp)
            await self.code_analysis_tools.register_tools()

            # Initialize repository management tools
            self.repo_management_tools = RepositoryManagementTools(
                session, self.openai_client, self.mcp,
            )
            await self.repo_management_tools.register_tools()
            
            # Initialize domain-driven design tools
            self.domain_tools = DomainTools(session, self.openai_client, self.mcp)
            await self.domain_tools.register_tools()
            
            # Initialize advanced analysis tools
            self.analysis_tools = AnalysisTools(session, self.mcp)
            await self.analysis_tools.register_tools()

        logger.info("All tools registered successfully")

    async def _shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("Shutting down MCP Code Analysis Server")

        # Close database connections
        if self.engine:
            await self.engine.dispose()

        logger.info("Server shutdown complete")

    def get_session(self):
        """Get database session context manager.

        Returns:
            Database session context manager
        """
        if not self.session_factory:
            raise RuntimeError("Server not initialized")
        return self.session_factory()

    async def scan_repository(
        self,
        repository_url: str,
        branch: str | None = None,
        generate_embeddings: bool = True,
    ) -> dict[str, Any]:
        """Scan a repository.

        Args:
            repository_url: Repository URL
            branch: Branch to scan (optional)
            generate_embeddings: Whether to generate embeddings

        Returns:
            Scan results
        """
        if not self._initialized:
            await self.initialize()

        async with self.get_session() as session:
            scanner = RepositoryScanner(session, self.openai_client)

            # Create repository config
            from src.mcp_server.config import RepositoryConfig

            repo_config = RepositoryConfig(
                url=repository_url,
                branch=branch,
            )

            result = await scanner.scan_repository(repo_config)

            # Process embeddings if requested
            if (
                generate_embeddings
                and self.openai_client
                and result.get("repository_id")
            ):
                embedding_service = EmbeddingService(session, self.openai_client)
                embeddings_result = (
                    await embedding_service.create_repository_embeddings(
                        result["repository_id"],
                    )
                )
                result["embeddings"] = embeddings_result

            return result

    async def update_embeddings(
        self,
        repository_id: int | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Update embeddings for repositories.

        Args:
            repository_id: Optional repository ID (all if None)
            force: Force regeneration of existing embeddings

        Returns:
            Update results
        """
        if not self._initialized:
            await self.initialize()

        if not self.openai_client:
            return {
                "status": "error",
                "message": "OpenAI client not available",
            }

        async with self.get_session() as session:
            embedding_service = EmbeddingService(session, self.openai_client)

            if repository_id:
                # Update single repository
                result = await embedding_service.create_repository_embeddings(
                    repository_id,
                )
                return {
                    "status": "success",
                    "repositories_updated": 1,
                    "embeddings": result,
                }
            # Update all repositories
            results = []
            repositories = await self.repo_management_tools.list_repositories()

            for repo in repositories:
                try:
                    result = await embedding_service.create_repository_embeddings(
                        repo["id"],
                    )
                    results.append(
                        {
                            "repository_id": repo["id"],
                            "repository_name": repo["name"],
                            "result": result,
                        },
                    )
                except Exception as e:
                    logger.exception(
                        f"Failed to update embeddings for repository {repo['id']}: {e}",
                    )
                    results.append(
                        {
                            "repository_id": repo["id"],
                            "repository_name": repo["name"],
                            "error": str(e),
                        },
                    )

            return {
                "status": "success",
                "repositories_updated": len(results),
                "results": results,
            }

    async def search(
        self,
        query: str,
        repository_id: int | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for code.

        Args:
            query: Search query
            repository_id: Optional repository ID to search in
            limit: Maximum number of results

        Returns:
            Search results
        """
        if not self._initialized:
            await self.initialize()

        async with self.get_session():
            if self.code_search_tools and self.openai_client:
                # Use semantic search
                return await self.code_search_tools.semantic_search_impl(
                    query,
                    repository_id=repository_id,
                    limit=limit,
                )
            # Fallback to keyword search
            logger.warning("OpenAI client not available, using keyword search")
            # TODO: Implement keyword-based search
            return []


def create_server() -> MCPCodeAnalysisServer:
    """Create and configure the MCP server.

    Returns:
        Configured server instance
    """
    return MCPCodeAnalysisServer()


# Create a module-level MCP instance that FastMCP can find
_server = None


def get_mcp():
    """Get the FastMCP instance."""
    global _server
    if _server is None:
        _server = create_server()
        # Initialize server on first access
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_server.initialize())
    return _server.mcp


# FastMCP will look for these variables
mcp = get_mcp()
server = mcp  # Alias
app = mcp  # Another alias


if __name__ == "__main__":
    # Run using FastMCP's built-in runner with HTTP transport
    mcp.run(transport="http")
