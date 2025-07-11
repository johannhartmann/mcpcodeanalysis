"""MCP Code Analysis Server implementation - Fixed version."""

from collections.abc import AsyncGenerator
from typing import Any

from fastmcp import FastMCP
from pydantic import Field
from sqlalchemy import select

from src.database.init_db import get_session_factory, init_database
from src.database.models import Repository
from src.logger import get_logger
from src.mcp_server.tools.code_analysis import CodeAnalysisTools
from src.mcp_server.tools.code_search import CodeSearchTools
from src.mcp_server.tools.package_analysis import (
    AnalyzePackagesRequest,
    FindCircularDependenciesRequest,
    GetPackageCouplingRequest,
    GetPackageDependenciesRequest,
    GetPackageDetailsRequest,
    GetPackageTreeRequest,
    analyze_packages,
    find_circular_dependencies,
    get_package_coupling_metrics,
    get_package_dependencies,
    get_package_details,
    get_package_tree,
)
from src.mcp_server.tools.repository_management import RepositoryManagementTools
from src.models import RepositoryConfig

logger = get_logger(__name__)

# Create the global FastMCP instance
mcp = FastMCP("Code Analysis Server")

# Global variables for shared resources
_engine = None
_session_factory = None
_settings = None


async def initialize_server() -> None:
    """Initialize server resources."""
    global _engine, _session_factory

    if _engine is not None:
        return  # Already initialized

    logger.info("Starting MCP Code Analysis Server")

    # Load settings
    # settings imported globally from src.config

    # Initialize database
    logger.info("Initializing database connection")
    _engine = await init_database()
    _session_factory = get_session_factory(_engine)

    # OpenAI initialization is now handled by individual components using LangChain

    logger.info("Server initialized successfully")


async def get_db_session() -> AsyncGenerator[Any, None]:
    """Get a database session."""
    if _session_factory is None:
        await initialize_server()

    async with _session_factory() as session:
        yield session


# Register repository management tools
@mcp.tool(name="add_repository", description="Add a new repository to track")
async def add_repository(
    url: str = Field(description="Repository URL (GitHub or file://)"),
    branch: str = Field(default=None, description="Branch to track (optional)"),
    scan_immediately: bool = Field(
        default=True,
        description="Scan repository immediately",
    ),
    generate_embeddings: bool = Field(
        default=True,
        description="Generate embeddings for code",
    ),
) -> dict[str, Any]:
    """Add a new repository to track."""
    await initialize_server()

    async for session in get_db_session():
        try:
            # Check for existing repository
            result = await session.execute(
                select(Repository).where(Repository.github_url == url),
            )
            existing_repo = result.scalar_one_or_none()

            if existing_repo:
                return {
                    "success": False,
                    "error": f"Repository already exists with ID {existing_repo.id}",
                    "repository_id": existing_repo.id,
                }

            # Add repository to database
            repo = Repository(
                github_url=url,
                owner=url.split("/")[-2] if "/" in url else "local",
                name=url.split("/")[-1].replace(".git", "") if "/" in url else url,
                default_branch=branch or "main",
            )
            session.add(repo)
            await session.commit()
            await session.refresh(repo)

            scan_result = {"repository_id": repo.id}

            # Scan if requested
            if scan_immediately:
                from src.scanner.repository_scanner import RepositoryScanner

                repo_config = RepositoryConfig(
                    url=url,
                    branch=branch,
                    repository_id=repo.id,
                )

                scanner = RepositoryScanner(session)
                scan_result = await scanner.scan_repository(repo_config)

                # Generate embeddings if requested
                if generate_embeddings:
                    from src.embeddings.embedding_service import EmbeddingService

                    embedding_service = EmbeddingService(session)
                    embedding_result = (
                        await embedding_service.create_repository_embeddings(
                            scan_result["repository_id"],
                        )
                    )
                    scan_result["embeddings"] = embedding_result

            return {
                "success": True,
                "repository": {
                    "id": repo.id,
                    "url": url,
                    "branch": branch or "default",
                },
                "scan_result": scan_result,
            }
        except Exception as e:
            logger.exception("Failed to add repository: %s")
            return {
                "success": False,
                "error": str(e),
            }
    return None


@mcp.tool(name="list_repositories", description="List all tracked repositories")
async def list_repositories(
    include_stats: bool = Field(
        default=False,
        description="Include repository statistics",
    ),
) -> dict[str, Any]:
    """List all tracked repositories."""
    await initialize_server()

    async for session in get_db_session():
        try:
            # Get repositories
            result = await session.execute(select(Repository).order_by(Repository.name))
            repositories = result.scalars().all()

            repo_list = []
            for repo in repositories:
                repo_data = {
                    "id": repo.id,
                    "name": repo.name,
                    "owner": repo.owner,
                    "url": repo.github_url,
                    "branch": repo.default_branch,
                    "last_synced": (
                        repo.last_synced.isoformat() if repo.last_synced else None
                    ),
                }

                if include_stats:
                    # Get file count
                    from sqlalchemy import func

                    from src.database.models import CodeEmbedding, File

                    file_count_result = await session.execute(
                        select(func.count(File.id)).where(
                            File.repository_id == repo.id,
                        ),
                    )
                    file_count = file_count_result.scalar() or 0

                    # Get embedding count
                    embedding_count_result = await session.execute(
                        select(func.count(CodeEmbedding.id))
                        .join(File)
                        .where(File.repository_id == repo.id),
                    )
                    embedding_count = embedding_count_result.scalar() or 0

                    repo_data["stats"] = {
                        "total_files": file_count,
                        "total_embeddings": embedding_count,
                    }

                repo_list.append(repo_data)

            return {
                "success": True,
                "repositories": repo_list,
                "count": len(repo_list),
            }
        except Exception as e:
            logger.exception("Failed to list repositories: %s")
            return {
                "success": False,
                "error": str(e),
                "repositories": [],
            }
    return None


@mcp.tool(name="scan_repository", description="Scan or rescan a repository")
async def scan_repository(
    repository_id: int = Field(description="Repository ID to scan"),
    force_full_scan: bool = Field(default=False, description="Force full rescan"),
    generate_embeddings: bool = Field(default=True, description="Generate embeddings"),
) -> dict[str, Any]:
    """Scan or rescan a repository."""
    await initialize_server()

    async for session in get_db_session():
        repo_tools = RepositoryManagementTools(session, mcp)
        return await repo_tools.scan_repository(
            {
                "repository_id": repository_id,
                "force_full_scan": force_full_scan,
                "generate_embeddings": generate_embeddings,
            },
        )
    return None


@mcp.tool(name="remove_repository", description="Remove a repository from tracking")
async def remove_repository(
    repository_id: int = Field(description="Repository ID to remove"),
) -> dict[str, Any]:
    """Remove a repository from tracking."""
    await initialize_server()

    async for session in get_db_session():
        repo_tools = RepositoryManagementTools(session, mcp)
        return await repo_tools.remove_repository(repository_id=repository_id)
    return None


@mcp.tool(name="update_repository_settings", description="Update repository settings")
async def update_repository_settings(
    repository_id: int = Field(description="Repository ID"),
    branch: str = Field(default=None, description="New branch to track"),
    auto_scan: bool = Field(default=None, description="Enable automatic scanning"),
) -> dict[str, Any]:
    """Update repository settings."""
    await initialize_server()

    async for session in get_db_session():
        repo_tools = RepositoryManagementTools(session, mcp)
        return await repo_tools.update_repository_settings(
            {
                "repository_id": repository_id,
                "branch": branch,
                "auto_scan": auto_scan,
            },
        )
    return None


# Register code search tools
@mcp.tool(name="semantic_search", description="Search code using natural language")
async def semantic_search(
    query: str = Field(description="Natural language search query"),
    repository_id: int = Field(
        default=None,
        description="Limit to specific repository",
    ),
    limit: int = Field(default=10, description="Maximum results to return"),
) -> dict[str, Any]:
    """Search code using natural language."""
    await initialize_server()

    async for session in get_db_session():
        search_tools = CodeSearchTools(session, mcp)
        return await search_tools.semantic_search(
            {
                "query": query,
                "repository_id": repository_id,
                "limit": limit,
            },
        )
    return None


@mcp.tool(name="keyword_search", description="Search code using keywords")
async def keyword_search(
    keywords: list[str] = Field(description="Keywords to search for"),
    scope: str = Field(
        default="all",
        description="Search scope: all, functions, classes, modules",
    ),
    repository_id: int = Field(
        default=None,
        description="Limit to specific repository",
    ),
    limit: int = Field(default=20, description="Maximum results"),
) -> dict[str, Any]:
    """Search code using keywords."""
    await initialize_server()

    async for session in get_db_session():
        search_tools = CodeSearchTools(session, mcp)
        return await search_tools.keyword_search(
            {
                "keywords": keywords,
                "scope": scope,
                "repository_id": repository_id,
                "limit": limit,
            },
        )
    return None


@mcp.tool(name="find_similar_code", description="Find code similar to a given snippet")
async def find_similar_code(
    code_snippet: str = Field(description="Code snippet to find similar code for"),
    repository_id: int = Field(
        default=None,
        description="Limit to specific repository",
    ),
    threshold: float = Field(default=0.7, description="Similarity threshold (0-1)"),
    limit: int = Field(default=10, description="Maximum results"),
) -> dict[str, Any]:
    """Find code similar to a given snippet."""
    await initialize_server()

    async for session in get_db_session():
        search_tools = CodeSearchTools(session, mcp)
        return await search_tools.find_similar_code(
            {
                "code_snippet": code_snippet,
                "repository_id": repository_id,
                "threshold": threshold,
                "limit": limit,
            },
        )
    return None


# Register code analysis tools
@mcp.tool(name="get_code", description="Get code for a specific entity")
async def get_code(
    entity_type: str = Field(description="Entity type: function, class, or module"),
    entity_id: int = Field(description="Entity ID"),
    include_context: bool = Field(
        default=False,
        description="Include surrounding context",
    ),
) -> dict[str, Any]:
    """Get code for a specific entity."""
    await initialize_server()

    async for session in get_db_session():
        analysis_tools = CodeAnalysisTools(session, mcp)
        return await analysis_tools.get_code(
            {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "include_context": include_context,
            },
        )
    return None


@mcp.tool(name="analyze_file", description="Analyze a specific file")
async def analyze_file(
    file_path: str = Field(description="File path within repository"),
    repository_id: int = Field(description="Repository ID"),
) -> dict[str, Any]:
    """Analyze a specific file."""
    await initialize_server()

    async for session in get_db_session():
        analysis_tools = CodeAnalysisTools(session, mcp)
        return await analysis_tools.analyze_file(
            {
                "file_path": file_path,
                "repository_id": repository_id,
            },
        )
    return None


@mcp.tool(name="get_file_structure", description="Get the structure of a file")
async def get_file_structure(
    file_path: str = Field(description="File path within repository"),
    repository_id: int = Field(description="Repository ID"),
    include_imports: bool = Field(
        default=True,
        description="Include import statements",
    ),
) -> dict[str, Any]:
    """Get the structure of a file."""
    await initialize_server()

    async for session in get_db_session():
        analysis_tools = CodeAnalysisTools(session, mcp)
        return await analysis_tools.get_file_structure(
            {
                "file_path": file_path,
                "repository_id": repository_id,
                "include_imports": include_imports,
            },
        )
    return None


@mcp.tool(
    name="analyze_dependencies",
    description="Analyze dependencies of a module or file",
)
async def analyze_dependencies(
    file_path: str = Field(description="File path to analyze"),
    repository_id: int = Field(description="Repository ID"),
    depth: int = Field(default=1, description="Depth of dependency analysis"),
) -> dict[str, Any]:
    """Analyze dependencies of a module or file."""
    await initialize_server()

    async for session in get_db_session():
        analysis_tools = CodeAnalysisTools(session, mcp)
        return await analysis_tools.analyze_dependencies(
            {
                "file_path": file_path,
                "repository_id": repository_id,
                "depth": depth,
            },
        )
    return None


# Package analysis tools
@mcp.tool(
    name="analyze_package_structure",
    description="Analyze the package structure of a repository",
)
async def analyze_package_structure_tool(
    repository_id: int = Field(description="Repository ID to analyze"),
    force_refresh: bool = Field(
        default=False, description="Force re-analysis even if data exists"
    ),
) -> dict[str, Any]:
    """Analyze the package structure of a repository."""
    await initialize_server()

    async for session in get_db_session():
        return await analyze_packages(
            AnalyzePackagesRequest(
                repository_id=repository_id, force_refresh=force_refresh
            ),
            session,
        )
    return None


@mcp.tool(
    name="get_package_tree",
    description="Get the hierarchical package structure",
)
async def get_package_tree_tool(
    repository_id: int = Field(description="Repository ID"),
) -> dict[str, Any]:
    """Get the hierarchical package structure of a repository."""
    await initialize_server()

    async for session in get_db_session():
        return await get_package_tree(
            GetPackageTreeRequest(repository_id=repository_id), session
        )
    return None


@mcp.tool(
    name="get_package_details",
    description="Get detailed information about a specific package",
)
async def get_package_details_tool(
    repository_id: int = Field(description="Repository ID"),
    package_path: str = Field(description="Package path (e.g., 'src/utils')"),
) -> dict[str, Any]:
    """Get detailed information about a specific package."""
    await initialize_server()

    async for session in get_db_session():
        return await get_package_details(
            GetPackageDetailsRequest(
                repository_id=repository_id, package_path=package_path
            ),
            session,
        )
    return None


@mcp.tool(
    name="get_package_dependencies",
    description="Get dependencies for a specific package",
)
async def get_package_dependencies_tool(
    repository_id: int = Field(description="Repository ID"),
    package_path: str = Field(description="Package path"),
    direction: str = Field(
        default="both", description="Direction: 'imports', 'imported_by', or 'both'"
    ),
) -> dict[str, Any]:
    """Get dependencies for a specific package."""
    await initialize_server()

    async for session in get_db_session():
        return await get_package_dependencies(
            GetPackageDependenciesRequest(
                repository_id=repository_id,
                package_path=package_path,
                direction=direction,
            ),
            session,
        )
    return None


@mcp.tool(
    name="find_circular_dependencies",
    description="Find circular dependencies between packages",
)
async def find_circular_dependencies_tool(
    repository_id: int = Field(description="Repository ID"),
) -> dict[str, Any]:
    """Find circular dependencies between packages."""
    await initialize_server()

    async for session in get_db_session():
        return await find_circular_dependencies(
            FindCircularDependenciesRequest(repository_id=repository_id), session
        )
    return None


@mcp.tool(
    name="get_package_coupling_metrics",
    description="Get coupling metrics for all packages",
)
async def get_package_coupling_metrics_tool(
    repository_id: int = Field(description="Repository ID"),
) -> dict[str, Any]:
    """Get coupling metrics for all packages in a repository."""
    await initialize_server()

    async for session in get_db_session():
        return await get_package_coupling_metrics(
            GetPackageCouplingRequest(repository_id=repository_id), session
        )
    return None


# Aliases for compatibility
server = mcp
app = mcp


class MockServer:
    """Mock server for compatibility."""

    async def initialize(self) -> None:
        await initialize_server()

    async def shutdown(self) -> None:
        if _engine:
            await _engine.dispose()

    async def scan_repository(
        self, url: str, branch: str | None = None, generate_embeddings: bool = True
    ) -> dict[str, Any] | None:
        await initialize_server()
        async for session in get_db_session():
            repo_tools = RepositoryManagementTools(session, mcp)
            return await repo_tools.add_repository(
                {
                    "url": url,
                    "branch": branch,
                    "scan_immediately": True,
                    "generate_embeddings": generate_embeddings,
                },
            )
        return None

    async def search(
        self, query: str, repository_id: int | None = None, limit: int = 10
    ) -> dict[str, Any] | None:
        await initialize_server()
        async for session in get_db_session():
            search_tools = CodeSearchTools(session, mcp)
            return await search_tools.semantic_search(
                {
                    "query": query,
                    "repository_id": repository_id,
                    "limit": limit,
                },
            )
        return None


def create_server() -> MockServer:
    """Create server instance for compatibility."""
    return MockServer()


if __name__ == "__main__":
    # Run using FastMCP's built-in runner with HTTP transport
    mcp.run(transport="http")
