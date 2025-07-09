"""Repository management tools for MCP server."""

from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field, SecretStr
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Commit, File, Repository
from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.openai_client import OpenAIClient
from src.embeddings.vector_search import VectorSearch
from src.mcp_server.config import RepositoryConfig
from src.scanner.repository_scanner import RepositoryScanner
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AddRepositoryRequest(BaseModel):
    """Add repository request."""

    url: str = Field(..., description="GitHub repository URL")
    branch: str | None = Field(None, description="Branch to track")
    access_token: SecretStr | None = Field(
        None,
        description="GitHub access token for private repos",
    )
    scan_immediately: bool = Field(
        default=True,
        description="Start scanning immediately",
    )
    generate_embeddings: bool = Field(
        default=True,
        description="Generate embeddings after scanning",
    )


class ScanRepositoryRequest(BaseModel):
    """Scan repository request."""

    repository_id: int = Field(..., description="Repository ID to scan")
    force_full_scan: bool = Field(
        default=False,
        description="Force full scan instead of incremental",
    )
    generate_embeddings: bool = Field(
        default=True,
        description="Generate embeddings after scanning",
    )


class UpdateEmbeddingsRequest(BaseModel):
    """Update embeddings request."""

    repository_id: int = Field(..., description="Repository ID")
    file_limit: int | None = Field(
        None,
        description="Limit number of files to process",
    )


class RepositoryStatsRequest(BaseModel):
    """Repository statistics request."""

    repository_id: int = Field(..., description="Repository ID")
    include_commit_history: bool = Field(
        default=False,
        description="Include recent commit history",
    )


class RepositoryManagementTools:
    """Repository management tools for MCP."""

    def __init__(
        self,
        db_session: AsyncSession,
        openai_client: OpenAIClient | None,
        mcp: FastMCP,
    ) -> None:
        """Initialize repository management tools.

        Args:
            db_session: Database session
            openai_client: OpenAI client for embeddings
            mcp: FastMCP instance
        """
        self.db_session = db_session
        self.openai_client = openai_client
        self.mcp = mcp
        self.embedding_service = (
            EmbeddingService(db_session, openai_client) if openai_client else None
        )

    async def register_tools(self):
        """Register all repository management tools."""

        @self.mcp.tool(
            name="add_repository",
            description="Add a new repository to track",
        )
        async def add_repository(request: AddRepositoryRequest) -> dict[str, Any]:
            """Add a new repository for tracking and analysis.

            Args:
                request: Add repository request parameters

            Returns:
                Repository information and scan results
            """
            try:
                # Check if repository already exists
                existing = await self.db_session.execute(
                    select(Repository).where(Repository.github_url == request.url),
                )
                if existing.scalar_one_or_none():
                    return {
                        "success": False,
                        "error": "Repository already exists",
                    }

                # Create repository config
                repo_config = RepositoryConfig(
                    url=request.url,
                    branch=request.branch,
                    access_token=request.access_token,
                )

                # Start scanning if requested
                scan_result = None
                if request.scan_immediately:
                    scanner = RepositoryScanner(self.db_session, self.openai_client)
                    scan_result = await scanner.scan_repository(repo_config)

                    # Generate embeddings if requested
                    if request.generate_embeddings and self.embedding_service:
                        embedding_result = (
                            await self.embedding_service.create_repository_embeddings(
                                scan_result["repository_id"],
                            )
                        )
                        scan_result["embeddings"] = embedding_result

                return {
                    "success": True,
                    "repository": {
                        "url": request.url,
                        "branch": request.branch or "default",
                    },
                    "scan_result": scan_result,
                }

            except Exception as e:
                logger.exception(f"Failed to add repository: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }

        @self.mcp.tool(
            name="list_repositories",
            description="List all tracked repositories",
        )
        async def list_repositories(
            include_stats: bool = Field(
                default=False,
                description="Include repository statistics",
            ),
        ) -> dict[str, Any]:
            """List all tracked repositories.

            Args:
                include_stats: Include file and embedding counts

            Returns:
                List of repositories
            """
            try:
                # Get repositories
                result = await self.db_session.execute(
                    select(Repository).order_by(Repository.name),
                )
                repositories = result.scalars().all()

                repo_list = []
                for repo in repositories:
                    repo_info = {
                        "id": repo.id,
                        "name": repo.name,
                        "owner": repo.owner,
                        "url": repo.github_url,
                        "default_branch": repo.default_branch,
                        "last_synced": (
                            repo.last_synced.isoformat() if repo.last_synced else None
                        ),
                    }

                    if include_stats:
                        # Get file count
                        file_count = await self.db_session.execute(
                            select(func.count(File.id)).where(
                                File.repository_id == repo.id,
                            ),
                        )

                        # Get embedding count
                        if self.embedding_service:
                            vector_search = VectorSearch(
                                self.db_session,
                                self.openai_client,
                            )
                            stats = await vector_search.get_repository_stats(repo.id)
                            repo_info["stats"] = {
                                "files": file_count.scalar() or 0,
                                "embeddings": stats["total_embeddings"],
                                "embeddings_by_type": stats["embeddings_by_type"],
                            }
                        else:
                            repo_info["stats"] = {
                                "files": file_count.scalar() or 0,
                            }

                    repo_list.append(repo_info)

                return {
                    "success": True,
                    "repositories": repo_list,
                    "count": len(repo_list),
                }

            except Exception as e:
                logger.exception(f"Failed to list repositories: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "repositories": [],
                }

        @self.mcp.tool(
            name="scan_repository",
            description="Scan or rescan a repository",
        )
        async def scan_repository(request: ScanRepositoryRequest) -> dict[str, Any]:
            """Scan or rescan a repository.

            Args:
                request: Scan repository request parameters

            Returns:
                Scan results
            """
            try:
                # Get repository
                result = await self.db_session.execute(
                    select(Repository).where(Repository.id == request.repository_id),
                )
                repo = result.scalar_one_or_none()

                if not repo:
                    return {
                        "success": False,
                        "error": f"Repository {request.repository_id} not found",
                    }

                # Create repository config
                repo_config = RepositoryConfig(
                    url=repo.github_url,
                    branch=repo.default_branch,
                    access_token=(
                        SecretStr(repo.access_token_id)
                        if repo.access_token_id
                        else None
                    ),
                )

                # Scan repository
                scanner = RepositoryScanner(self.db_session, self.openai_client)
                scan_result = await scanner.scan_repository(
                    repo_config,
                    force_full_scan=request.force_full_scan,
                )

                # Generate embeddings if requested
                if request.generate_embeddings and self.embedding_service:
                    embedding_result = (
                        await self.embedding_service.create_repository_embeddings(
                            request.repository_id,
                        )
                    )
                    scan_result["embeddings"] = embedding_result

                return {
                    "success": True,
                    "repository": {
                        "id": repo.id,
                        "name": repo.name,
                        "url": repo.github_url,
                    },
                    "scan_result": scan_result,
                }

            except Exception as e:
                logger.exception(f"Failed to scan repository: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }

        @self.mcp.tool(
            name="update_embeddings",
            description="Update embeddings for a repository",
        )
        async def update_embeddings(request: UpdateEmbeddingsRequest) -> dict[str, Any]:
            """Update embeddings for repository files.

            Args:
                request: Update embeddings request parameters

            Returns:
                Embedding generation results
            """
            try:
                if not self.embedding_service:
                    return {
                        "success": False,
                        "error": "Embedding service not available",
                    }

                # Get repository
                result = await self.db_session.execute(
                    select(Repository).where(Repository.id == request.repository_id),
                )
                repo = result.scalar_one_or_none()

                if not repo:
                    return {
                        "success": False,
                        "error": f"Repository {request.repository_id} not found",
                    }

                # Generate embeddings
                embedding_result = (
                    await self.embedding_service.create_repository_embeddings(
                        request.repository_id,
                        limit=request.file_limit,
                    )
                )

                return {
                    "success": True,
                    "repository": {
                        "id": repo.id,
                        "name": repo.name,
                    },
                    "embeddings": embedding_result,
                }

            except Exception as e:
                logger.exception(f"Failed to update embeddings: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }

        @self.mcp.tool(
            name="get_repository_stats",
            description="Get detailed statistics for a repository",
        )
        async def get_repository_stats(
            request: RepositoryStatsRequest,
        ) -> dict[str, Any]:
            """Get detailed statistics for a repository.

            Args:
                request: Repository stats request parameters

            Returns:
                Repository statistics
            """
            try:
                # Get repository
                result = await self.db_session.execute(
                    select(Repository).where(Repository.id == request.repository_id),
                )
                repo = result.scalar_one_or_none()

                if not repo:
                    return {
                        "success": False,
                        "error": f"Repository {request.repository_id} not found",
                    }

                stats = {
                    "repository": {
                        "id": repo.id,
                        "name": repo.name,
                        "owner": repo.owner,
                        "url": repo.github_url,
                        "last_synced": (
                            repo.last_synced.isoformat() if repo.last_synced else None
                        ),
                    },
                }

                # Get file statistics
                from src.database.models import Class, Function, Module

                file_count = await self.db_session.execute(
                    select(func.count(File.id)).where(
                        and_(
                            File.repository_id == request.repository_id,
                            not File.is_deleted,
                        ),
                    ),
                )

                # Get entity counts
                module_count = await self.db_session.execute(
                    select(func.count(Module.id))
                    .join(Module.file)
                    .where(File.repository_id == request.repository_id),
                )

                class_count = await self.db_session.execute(
                    select(func.count(Class.id))
                    .join(Class.file)
                    .where(File.repository_id == request.repository_id),
                )

                function_count = await self.db_session.execute(
                    select(func.count(Function.id))
                    .join(Function.file)
                    .where(File.repository_id == request.repository_id),
                )

                stats["code_stats"] = {
                    "files": file_count.scalar() or 0,
                    "modules": module_count.scalar() or 0,
                    "classes": class_count.scalar() or 0,
                    "functions": function_count.scalar() or 0,
                }

                # Get language distribution
                lang_result = await self.db_session.execute(
                    select(File.language, func.count(File.id).label("count"))
                    .where(
                        and_(
                            File.repository_id == request.repository_id,
                            not File.is_deleted,
                        ),
                    )
                    .group_by(File.language),
                )

                stats["languages"] = {row[0]: row[1] for row in lang_result if row[0]}

                # Get embedding statistics if available
                if self.embedding_service:
                    vector_search = VectorSearch(self.db_session, self.openai_client)
                    embedding_stats = await vector_search.get_repository_stats(
                        request.repository_id,
                    )
                    stats["embeddings"] = embedding_stats

                # Get commit history if requested
                if request.include_commit_history:
                    commits_result = await self.db_session.execute(
                        select(Commit)
                        .where(Commit.repository_id == request.repository_id)
                        .order_by(Commit.timestamp.desc())
                        .limit(20),
                    )
                    commits = commits_result.scalars().all()

                    stats["recent_commits"] = [
                        {
                            "sha": c.sha[:7],
                            "message": c.message[:100],
                            "author": c.author,
                            "timestamp": (
                                c.timestamp.isoformat() if c.timestamp else None
                            ),
                            "files_changed": (
                                len(c.files_changed) if c.files_changed else 0
                            ),
                        }
                        for c in commits
                    ]

                return {
                    "success": True,
                    **stats,
                }

            except Exception as e:
                logger.exception(f"Failed to get repository stats: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }

        @self.mcp.tool(
            name="delete_repository",
            description="Delete a repository and all its data",
        )
        async def delete_repository(
            repository_id: int = Field(..., description="Repository ID to delete"),
            confirm: bool = Field(
                default=False,
                description="Confirm deletion (required)",
            ),
        ) -> dict[str, Any]:
            """Delete a repository and all associated data.

            Args:
                repository_id: Repository ID to delete
                confirm: Must be True to confirm deletion

            Returns:
                Deletion result
            """
            try:
                if not confirm:
                    return {
                        "success": False,
                        "error": "Deletion not confirmed. Set confirm=true to delete.",
                    }

                # Get repository
                result = await self.db_session.execute(
                    select(Repository).where(Repository.id == repository_id),
                )
                repo = result.scalar_one_or_none()

                if not repo:
                    return {
                        "success": False,
                        "error": f"Repository {repository_id} not found",
                    }

                # Delete repository (cascades to all related data)
                await self.db_session.delete(repo)
                await self.db_session.commit()

                return {
                    "success": True,
                    "message": f"Repository '{repo.name}' deleted successfully",
                }

            except Exception as e:
                logger.exception(f"Failed to delete repository: {e}")
                await self.db_session.rollback()
                return {
                    "success": False,
                    "error": str(e),
                }

        logger.info("Repository management tools registered")
