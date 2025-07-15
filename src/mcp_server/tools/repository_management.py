"""Repository management tools for MCP server."""

from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field, SecretStr
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Commit, File, Repository
from src.embeddings.embedding_service import EmbeddingService
from src.embeddings.vector_search import VectorSearch
from src.logger import get_logger
from src.models import RepositoryConfig
from src.scanner.repository_scanner import RepositoryScanner

logger = get_logger(__name__)


class AddRepositoryRequest(BaseModel):
    """Add repository request."""

    url: str = Field(..., description="GitHub repository URL")
    branch: str | None = Field(None, description="Branch to track")
    access_token: SecretStr | None = Field(
        None,
        description="GitHub access token for private repos",
    )


class ScanRepositoryRequest(BaseModel):
    """Scan repository request."""

    repository_id: int = Field(..., description="Repository ID to scan")
    force_full_scan: bool = Field(
        default=False,
        description="Force full scan instead of incremental",
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
        mcp: FastMCP,
    ) -> None:
        """Initialize repository management tools.

        Args:
            db_session: Database session
            mcp: FastMCP instance
        """
        self.db_session = db_session
        self.mcp = mcp
        self.embedding_service = EmbeddingService(db_session)

    async def register_tools(self) -> None:
        """Register all repository management tools."""

        @self.mcp.tool(
            name="add_repository",
            description="""Add a new repository for code analysis and tracking.

            Automatically performs:
            1. Clones/updates the repository
            2. Scans all code files
            3. Extracts code structure (classes, functions, imports)
            4. Generates semantic embeddings for search
            5. Analyzes package structure and dependencies

            Process typically takes 1-5 minutes depending on repository size.

            After adding, you can:
            - Search code with natural language queries
            - Analyze dependencies and coupling
            - Extract domain models
            - Plan migrations
            - Find refactoring opportunities""",
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

                # Always scan the repository (required for functionality)
                scanner = RepositoryScanner(self.db_session)
                scan_result = await scanner.scan_repository(repo_config)

                # Always generate embeddings (core functionality)
                if self.embedding_service:
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

            except (AttributeError, KeyError, ValueError, TypeError) as e:
                logger.exception("Failed to add repository: %s")
                return {
                    "success": False,
                    "error": str(e),
                }

        @self.mcp.tool(
            name="list_repositories",
            description="""List all tracked repositories with their current status.

            Returns information about each repository including:
            - Repository ID (needed for other operations)
            - Name and owner
            - GitHub URL
            - Default branch
            - Last sync time
            - Optional: File counts, embedding counts, language stats

            Use the repository IDs from this tool when calling other tools that
            require a repository_id parameter.""",
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
                            vector_search = VectorSearch(self.db_session)
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

            except (AttributeError, KeyError, ValueError, TypeError) as e:
                logger.exception("Failed to list repositories: %s")
                return {
                    "success": False,
                    "error": str(e),
                    "repositories": [],
                }

        @self.mcp.tool(
            name="scan_repository",
            description="""Update a repository with latest code changes.

            What it does:
            1. Pulls latest changes from Git
            2. Identifies new/modified/deleted files
            3. Updates code structure analysis
            4. Regenerates embeddings for changed files
            5. Updates dependency graphs

            Use when:
            - Code has been updated in the repository
            - Search results seem outdated
            - After merging pull requests
            - Fixing sync issues

            Note: Incremental scan only processes changed files for efficiency.""",
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
                scanner = RepositoryScanner(self.db_session)
                scan_result = await scanner.scan_repository(
                    repo_config,
                    force_full_scan=request.force_full_scan,
                )

                # Always generate embeddings (core functionality)
                if self.embedding_service:
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

            except (AttributeError, KeyError, ValueError, TypeError) as e:
                logger.exception("Failed to scan repository: %s")
                return {
                    "success": False,
                    "error": str(e),
                }

        @self.mcp.tool(
            name="update_embeddings",
            description="""Regenerate semantic search embeddings for improved code search.

            Embeddings enable:
            - Natural language code search
            - Finding similar code patterns
            - Semantic code understanding

            Use when:
            - Search results seem inaccurate
            - After major code refactoring
            - Switching embedding models
            - Fixing embedding generation failures

            Note: This is usually done automatically during scanning.
            Manual updates are rarely needed.""",
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

            except (AttributeError, KeyError, ValueError, TypeError) as e:
                logger.exception("Failed to update embeddings: %s")
                return {
                    "success": False,
                    "error": str(e),
                }

        @self.mcp.tool(
            name="get_repository_stats",
            description="""Get comprehensive statistics and metrics for a repository.

            Returns detailed information including:
            - Code metrics: files, classes, functions, modules
            - Language distribution with percentages
            - Embedding coverage for semantic search
            - Recent commit history (optional)
            - Complexity metrics
            - Size statistics

            Use when:
            - Creating dashboards or reports
            - Understanding codebase composition
            - Monitoring repository growth
            - Planning refactoring efforts""",
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
                    vector_search = VectorSearch(self.db_session)
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

            except (AttributeError, KeyError, ValueError, TypeError) as e:
                logger.exception("Failed to get repository stats: %s")
                return {
                    "success": False,
                    "error": str(e),
                }

        @self.mcp.tool(
            name="delete_repository",
            description="""Permanently delete a repository and all associated data.

            ⚠️ WARNING: This action cannot be undone!

            Deletes:
            - All file and code analysis data
            - All embeddings and search indexes
            - All migration plans for this repository
            - All dependency analysis
            - All extracted domain models

            Requires: confirm=true parameter for safety

            Use when:
            - Repository is no longer needed
            - Cleaning up test repositories
            - Freeing up storage space""",
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

            except (AttributeError, KeyError, ValueError, TypeError) as e:
                logger.exception("Failed to delete repository: %s")
                await self.db_session.rollback()
                return {
                    "success": False,
                    "error": str(e),
                }

        logger.info("Repository management tools registered")
