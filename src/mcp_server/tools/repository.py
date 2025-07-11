"""Repository management tools for MCP server."""

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Commit, File
from src.database.repositories import CommitRepo, FileRepo, RepositoryRepo
from src.logger import get_logger
from src.scanner.git_sync import GitSync
from src.scanner.github_monitor import GitHubMonitor

logger = get_logger(__name__)

# Time constants
SECONDS_PER_HOUR = 3600
SECONDS_PER_MINUTE = 60


class RepositoryTool:
    """MCP tools for repository management."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.github_monitor = GitHubMonitor()
        self.git_sync = GitSync()
        self.repo_repo = RepositoryRepo(session)
        self.file_repo = FileRepo(session)
        self.commit_repo = CommitRepo(session)

    async def list_repositories(self) -> list[dict[str, Any]]:
        """
        List all monitored GitHub repositories with their sync status.

        Returns:
            List of repositories with their current status
        """
        try:
            repo_list = await self.repo_repo.list_all()

            repositories = []
            for repo in repo_list:
                # Get file and commit counts
                from sqlalchemy import func

                file_count_result = await self.session.execute(
                    select(func.count(File.id)).where(File.repository_id == repo.id),
                )
                file_count = file_count_result.scalar() or 0

                commit_count_result = await self.session.execute(
                    select(func.count(Commit.id)).where(
                        Commit.repository_id == repo.id
                    ),
                )
                commit_count = commit_count_result.scalar() or 0

                # Get last commit
                last_commit = await self.commit_repo.get_latest(repo.id)

                # Check local directory exists
                local_path = self.git_sync._get_repo_path(repo.owner, repo.name)
                is_cloned = local_path.exists()

                repositories.append(
                    {
                        "id": repo.id,
                        "name": repo.name,
                        "owner": repo.owner,
                        "url": repo.github_url,
                        "branch": repo.default_branch,
                        "last_synced": (
                            repo.last_synced.isoformat() if repo.last_synced else None
                        ),
                        "status": {
                            "is_cloned": is_cloned,
                            "file_count": file_count,
                            "commit_count": commit_count,
                            "last_commit": (
                                {
                                    "sha": (
                                        last_commit.sha[:7] if last_commit else None
                                    ),
                                    "message": (
                                        last_commit.message if last_commit else None
                                    ),
                                    "timestamp": (
                                        last_commit.timestamp.isoformat()
                                        if last_commit
                                        else None
                                    ),
                                }
                                if last_commit
                                else None
                            ),
                        },
                        "sync_age": self._calculate_sync_age(repo.last_synced),
                    },
                )

            return repositories

        except Exception as e:
            logger.exception("Error in list_repositories: %s")
            return [{"error": str(e)}]

    async def sync_repository(self, repository_url: str) -> dict[str, Any]:
        """
        Manually trigger sync for a specific repository.

        Args:
            repository_url: GitHub repository URL

        Returns:
            Sync status and results
        """
        try:
            # Find repository
            repo = await self.repo_repo.get_by_url(repository_url)
            result: dict[str, Any]

            if not repo:
                # Try to add new repository
                try:
                    # Get repository info from GitHub
                    repo_info = await self.github_monitor.get_repository_info(
                        repository_url,
                    )

                    # Create repository record
                    repo = await self.repo_repo.create(
                        github_url=repository_url,
                        owner=repo_info["owner"],
                        name=repo_info["name"],
                        default_branch=repo_info["default_branch"],
                    )

                    result = {
                        "status": "added",
                        "repository": {
                            "id": repo.id,
                            "name": repo.name,
                            "owner": repo.owner,
                            "url": repo.github_url,
                        },
                        "message": "Repository added and sync initiated",
                    }
                except (ValueError, ConnectionError, OSError) as e:
                    return {
                        "status": "error",
                        "error": f"Failed to add repository: {e!s}",
                        "repository_url": repository_url,
                    }
            else:
                result = {
                    "status": "syncing",
                    "repository": {
                        "id": repo.id,
                        "name": repo.name,
                        "owner": repo.owner,
                        "url": repo.github_url,
                    },
                    "message": "Sync initiated for existing repository",
                }

            # Check if cloned
            local_path = self.git_sync._get_repo_path(repo.owner, repo.name)

            if not local_path.exists():
                # Clone repository
                git_repo = await self.git_sync.clone_repository(
                    repo.github_url,
                    repo.default_branch,
                )
                result["cloned"] = True
                result["local_path"] = str(local_path)
            else:
                # Update repository
                git_repo = await self.git_sync.update_repository(
                    repo.github_url,
                    repo.default_branch,
                )
                result["updated"] = True

            # Get changed files count
            if repo.last_synced:
                # TODO: Implement changed files detection
                result["changed_files"] = 0

            # Update last synced
            await self.repo_repo.update_last_synced(repo.id)

            # Get latest commits
            if repo.last_synced:
                commits = await self.github_monitor.get_commits_since(
                    repo.owner,
                    repo.name,
                    repo.last_synced,
                    repo.default_branch,
                )
                result["new_commits"] = len(commits)

                # Store commits
                if commits:
                    await self.commit_repo.create_batch(
                        [
                            {
                                "repository_id": repo.id,
                                "sha": commit["sha"],
                                "message": commit["message"],
                                "author": commit["author"],
                                "author_email": commit["author_email"],
                                "timestamp": commit["timestamp"],
                            }
                            for commit in commits
                        ],
                    )

            return result

        except Exception as e:
            logger.exception("Error in sync_repository: %s")
            return {
                "status": "error",
                "error": str(e),
                "repository_url": repository_url,
            }

    def _calculate_sync_age(self, last_synced: datetime | None) -> str | None:
        """Calculate human-readable sync age."""
        if not last_synced:
            return "Never synced"

        from datetime import UTC

        age = datetime.now(tz=UTC) - last_synced

        if age.days > 0:
            return f"{age.days} day{'s' if age.days > 1 else ''} ago"
        if age.seconds > SECONDS_PER_HOUR:
            hours = age.seconds // SECONDS_PER_HOUR
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        if age.seconds > SECONDS_PER_MINUTE:
            minutes = age.seconds // SECONDS_PER_MINUTE
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        return "Just now"
