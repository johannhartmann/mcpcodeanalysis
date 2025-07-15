"""Git repository synchronization functionality."""

import asyncio
import builtins
import contextlib
import hashlib
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import git
from git.exc import GitCommandError, InvalidGitRepositoryError

from src.config import settings
from src.logger import get_logger
from src.utils.exceptions import RepositoryError, ValidationError

logger = get_logger(__name__)

# Constants
EXPECTED_REPO_PATH_PARTS = 2


class GitSync:
    """Git repository synchronization manager."""

    def __init__(self) -> None:
        # Using global settings from src.config
        # Ensure we use absolute path for repositories
        root_path = (
            settings.scanner.root_paths[0]
            if settings.scanner.root_paths
            else "repositories"
        )
        if not Path(root_path).is_absolute():
            # Make it absolute relative to /app
            self.storage_path = Path("/app") / root_path
        else:
            self.storage_path = Path(root_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_repo_path(self, owner: str, name: str) -> Path:
        """Get local path for a repository."""
        return self.storage_path / owner / name

    def extract_owner_repo(self, github_url: str) -> tuple[str, str]:
        """Extract owner and repository name from GitHub URL."""
        # Handle file:// URLs specially
        if github_url.startswith("file://"):
            # For file URLs, use the last two path components as owner/repo
            path = github_url.replace("file://", "")
            parts = path.rstrip("/").split("/")
            if len(parts) >= 2:
                return parts[-2], parts[-1]
            return "local", parts[-1] if parts else "unknown"

        # Handle both HTTPS and SSH URLs for GitHub
        if github_url.startswith("https://github.com/"):
            path = github_url.replace("https://github.com/", "")
        elif github_url.startswith("git@github.com:"):
            path = github_url.replace("git@github.com:", "")
        else:
            msg = "Invalid URL"
            raise ValidationError(msg)

        # Remove .git suffix if present
        if path.endswith(".git"):
            path = path[:-4]

        parts = path.split("/")
        if len(parts) != EXPECTED_REPO_PATH_PARTS:
            msg = "Invalid path"
            raise ValidationError(msg)

        return parts[0], parts[1]

    async def clone_repository(
        self,
        github_url: str,
        branch: str | None = None,
        access_token: str | None = None,
    ) -> git.Repo:
        """Clone a GitHub repository."""
        owner, repo_name = self.extract_owner_repo(github_url)
        repo_path = self._get_repo_path(owner, repo_name)

        logger.info(
            "Cloning repository",
            url=github_url,
            path=str(repo_path),
            branch=branch,
        )

        # Prepare clone URL with authentication
        clone_url = github_url
        if access_token and github_url.startswith("https://"):
            # Insert token for HTTPS URLs
            clone_url = github_url.replace(
                "https://github.com/",
                f"https://{access_token}@github.com/",
            )

        # Remove existing directory if it exists but is not a git repo
        if repo_path.exists():
            try:
                git.Repo(repo_path)
                logger.warning("Repository already exists", path=str(repo_path))
                return await self.update_repository(github_url, branch, access_token)
            except InvalidGitRepositoryError:
                logger.warning(
                    "Removing invalid repository directory",
                    path=str(repo_path),
                )
                shutil.rmtree(repo_path)

        # Create parent directory
        repo_path.parent.mkdir(parents=True, exist_ok=True)

        # Clone repository
        try:
            # Run git clone in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            repo = await loop.run_in_executor(
                None,
                lambda: git.Repo.clone_from(
                    clone_url,
                    repo_path,
                    branch=branch,
                    depth=10,  # Shallow clone with some history
                ),
            )

            logger.info("Repository cloned successfully", path=str(repo_path))
            return repo

        except GitCommandError as e:
            logger.exception("Failed to clone repository", url=github_url, error=str(e))
            msg = "Clone failed"
            raise RepositoryError(msg) from e

    async def update_repository(
        self,
        github_url: str,
        branch: str | None = None,
        access_token: str | None = None,
    ) -> git.Repo:
        """Update an existing repository."""
        owner, repo_name = self.extract_owner_repo(github_url)
        repo_path = self._get_repo_path(owner, repo_name)

        if not repo_path.exists():
            return await self.clone_repository(github_url, branch, access_token)

        try:
            repo = git.Repo(repo_path)
        except InvalidGitRepositoryError:
            logger.warning("Invalid repository, re-cloning", path=str(repo_path))
            shutil.rmtree(repo_path)
            return await self.clone_repository(github_url, branch, access_token)

        logger.info("Updating repository", path=str(repo_path), branch=branch)

        try:
            # Update remote URL if access token provided
            if access_token and github_url.startswith("https://"):
                remote_url = github_url.replace(
                    "https://github.com/",
                    f"https://{access_token}@github.com/",
                )
                repo.remotes.origin.set_url(remote_url)

            # Fetch updates
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, repo.remotes.origin.fetch)

            # Checkout branch
            if branch:
                if branch not in repo.heads:
                    # Create local branch tracking remote
                    repo.create_head(branch, f"origin/{branch}")
                repo.heads[branch].checkout()

            # Pull updates
            active_branch = repo.active_branch
            await loop.run_in_executor(
                None,
                lambda: repo.remotes.origin.pull(active_branch.name),
            )

            logger.info("Repository updated successfully", path=str(repo_path))
            return repo

        except GitCommandError as e:
            logger.exception(
                "Failed to update repository",
                path=str(repo_path),
                error=str(e),
            )
            msg = "Update failed"
            raise RepositoryError(msg) from e

    def get_repository(self, github_url: str) -> git.Repo | None:
        """Get a repository if it exists locally."""
        owner, repo_name = self.extract_owner_repo(github_url)
        repo_path = self._get_repo_path(owner, repo_name)

        if not repo_path.exists():
            return None

        try:
            return git.Repo(repo_path)
        except InvalidGitRepositoryError:
            return None

    def get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with file_path.open("rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def get_changed_files(
        self,
        repo: git.Repo,
        since_commit: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Get files changed since a specific commit."""
        changed_files = {}

        try:
            if since_commit:
                # Get diff between commits
                diff = repo.commit(since_commit).diff(repo.head.commit)
            else:
                # Get all files in current commit
                diff = repo.head.commit.diff(None)

            # Process changed files
            for item in diff:
                file_path = item.a_path or item.b_path

                change_type = "modified"
                if item.new_file:
                    change_type = "added"
                elif item.deleted_file:
                    change_type = "deleted"
                elif item.renamed_file:
                    change_type = "renamed"

                changed_files[file_path] = {
                    "change_type": change_type,
                    "old_path": item.a_path,
                    "new_path": item.b_path,
                }

            # Also check untracked files if no since_commit
            if not since_commit:
                for file_path in repo.untracked_files:
                    changed_files[file_path] = {
                        "change_type": "added",
                        "old_path": None,
                        "new_path": file_path,
                    }

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.exception("Error getting changed files", error=str(e))
            msg = "Get changes failed"
            raise RepositoryError(msg) from e

        return changed_files

    async def scan_repository_files(
        self,
        repo: git.Repo,
        file_extensions: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Scan all files in a repository."""
        if file_extensions is None:
            file_extensions = {".py"}  # Default to Python files

        repo_path = Path(repo.working_dir)
        exclude_patterns = set(settings.scanner.exclude_patterns)

        files = []

        # Walk through repository files
        for root, dirs, filenames in os.walk(repo_path):
            root_path = Path(root)

            # Filter out excluded directories
            dirs[:] = [
                d
                for d in dirs
                if not any(
                    root_path.joinpath(d).match(pattern) for pattern in exclude_patterns
                )
            ]

            # Process files
            for filename in filenames:
                file_path = root_path / filename

                # Check if file should be excluded
                if any(file_path.match(pattern) for pattern in exclude_patterns):
                    continue

                # Check file extension
                if file_extensions and file_path.suffix not in file_extensions:
                    continue

                # Get relative path from repository root
                relative_path = file_path.relative_to(repo_path)

                # Get file info
                try:
                    stat = file_path.stat()

                    # Get git hash if file is tracked
                    git_hash = None
                    with contextlib.suppress(builtins.BaseException):
                        git_hash = repo.odb.stream(
                            repo.head.commit.tree[str(relative_path)].binsha,
                        ).binsha.hex()

                    files.append(
                        {
                            "path": str(relative_path),
                            "absolute_path": str(file_path),
                            "size": stat.st_size,
                            "modified_time": datetime.fromtimestamp(
                                stat.st_mtime, tz=UTC
                            ).replace(tzinfo=None),
                            "content_hash": self.get_file_hash(file_path),
                            "git_hash": git_hash,
                            "language": self._detect_language(file_path),
                        },
                    )

                except (OSError, ValueError, UnicodeDecodeError) as e:
                    logger.warning(
                        "Error processing file",
                        file=str(file_path),
                        error=str(e),
                    )

        return files

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".rb": "ruby",
            ".go": "go",
            ".rs": "rust",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".m": "objc",
            ".mm": "objcpp",
        }

        return extension_map.get(file_path.suffix.lower(), "unknown")

    def get_commit_info(self, repo: git.Repo, commit_sha: str) -> dict[str, Any]:
        """Get information about a specific commit."""
        try:
            commit = repo.commit(commit_sha)

            # Get basic commit info
            commit_info: dict[str, Any] = {
                "sha": commit.hexsha,
                "message": commit.message.strip(),
                "author": commit.author.name,
                "author_email": commit.author.email,
                "timestamp": datetime.fromtimestamp(
                    commit.committed_date, tz=UTC
                ).replace(tzinfo=None),
                "files_changed": [],
                "additions": 0,
                "deletions": 0,
            }

            # Try to get stats, but don't fail if it doesn't work (e.g., shallow clone)
            try:
                stats = commit.stats
                commit_info["files_changed"] = list(stats.files.keys())
                commit_info["additions"] = stats.total["insertions"]
                commit_info["deletions"] = stats.total["deletions"]
            except (
                AttributeError,
                KeyError,
                TypeError,
                git.exc.GitCommandError,
            ) as stats_error:
                logger.warning(
                    "Could not get commit stats for %s: %s",
                    commit_sha,
                    stats_error,
                )

            return commit_info
        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.exception(
                "Error getting commit info",
                commit_sha=commit_sha,
                error=str(e),
            )
            msg = "Get commit failed"
            raise RepositoryError(msg) from e

    async def get_recent_commits(
        self,
        repo: git.Repo,
        branch: str | None = None,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent commits from repository."""
        commits = []

        try:
            # Get the branch to analyze
            if branch:
                if branch in repo.heads:
                    commit_iter = repo.iter_commits(branch, max_count=limit)
                else:
                    logger.warning("Branch %s not found, using HEAD", branch)
                    commit_iter = repo.iter_commits("HEAD", max_count=limit)
            else:
                commit_iter = repo.iter_commits("HEAD", max_count=limit)

            # Filter by date if needed
            for commit in commit_iter:
                commit_date = datetime.fromtimestamp(
                    commit.committed_date, tz=UTC
                ).replace(tzinfo=None)
                if since and commit_date < since:
                    break

                commits.append(self.get_commit_info(repo, commit.hexsha))

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.exception("Error getting recent commits", error=str(e))
            msg = "Get commits failed"
            raise RepositoryError(msg) from e

        return commits

    def get_file_git_metadata(
        self,
        repo: git.Repo,
        file_path: Path,
    ) -> dict[str, Any]:
        """Get git metadata for a specific file.

        Args:
            repo: Git repository
            file_path: Path to file (absolute or relative to repo)

        Returns:
            Dictionary with git metadata
        """
        try:
            repo_path = Path(repo.working_dir)

            # Get relative path
            if file_path.is_absolute():
                relative_path = file_path.relative_to(repo_path)
            else:
                relative_path = file_path

            # Get git hash if file is tracked
            git_hash = None
            last_commit = None

            try:
                # Get the file's git object
                git_file = repo.head.commit.tree[str(relative_path)]
                git_hash = git_file.binsha.hex()

                # Get last commit that modified this file
                commits = list(repo.iter_commits(paths=str(relative_path), max_count=1))
                if commits:
                    last_commit = commits[0]

            except (KeyError, git.exc.GitCommandError):
                # File not tracked in git
                logger.debug("File not tracked in git: %s", relative_path)

            # Get file stats
            stat = (
                file_path.stat()
                if file_path.is_absolute()
                else (repo_path / file_path).stat()
            )

            metadata = {
                "git_hash": git_hash,
                "last_modified": datetime.fromtimestamp(stat.st_mtime, tz=UTC).replace(
                    tzinfo=None
                ),
                "size": stat.st_size,
            }

            if last_commit:
                metadata.update(
                    {
                        "last_commit_sha": last_commit.hexsha,
                        "last_commit_date": datetime.fromtimestamp(
                            last_commit.committed_date, tz=UTC
                        ).replace(tzinfo=None),
                        "last_commit_author": last_commit.author.name,
                        "last_commit_message": last_commit.message.strip(),
                    }
                )

            return metadata

        except (OSError, git.exc.GitCommandError, ValueError) as e:
            logger.warning("Error getting git metadata for file %s: %s", file_path, e)
            # Return basic metadata on error
            return {
                "git_hash": None,
                "last_modified": datetime.now(UTC).replace(tzinfo=None),
                "size": 0,
            }
