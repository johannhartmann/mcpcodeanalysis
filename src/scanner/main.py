"""Scanner service main entry point."""

import asyncio
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

from src.config import settings
from src.database import get_session_factory, init_database
from src.database.repositories import CommitRepo, FileRepo, RepositoryRepo
from src.database.session_manager import ParallelSessionManager
from src.logger import get_logger, setup_logging
from src.scanner.code_processor import CodeProcessor
from src.scanner.git_sync import GitSync
from src.scanner.github_monitor import GitHubMonitor

logger = get_logger(__name__)


class ScannerService:
    """Main scanner service for monitoring and syncing repositories."""

    def __init__(self) -> None:
        self.github_monitor = GitHubMonitor()
        self.git_sync = GitSync()
        self.running = False
        self.tasks: list[asyncio.Task] = []
        self.engine: AsyncEngine | None = None
        self.session_factory: Callable[[], Any] | None = None
        self.parallel_session_manager: ParallelSessionManager | None = None

    async def start(self) -> None:
        """Start the scanner service."""
        logger.info("Starting scanner service")

        # Initialize database
        self.engine = await init_database()
        self.session_factory = get_session_factory(self.engine)
        self.parallel_session_manager = ParallelSessionManager(self.session_factory)

        self.running = True

        # Schedule initial sync for all repositories
        for repo_config in settings.repositories:
            task = asyncio.create_task(self.sync_repository(repo_config))
            self.tasks.append(task)

        # Start periodic sync
        sync_task = asyncio.create_task(self.periodic_sync())
        self.tasks.append(sync_task)

        logger.info("Scanner service started")

    async def stop(self) -> None:
        """Stop the scanner service."""
        logger.info("Stopping scanner service")

        self.running = False

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)

        # Close GitHub monitor
        await self.github_monitor.close()

        logger.info("Scanner service stopped")

    async def sync_repository(self, repo_config: dict[str, Any]) -> None:
        """Sync a single repository."""
        repo_url = repo_config["url"]

        try:
            logger.info("Syncing repository: %s", repo_url)

            # Get repository info from GitHub
            repo_info = await self.github_monitor.get_repository_info(
                repo_url,
                repo_config.get("access_token"),
            )

            # Create or update repository in database
            if self.session_factory is None:
                msg = "Session factory not initialized"
                raise RuntimeError(msg)  # noqa: TRY301
            async with self.session_factory() as session:
                repo_repo = RepositoryRepo(session)
                db_repo = await repo_repo.get_by_url(repo_url)

                if not db_repo:
                    db_repo = await repo_repo.create(
                        github_url=repo_url,
                        owner=repo_info["owner"],
                        name=repo_info["name"],
                        default_branch=repo_config.get("branch")
                        or repo_info["default_branch"],
                        access_token_id=repo_config.get("access_token"),
                    )

            # Clone or update repository
            if not self.git_sync._get_repo_path(
                repo_info["owner"],
                repo_info["name"],
            ).exists():
                # Initial clone
                git_repo = await self.git_sync.clone_repository(
                    repo_url,
                    db_repo.default_branch,
                    repo_config.get("access_token"),
                )

                # Process all files
                await self.process_all_files(
                    db_repo,
                    repo_info["owner"],
                    repo_info["name"],
                )
            # Get commits since last sync
            elif db_repo.last_synced:
                commits = await self.github_monitor.get_commits_since(
                    repo_info["owner"],
                    repo_info["name"],
                    db_repo.last_synced,
                    db_repo.default_branch,
                    repo_config.get("access_token"),
                )

                # Store commits
                if self.session_factory is None:
                    msg = "Session factory not initialized"
                    raise RuntimeError(msg)  # noqa: TRY301
                async with self.session_factory() as session:
                    commit_repo = CommitRepo(session)
                    await commit_repo.create_batch(
                        [
                            {
                                "repository_id": db_repo.id,
                                "sha": commit["sha"],
                                "message": commit["message"],
                                "author": commit["author"],
                                "author_email": commit["author_email"],
                                "timestamp": commit["timestamp"],
                            }
                            for commit in commits
                        ],
                    )

                # Update repository
                git_repo = await self.git_sync.update_repository(
                    repo_url,
                    db_repo.default_branch,
                    repo_config.get("access_token"),
                )

                # Get changed files since last sync
                changed_files = set()
                if db_repo.last_synced:
                    # Get commits since last sync
                    recent_commits = await self.git_sync.get_recent_commits(
                        git_repo,
                        db_repo.default_branch,
                        limit=100,
                        since=db_repo.last_synced,
                    )

                    # Collect all changed files from commits
                    for commit_info in recent_commits:
                        changed_files.update(commit_info.get("files_changed", []))

                    logger.info(
                        "Found %d changed files since last sync",
                        len(changed_files),
                    )

                await self.process_changed_files(
                    db_repo,
                    repo_info["owner"],
                    repo_info["name"],
                    changed_files,
                )
            else:
                # No previous commits, process all files
                await self.process_all_files(
                    db_repo,
                    repo_info["owner"],
                    repo_info["name"],
                )

            # Update last synced time
            if self.session_factory is None:
                msg = "Session factory not initialized"
                raise RuntimeError(msg)  # noqa: TRY301
            async with self.session_factory() as session:
                repo_repo = RepositoryRepo(session)
                await repo_repo.update_last_synced(db_repo.id)

            logger.info("Successfully synced repository: %s", repo_url)

        except Exception:
            logger.exception("Error syncing repository %s", repo_url)

    async def process_all_files(self, db_repo: Any, owner: str, name: str) -> None:
        """Process all files in a repository using parallel processing."""
        logger.info("Processing all files for %s/%s", owner, name)

        # Get repository path
        repo_path = self.git_sync._get_repo_path(owner, name)

        # List all supported files
        from src.parser.language_loader import get_configured_extensions

        supported_extensions = get_configured_extensions()

        supported_files = []
        for ext in supported_extensions:
            supported_files.extend(repo_path.rglob(f"*{ext}"))

        if not supported_files:
            logger.info("No supported files found in %s/%s", owner, name)
            return

        # Process files in parallel using the session manager
        if self.parallel_session_manager:

            async def process_file_with_session(file_path: Path, session: Any) -> None:
                return await self._process_file_with_session(
                    db_repo.id, owner, name, file_path, session
                )

            await self.parallel_session_manager.process_files_parallel(
                [str(fp) for fp in supported_files],
                lambda fp, session: process_file_with_session(Path(fp), session),
                batch_size=5,
            )
        else:
            # Fallback to sequential processing
            for file_path in supported_files:
                await self.process_file(db_repo.id, owner, name, file_path)

    async def process_changed_files(
        self,
        db_repo: Any,
        owner: str,
        name: str,
        changed_files: set,
    ) -> None:
        """Process changed files in a repository using parallel processing."""
        logger.info(
            "Processing %s changed files for %s/%s",
            len(changed_files),
            owner,
            name,
        )

        repo_path = self.git_sync._get_repo_path(owner, name)

        # Filter for supported files that exist
        valid_files = []
        for file_path in changed_files:
            full_path = repo_path / file_path
            if full_path.suffix == ".py" and full_path.exists():
                valid_files.append(full_path)

        if not valid_files:
            logger.info("No valid changed files found for %s/%s", owner, name)
            return

        # Process files in parallel using the session manager
        if self.parallel_session_manager:

            async def process_file_with_session(file_path: Path, session: Any) -> None:
                return await self._process_file_with_session(
                    db_repo.id, owner, name, file_path, session
                )

            await self.parallel_session_manager.process_files_parallel(
                [str(fp) for fp in valid_files],
                lambda fp, session: process_file_with_session(Path(fp), session),
                batch_size=3,  # Smaller batch size for changed files
            )
        else:
            # Fallback to sequential processing
            for file_path in valid_files:
                await self.process_file(db_repo.id, owner, name, file_path)

    async def process_file(
        self,
        repo_id: int,
        owner: str,
        name: str,
        file_path: Path,
    ) -> None:
        """Process a single file."""
        try:
            # Get relative path
            repo_path = self.git_sync._get_repo_path(owner, name)
            relative_path = str(file_path.relative_to(repo_path))

            # Get git metadata for the file
            repo_url = f"https://github.com/{owner}/{name}"
            git_repo = self.git_sync.get_repository(repo_url)
            if git_repo:
                metadata = self.git_sync.get_file_git_metadata(git_repo, file_path)
            else:
                # Fallback if git repo not available
                metadata = {
                    "last_modified": datetime.now(UTC).replace(tzinfo=None),
                    "git_hash": None,
                    "size": file_path.stat().st_size,
                }

            # Create or update file in database
            if self.session_factory is None:
                msg = "Session factory not initialized"
                raise RuntimeError(msg)  # noqa: TRY301
            async with self.session_factory() as session:
                file_repo = FileRepo(session)

                # Check if file exists
                db_file = await file_repo.get_by_path(repo_id, relative_path)

                if db_file:
                    # Update existing file
                    db_file.last_modified = metadata.get(
                        "last_modified"
                    ) or datetime.now(UTC).replace(tzinfo=None)
                    git_hash = metadata.get("git_hash")
                    if git_hash:
                        db_file.git_hash = git_hash
                    db_file.size = metadata.get("size", file_path.stat().st_size)
                else:
                    # Create new file
                    db_file = await file_repo.create(
                        repository_id=repo_id,
                        path=relative_path,
                        last_modified=metadata.get(
                            "last_modified", datetime.now(UTC).replace(tzinfo=None)
                        ),
                        git_hash=metadata.get("git_hash"),
                        size=metadata.get("size", file_path.stat().st_size),
                        language="python",
                    )

                # Process code entities using CodeProcessor
                code_processor = CodeProcessor(
                    db_session=session,
                    repository_path=repo_path,
                    enable_domain_analysis=False,
                    enable_parallel=False,
                )

                # Process the file to extract and store entities
                result = await code_processor.process_file(db_file)

                if result["status"] == "success":
                    stats = result.get("statistics", {})
                    logger.debug(
                        "Processed entities from %s: %d modules, %d classes, %d functions",
                        relative_path,
                        stats.get("modules", 0),
                        stats.get("classes", 0),
                        stats.get("functions", 0),
                    )
                else:
                    logger.warning(
                        "Failed to process entities from %s: %s",
                        relative_path,
                        result.get("reason", "unknown"),
                    )

                # Commit is handled inside repository methods

            logger.debug("Processed file: %s", relative_path)

        except Exception:
            logger.exception("Error processing file %s", file_path)

    async def _process_file_with_session(
        self,
        repo_id: int,
        owner: str,
        name: str,
        file_path: Path,
        session: Any,
    ) -> None:
        """Process a single file with provided session (for parallel processing)."""
        try:
            # Get relative path
            repo_path = self.git_sync._get_repo_path(owner, name)
            relative_path = str(file_path.relative_to(repo_path))

            # Get git metadata for the file
            repo_url = f"https://github.com/{owner}/{name}"
            git_repo = self.git_sync.get_repository(repo_url)
            if git_repo:
                metadata = self.git_sync.get_file_git_metadata(git_repo, file_path)
            else:
                # Fallback if git repo not available
                metadata = {
                    "last_modified": datetime.now(UTC).replace(tzinfo=None),
                    "git_hash": None,
                    "size": file_path.stat().st_size,
                }

            # Create or update file in database using provided session
            file_repo = FileRepo(session)

            # Check if file exists
            db_file = await file_repo.get_by_path(repo_id, relative_path)

            if db_file:
                # Update existing file
                db_file.last_modified = metadata.get("last_modified") or datetime.now(
                    UTC
                ).replace(tzinfo=None)
                git_hash = metadata.get("git_hash")
                if git_hash:
                    db_file.git_hash = git_hash
                db_file.size = metadata.get("size", file_path.stat().st_size)
            else:
                # Create new file
                db_file = await file_repo.create(
                    repository_id=repo_id,
                    path=relative_path,
                    last_modified=metadata.get(
                        "last_modified", datetime.now(UTC).replace(tzinfo=None)
                    ),
                    git_hash=metadata.get("git_hash"),
                    size=metadata.get("size", file_path.stat().st_size),
                    language="python",
                )

            # Process code entities using CodeProcessor
            try:
                # Create CodeProcessor instance with the session
                code_processor = CodeProcessor(
                    db_session=session,
                    repository_path=repo_path,
                    enable_domain_analysis=False,
                    enable_parallel=False,
                )

                # Process the file to extract and store entities
                result = await code_processor.process_file(db_file)

                if result["status"] == "success":
                    stats = result.get("statistics", {})
                    logger.debug(
                        "Processed entities from %s: %d modules, %d classes, %d functions",
                        relative_path,
                        stats.get("modules", 0),
                        stats.get("classes", 0),
                        stats.get("functions", 0),
                    )
                else:
                    logger.warning(
                        "Failed to process entities from %s: %s",
                        relative_path,
                        result.get("reason", "unknown"),
                    )
            except Exception:
                logger.exception("Error processing entities from %s", relative_path)

            logger.debug("Processed file: %s", relative_path)

        except Exception:
            logger.exception("Error processing file %s", file_path)

    async def periodic_sync(self) -> None:
        """Periodically sync all repositories."""
        while self.running:
            try:
                # Wait for the configured interval
                sync_interval = getattr(
                    settings.scanner, "sync_interval", 300
                )  # Default 5 minutes
                await asyncio.sleep(sync_interval)

                if not self.running:
                    break

                logger.info("Starting periodic sync")

                # Sync all repositories
                tasks = []
                for repo_config in settings.repositories:
                    task = asyncio.create_task(self.sync_repository(repo_config))
                    tasks.append(task)

                await asyncio.gather(*tasks, return_exceptions=True)

            except Exception:
                logger.exception("Error in periodic sync: %s")


async def main() -> None:
    """Main entry point for scanner service."""
    # Set up logging
    setup_logging()

    # Create scanner service
    scanner = ScannerService()

    # Handle shutdown signals
    def signal_handler(sig: int, frame: Any) -> None:  # noqa: ARG001
        logger.info("Received signal %s", sig)
        asyncio.create_task(scanner.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start scanner
        await scanner.start()

        # Keep running until stopped
        while scanner.running:
            await asyncio.sleep(1)

    except Exception:
        logger.exception("Scanner service error: %s")
        sys.exit(1)
    finally:
        await scanner.stop()


if __name__ == "__main__":
    asyncio.run(main())
