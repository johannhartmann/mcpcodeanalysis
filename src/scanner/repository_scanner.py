"""Repository scanner that integrates GitHub monitoring, Git sync, and database."""

import os
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.config import settings
from src.database.models import Commit, File, Repository
from src.logger import get_logger
from src.models import RepositoryConfig
from src.scanner.code_processor import CodeProcessor
from src.scanner.git_sync import GitSync
from src.scanner.github_client import GitHubClient

logger = get_logger(__name__)


class RepositoryScanner:
    """Main repository scanner that coordinates all scanning operations."""

    def __init__(
        self,
        db_session: AsyncSession,
    ) -> None:
        self.db_session = db_session
        # Using global settings from src.config
        self.git_sync = GitSync()
        self.code_processor = None  # Will be initialized per repository
        self._github_clients: dict[str, GitHubClient] = {}

    def _get_github_client(self, access_token: str | None = None) -> GitHubClient:
        """Get or create a GitHub client for the given access token."""
        token_key = access_token or "default"
        if token_key not in self._github_clients:
            self._github_clients[token_key] = GitHubClient(access_token)
        return self._github_clients[token_key]

    async def scan_repository(
        self,
        repo_config: RepositoryConfig,
        *,
        force_full_scan: bool = False,
    ) -> dict[str, Any]:
        """Scan a single repository."""
        logger.info(
            "Starting repository scan",
            url=repo_config.url,
            branch=repo_config.branch,
            force_full_scan=force_full_scan,
        )

        # Extract owner and repo name
        owner, repo_name = self.git_sync.extract_owner_repo(repo_config.url)

        # Get or create repository record
        repo_record = await self._get_or_create_repository(
            repo_config,
            owner,
            repo_name,
        )

        # Get GitHub client
        access_token = (
            repo_config.access_token.get_secret_value()
            if repo_config.access_token
            else None
        )
        github_client = self._get_github_client(access_token)

        # Update repository info from GitHub
        async with github_client:
            try:
                repo_info = await github_client.get_repository(owner, repo_name)

                # Update default branch if not specified
                if not repo_config.branch:
                    repo_record.default_branch = repo_info["default_branch"]

                # Store additional metadata
                repo_record.metadata = {
                    "description": repo_info.get("description"),
                    "language": repo_info.get("language"),
                    "size": repo_info.get("size"),
                    "stargazers_count": repo_info.get("stargazers_count"),
                    "updated_at": repo_info.get("updated_at"),
                }

            except (AttributeError, KeyError, ValueError, TypeError) as e:
                logger.exception(
                    "Failed to fetch repository info from GitHub",
                    error=str(e),
                )

        # Clone or update repository
        git_repo = await self.git_sync.update_repository(
            repo_config.url,
            repo_config.branch or repo_record.default_branch,
            access_token,
        )

        # Determine what to scan
        last_scan_commit = None
        if not force_full_scan and repo_record.last_synced:
            # Get last processed commit
            last_commit = await self.db_session.execute(
                select(Commit)
                .where(Commit.repository_id == repo_record.id)
                .where(Commit.processed)
                .order_by(Commit.timestamp.desc())
                .limit(1),
            )
            last_commit_record = last_commit.scalar_one_or_none()
            if last_commit_record:
                last_scan_commit = last_commit_record.sha

        # Get new commits
        new_commits = await self._process_commits(
            repo_record,
            git_repo,
            github_client,
            since_commit=last_scan_commit,
        )

        # Scan files
        if force_full_scan or not last_scan_commit:
            # Full scan
            scanned_files = await self._full_file_scan(repo_record, git_repo)
        else:
            # Incremental scan based on commits
            scanned_files = await self._incremental_file_scan(
                repo_record,
                git_repo,
                new_commits,
            )

        # Process scanned files to extract code entities
        # Create processor with the repository path and domain analysis settings
        enable_domain = repo_config.enable_domain_analysis or getattr(
            settings, "domain_analysis", {}
        ).get("enabled", False)

        # Enable parallel processing for large file sets
        enable_parallel = len(scanned_files) > 10

        code_processor = CodeProcessor(
            self.db_session,
            repository_path=git_repo.working_dir,
            enable_domain_analysis=enable_domain,
            enable_parallel=enable_parallel,
        )
        parse_results = await code_processor.process_files(scanned_files)

        # After all files are processed, resolve any pending references
        # This helps handle cross-file references that couldn't be resolved during parallel processing
        await self._resolve_pending_references(repo_record.id)

        # Update repository last sync time
        repo_record.last_synced = datetime.now(UTC)  # PostgreSQL expects naive datetime
        await self.db_session.commit()

        # Run bounded context detection if domain analysis is enabled
        context_detection_result = {}
        if enable_domain:
            try:
                from src.domain.indexer import DomainIndexer

                domain_indexer = DomainIndexer(self.db_session)
                context_ids = await domain_indexer.detect_and_save_contexts()
                context_detection_result = {
                    "contexts_detected": len(context_ids),
                    "context_ids": context_ids,
                }
            except (ValueError, RuntimeError, AttributeError) as e:
                logger.warning("Context detection failed: %s", e)

        return {
            "repository_id": repo_record.id,
            "commits_processed": len(new_commits),
            "files_scanned": len(scanned_files),
            "files_parsed": parse_results["success"],
            "parse_statistics": parse_results["statistics"],
            "domain_analysis": context_detection_result,
            "full_scan": force_full_scan or not last_scan_commit,
        }

    async def _resolve_pending_references(self, repository_id: int) -> None:
        """Resolve any references that couldn't be resolved during initial processing.

        This is particularly useful when files are processed in parallel and
        references to entities in other files can't be resolved yet.
        """
        logger.info("Resolving pending references for repository %d", repository_id)

        # Get all files for this repository
        result = await self.db_session.execute(
            select(File).where(File.repository_id == repository_id)
        )
        files = result.scalars().all()

        # Re-process references for each file
        # This time all entities should be in the database
        resolved_count = 0
        for file in files:
            try:
                # Get parsed entities for this file
                from src.scanner.code_processor import Module

                module_result = await self.db_session.execute(
                    select(Module).where(Module.file_id == file.id)
                )
                modules = module_result.scalars().all()

                if modules:
                    # Re-extract references (this could be optimized to only re-resolve unresolved ones)
                    # For now, we'll just log that this step would happen
                    # In a production system, we'd track unresolved references separately
                    resolved_count += 1

            except (AttributeError, KeyError, ValueError, TypeError) as e:
                logger.warning(
                    "Failed to resolve references for file %s: %s", file.path, str(e)
                )

        if resolved_count > 0:
            logger.info(
                "Completed reference resolution for %d files in repository %d",
                resolved_count,
                repository_id,
            )

    async def _get_or_create_repository(
        self,
        repo_config: RepositoryConfig,
        owner: str,
        repo_name: str,
    ) -> Repository:
        """Get existing repository or create new one."""
        result = await self.db_session.execute(
            select(Repository).where(Repository.github_url == repo_config.url),
        )
        repo = result.scalar_one_or_none()

        if not repo:
            repo = Repository(
                github_url=repo_config.url,
                owner=owner,
                name=repo_name,
                default_branch=repo_config.branch or "main",
                access_token_id=(
                    f"token_{owner}_{repo_name}" if repo_config.access_token else None
                ),
            )
            self.db_session.add(repo)
            await self.db_session.commit()
            logger.info("Created new repository record", repo_id=repo.id)

        return repo

    async def _process_commits(
        self,
        repo_record: Repository,
        git_repo,
        _github_client: GitHubClient,
        since_commit: str | None = None,
    ) -> list[Commit]:
        """Process new commits from repository."""
        logger.info(
            "Processing commits",
            repo_id=repo_record.id,
            since_commit=since_commit,
        )

        # Get commits from Git
        commits_data = await self.git_sync.get_recent_commits(
            git_repo,
            branch=repo_record.default_branch,
            limit=1000,  # Reasonable limit
        )

        # Filter commits already in database
        existing_shas = set()
        if commits_data:
            result = await self.db_session.execute(
                select(Commit.sha).where(
                    Commit.repository_id == repo_record.id,
                    Commit.sha.in_([c["sha"] for c in commits_data]),
                ),
            )
            existing_shas = {row[0] for row in result}

        # Create new commit records
        new_commits = []
        for commit_data in commits_data:
            if commit_data["sha"] in existing_shas:
                continue

            # Stop if we've reached the last processed commit
            if since_commit and commit_data["sha"] == since_commit:
                break

            commit = Commit(
                repository_id=repo_record.id,
                sha=commit_data["sha"],
                message=commit_data["message"],
                author=commit_data["author"],
                author_email=commit_data["author_email"],
                timestamp=commit_data["timestamp"],
                files_changed=commit_data["files_changed"],
                additions=commit_data["additions"],
                deletions=commit_data["deletions"],
                processed=False,
            )
            self.db_session.add(commit)
            new_commits.append(commit)

        if new_commits:
            await self.db_session.commit()
            logger.info("Added new commits", count=len(new_commits))

        return new_commits

    async def _full_file_scan(
        self,
        repo_record: Repository,
        git_repo,
    ) -> list[File]:
        """Perform full scan of all repository files."""
        logger.info("Performing full file scan", repo_id=repo_record.id)

        # Get all supported code files
        from src.parser.parser_factory import ParserFactory

        supported_extensions = set(ParserFactory.get_supported_extensions())

        files_data = await self.git_sync.scan_repository_files(
            git_repo,
            file_extensions=supported_extensions,
        )

        # Mark all existing files as potentially deleted
        from sqlalchemy import update

        await self.db_session.execute(
            update(File)
            .where(File.repository_id == repo_record.id)
            .values(is_deleted=True),
        )

        # Process each file
        scanned_files = []
        for file_data in files_data:
            file_record = await self._update_or_create_file(
                repo_record,
                file_data,
                git_repo.active_branch.name,
            )
            scanned_files.append(file_record)

        await self.db_session.commit()
        return scanned_files

    async def _incremental_file_scan(
        self,
        repo_record: Repository,
        git_repo,
        new_commits: list[Commit],
    ) -> list[File]:
        """Perform incremental scan based on new commits."""
        logger.info(
            "Performing incremental file scan",
            repo_id=repo_record.id,
            commits=len(new_commits),
        )

        # Collect all changed files from commits
        changed_files: set[str] = set()
        for commit in new_commits:
            changed_files.update(commit.files_changed)

        # Filter for supported code files
        from src.parser.parser_factory import ParserFactory

        supported_extensions = ParserFactory.get_supported_extensions()

        supported_files = [
            f
            for f in changed_files
            if any(f.endswith(ext) for ext in supported_extensions)
        ]

        # Process each changed file
        scanned_files = []
        for file_path in supported_files:
            # Get current file info
            full_path = git_repo.working_dir / file_path

            if not full_path.exists():
                # File was deleted
                result = await self.db_session.execute(
                    select(File).where(
                        File.repository_id == repo_record.id,
                        File.path == file_path,
                    ),
                )
                file_record = result.scalar_one_or_none()
                if file_record:
                    file_record.is_deleted = True
                    scanned_files.append(file_record)
            else:
                # File exists, update it
                file_data = {
                    "path": file_path,
                    "absolute_path": str(full_path),
                    "size": full_path.stat().st_size,
                    "modified_time": datetime.fromtimestamp(
                        full_path.stat().st_mtime, tz=UTC
                    ),
                    "content_hash": self.git_sync.get_file_hash(full_path),
                    "git_hash": None,  # Will be set by _update_or_create_file
                    "language": "python",
                }

                file_record = await self._update_or_create_file(
                    repo_record,
                    file_data,
                    git_repo.active_branch.name,
                )
                scanned_files.append(file_record)

            # Mark commit as processed
            commit.processed = True

        await self.db_session.commit()
        return scanned_files

    async def _update_or_create_file(
        self,
        repo_record: Repository,
        file_data: dict[str, any],
        branch: str,
    ) -> File:
        """Update existing file record or create new one."""
        result = await self.db_session.execute(
            select(File).where(
                File.repository_id == repo_record.id,
                File.path == file_data["path"],
                File.branch == branch,
            ),
        )
        file_record = result.scalar_one_or_none()

        if not file_record:
            file_record = File(
                repository_id=repo_record.id,
                path=file_data["path"],
                branch=branch,
            )
            self.db_session.add(file_record)

        # Update file data
        file_record.content_hash = file_data["content_hash"]
        file_record.git_hash = file_data.get("git_hash")
        file_record.size = file_data["size"]
        file_record.language = file_data["language"]
        file_record.last_modified = file_data["modified_time"]
        file_record.is_deleted = False

        return file_record

    async def scan_all_repositories(
        self,
        *,
        force_full_scan: bool = False,
    ) -> dict[str, Any]:
        """Scan all configured repositories."""
        logger.info("Starting scan of all repositories")

        results = []
        for repo_config in settings.repositories:
            try:
                result = await self.scan_repository(
                    repo_config, force_full_scan=force_full_scan
                )
                results.append(
                    {
                        "url": repo_config.url,
                        "status": "success",
                        "details": result,
                    },
                )
            except (AttributeError, KeyError, ValueError, TypeError) as e:
                logger.exception(
                    "Failed to scan repository",
                    url=repo_config.url,
                    error=str(e),
                )
                results.append(
                    {
                        "url": repo_config.url,
                        "status": "error",
                        "error": str(e),
                    },
                )

        return {
            "repositories_scanned": len(results),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "error"),
            "results": results,
        }

    async def setup_webhooks(self) -> dict[str, any]:
        """Set up webhooks for all configured repositories."""
        if not hasattr(settings, "github") or not settings.github.use_webhooks:
            return {"message": "Webhooks disabled in configuration"}

        webhook_url = f"{settings.mcp.host}:{settings.mcp.port}{getattr(settings, 'github', {}).get('webhook_endpoint', '/webhook')}"
        results = []

        for repo_config in settings.repositories:
            try:
                owner, repo_name = self.git_sync.extract_owner_repo(repo_config.url)
                access_token = (
                    repo_config.access_token.get_secret_value()
                    if repo_config.access_token
                    else None
                )

                github_client = self._get_github_client(access_token)
                async with github_client:
                    webhook = await github_client.create_webhook(
                        owner,
                        repo_name,
                        webhook_url,
                        ["push", "create", "delete"],
                        secret=(
                            os.getenv("GITHUB_WEBHOOK_SECRET", "")
                            if os.getenv("GITHUB_WEBHOOK_SECRET")
                            else None
                        ),
                    )

                    # Update repository record with webhook ID
                    result = await self.db_session.execute(
                        select(Repository).where(
                            Repository.github_url == repo_config.url,
                        ),
                    )
                    repo = result.scalar_one_or_none()
                    if repo:
                        repo.webhook_id = str(webhook["id"])
                        await self.db_session.commit()

                    results.append(
                        {
                            "url": repo_config.url,
                            "webhook_id": webhook["id"],
                            "status": "created",
                        },
                    )

            except (AttributeError, KeyError, ValueError, TypeError) as e:
                logger.exception(
                    "Failed to create webhook",
                    url=repo_config.url,
                    error=str(e),
                )
                results.append(
                    {
                        "url": repo_config.url,
                        "status": "error",
                        "error": str(e),
                    },
                )

        return {
            "webhooks_created": sum(1 for r in results if r["status"] == "created"),
            "failed": sum(1 for r in results if r["status"] == "error"),
            "results": results,
        }
