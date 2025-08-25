"""Tests for repository scanner."""

from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import git
import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Commit, File, Repository
from src.models import RepositoryConfig
from src.scanner.repository_scanner import RepositoryScanner
from src.utils.exceptions import RepositoryError


@pytest.fixture
def mock_db_session() -> Any:
    """Create mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def mock_settings() -> Any:
    """Create mock settings."""
    settings = MagicMock()
    settings.repositories = [
        RepositoryConfig(
            url="https://github.com/test-owner/test-repo",
            branch="main",
        ),
        RepositoryConfig(
            url="https://github.com/test-owner/test-repo2",
            branch="develop",
            access_token=SecretStr("secret_token"),
        ),
    ]
    settings.github.use_webhooks = True
    settings.github.webhook_endpoint = "/webhook"
    settings.mcp.host = "localhost"
    settings.mcp.port = 8000
    settings.scanner.webhook_secret = MagicMock(
        get_secret_value=lambda: "webhook_secret",
    )
    return settings


@pytest.fixture
def repository_scanner(
    mock_db_session: AsyncSession, mock_settings: Any
) -> Generator[RepositoryScanner, None, None]:
    """Create RepositoryScanner fixture."""
    with patch("src.scanner.repository_scanner.settings", mock_settings):
        with patch("src.scanner.repository_scanner.GitSync") as mock_git_sync_class:
            # Create a mock GitSync instance
            mock_git_sync = MagicMock()
            mock_git_sync_class.return_value = mock_git_sync

            scanner = RepositoryScanner(mock_db_session)
            # Ensure URL parsing returns owner/name for test URLs
            mock_git_sync.extract_owner_repo = MagicMock(
                return_value=("test-owner", "test-repo")
            )

            scanner.git_sync = mock_git_sync
            yield scanner


@pytest.fixture
def mock_repo_record() -> Repository:
    """Create mock repository database record."""
    repo = MagicMock(spec=Repository)
    repo.id = 1
    repo.github_url = "https://github.com/test-owner/test-repo"
    repo.owner = "test-owner"
    repo.name = "test-repo"
    repo.default_branch = "main"
    repo.last_synced = None
    repo.metadata = {}
    return repo


@pytest.fixture
def mock_git_repo() -> git.Repo:
    """Create mock git repository."""
    repo = MagicMock(spec=git.Repo)
    # working_dir needs to be a Path that supports division operator
    from pathlib import Path

    repo.working_dir = Path("/tmp/test-repo")  # nosec B108 - mock path for testing
    repo.active_branch = MagicMock()
    # mypy: ignore-errors

    repo.active_branch.name = "main"
    return repo


class TestRepositoryScanner:
    """Tests for RepositoryScanner class."""

    def test_get_github_client_new(self, repository_scanner: RepositoryScanner) -> None:
        """Test getting new GitHub client."""
        client1 = repository_scanner._get_github_client("token1")
        client2 = repository_scanner._get_github_client("token1")
        client3 = repository_scanner._get_github_client("token2")

        assert client1 is client2  # Same token returns same client
        assert client1 is not client3  # Different token returns different client

    def test_get_github_client_default(
        self, repository_scanner: RepositoryScanner
    ) -> None:
        """Test getting default GitHub client."""
        client1 = repository_scanner._get_github_client()
        client2 = repository_scanner._get_github_client(None)

        assert client1 is client2

    @pytest.mark.asyncio
    async def test_get_or_create_repository_existing(
        self,
        repository_scanner: RepositoryScanner,
        mock_repo_record: Repository,
        mock_settings: Any,
    ) -> None:
        """Test getting existing repository."""
        repo_config = mock_settings.repositories[0]

        # Mock database query
        result = MagicMock()
        result.scalar_one_or_none.return_value = mock_repo_record
        cast("Any", repository_scanner.db_session).execute.return_value = result

        repo = await repository_scanner._get_or_create_repository(
            repo_config,
            "test-owner",
            "test-repo",
        )

        assert repo == mock_repo_record
        cast("Any", repository_scanner.db_session).add.assert_not_called()
        cast("Any", repository_scanner.db_session).commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_create_repository_new(
        self,
        repository_scanner: RepositoryScanner,
        mock_settings: Any,
    ) -> None:
        """Test creating new repository."""
        repo_config = mock_settings.repositories[0]

        # Mock database query returning None
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        cast("Any", repository_scanner.db_session).execute.return_value = result

        repo = await repository_scanner._get_or_create_repository(
            repo_config,
            "test-owner",
            "test-repo",
        )

        assert isinstance(repo, Repository)
        assert repo.github_url == repo_config.url
        assert repo.owner == "test-owner"
        assert repo.name == "test-repo"
        cast("Any", repository_scanner.db_session).add.assert_called_once_with(repo)
        cast("Any", repository_scanner.db_session).commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_commits_new(
        self,
        repository_scanner: RepositoryScanner,
        mock_repo_record: Repository,
        mock_git_repo: git.Repo,
    ) -> None:
        """Test processing new commits."""
        # Mock git commits
        commits_data: list[dict[str, Any]] = [
            {
                "sha": "abc123",
                "message": "First commit",
                "author": "Test Author",
                "author_email": "test@example.com",
                "timestamp": datetime.now(tz=UTC),
                "files_changed": ["file1.py"],
                "additions": 10,
                "deletions": 5,
            },
            {
                "sha": "def456",
                "message": "Second commit",
                "author": "Test Author",
                "author_email": "test@example.com",
                "timestamp": datetime.now(tz=UTC),
                "files_changed": ["file2.py"],
                "additions": 20,
                "deletions": 0,
            },
        ]

        with patch.object(
            repository_scanner.git_sync,
            "get_recent_commits",
            new=AsyncMock(return_value=commits_data),
        ):
            # Mock database query for existing commits
            result = MagicMock()
            result.fetchall.return_value = []
            result.__iter__ = lambda x: iter([])
            cast("Any", repository_scanner.db_session).execute.return_value = result

            github_client = MagicMock()
            new_commits = await repository_scanner._process_commits(
                mock_repo_record,
                mock_git_repo,
                github_client,
            )

            assert len(new_commits) == 2
            assert all(isinstance(c, Commit) for c in new_commits)
            assert new_commits[0].sha == "abc123"
            assert new_commits[1].sha == "def456"
            cast("Any", repository_scanner.db_session).commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_commits_with_existing(
        self,
        repository_scanner: RepositoryScanner,
        mock_repo_record: Repository,
        mock_git_repo: git.Repo,
    ) -> None:
        """Test processing commits with some already existing."""
        commits_data: list[dict[str, Any]] = [
            {
                "sha": "abc123",
                "message": "Existing commit",
                "author": "Test Author",
                "author_email": "test@example.com",
                "timestamp": datetime.now(tz=UTC),
                "files_changed": ["file1.py"],
                "additions": 5,
                "deletions": 2,
            },
            {
                "sha": "def456",
                "message": "New commit",
                "author": "Test Author",
                "author_email": "test@example.com",
                "timestamp": datetime.now(tz=UTC),
                "files_changed": ["file2.py"],
                "additions": 10,
                "deletions": 0,
            },
        ]

        with patch.object(
            repository_scanner.git_sync,
            "get_recent_commits",
            new=AsyncMock(return_value=commits_data),
        ):
            # Mock database query showing abc123 exists
            result = MagicMock()
            result.__iter__ = lambda x: iter([("abc123",)])
            cast("Any", repository_scanner.db_session).execute.return_value = result

            github_client = MagicMock()
            new_commits = await repository_scanner._process_commits(
                mock_repo_record,
                mock_git_repo,
                github_client,
            )

            assert len(new_commits) == 1
            assert new_commits[0].sha == "def456"

    @pytest.mark.asyncio
    async def test_full_file_scan(
        self,
        repository_scanner: RepositoryScanner,
        mock_repo_record: Repository,
        mock_git_repo: git.Repo,
    ) -> None:
        """Test full file scan."""
        files_data: list[dict[str, Any]] = [
            {
                "path": "src/main.py",
                "absolute_path": "/tmp/test-repo/src/main.py",  # nosec B108 - mock path
                "size": 1000,
                "modified_time": datetime.now(tz=UTC),
                "content_hash": "hash123",
                "git_hash": "githash123",
                "language": "python",
            },
            {
                "path": "src/utils.py",
                "absolute_path": "/tmp/test-repo/src/utils.py",  # nosec B108 - mock path
                "size": 500,
                "modified_time": datetime.now(tz=UTC),
                "content_hash": "hash456",
                "git_hash": "githash456",
                "language": "python",
            },
        ]

        with (
            patch(
                "src.parser.parser_factory.ParserFactory.get_supported_extensions",
                return_value={".py"},
            ),
            patch.object(
                repository_scanner.git_sync,
                "scan_repository_files",
                new=AsyncMock(return_value=files_data),
            ),
            patch.object(
                repository_scanner,
                "_update_or_create_file",
            ) as mock_update_file,
        ):
            mock_file1 = MagicMock(spec=File)
            mock_file2 = MagicMock(spec=File)
            mock_update_file.side_effect = [mock_file1, mock_file2]

            scanned_files = await repository_scanner._full_file_scan(
                mock_repo_record,
                mock_git_repo,
            )

            assert len(scanned_files) == 2
            assert mock_update_file.call_count == 2
            cast("Any", repository_scanner.db_session).commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_or_create_file_existing(
        self,
        repository_scanner: RepositoryScanner,
        mock_repo_record: Repository,
    ) -> None:
        """Test updating existing file."""
        file_data: dict[str, Any] = {
            "path": "src/main.py",
            "content_hash": "newhash",
            "git_hash": "newgithash",
            "size": 1500,
            "language": "python",
            "modified_time": datetime.now(tz=UTC),
        }

        # Mock existing file
        existing_file = MagicMock(spec=File)
        existing_file.content_hash = "oldhash"

        result = MagicMock()
        result.scalar_one_or_none.return_value = existing_file
        cast("Any", repository_scanner.db_session).execute.return_value = result

        file_record = await repository_scanner._update_or_create_file(
            mock_repo_record,
            file_data,
            "main",
        )

        assert file_record == existing_file
        assert file_record.content_hash == "newhash"
        assert file_record.git_hash == "newgithash"
        assert file_record.size == 1500
        assert file_record.is_deleted is False

    @pytest.mark.asyncio
    async def test_update_or_create_file_new(
        self,
        repository_scanner: RepositoryScanner,
        mock_repo_record: Repository,
    ) -> None:
        """Test creating new file."""
        file_data: dict[str, Any] = {
            "path": "src/new.py",
            "content_hash": "hash123",
            "git_hash": None,
            "size": 1000,
            "language": "python",
            "modified_time": datetime.now(tz=UTC),
        }

        # Mock no existing file
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        cast("Any", repository_scanner.db_session).execute.return_value = result

        file_record = await repository_scanner._update_or_create_file(
            mock_repo_record,
            file_data,
            "main",
        )

        assert isinstance(file_record, File)
        assert file_record.path == "src/new.py"
        assert file_record.repository_id == mock_repo_record.id
        assert file_record.branch == "main"
        cast("Any", repository_scanner.db_session).add.assert_called_once_with(
            file_record
        )

    @pytest.mark.asyncio
    async def test_scan_repository_full(
        self,
        repository_scanner: RepositoryScanner,
        mock_repo_record: Repository,
        mock_git_repo: git.Repo,
        mock_settings: Any,
    ) -> None:
        """Test full repository scan."""
        repo_config = mock_settings.repositories[0]

        with (
            patch.object(
                repository_scanner,
                "_get_or_create_repository",
                new=AsyncMock(return_value=mock_repo_record),
            ),
            patch.object(
                repository_scanner.git_sync,
                "update_repository",
                new=AsyncMock(return_value=mock_git_repo),
            ),
            patch.object(
                repository_scanner,
                "_process_commits",
                new=AsyncMock(return_value=[]),
            ),
            patch.object(
                repository_scanner,
                "_full_file_scan",
                new=AsyncMock(return_value=[MagicMock(), MagicMock()]),
            ),
            patch(
                "src.scanner.repository_scanner.CodeProcessor"
            ) as mock_code_processor_class,
        ):
            # Mock CodeProcessor to avoid initialization issues
            mock_code_processor = MagicMock()
            mock_code_processor.process_files = AsyncMock(
                return_value={"success": 2, "statistics": {}}
            )
            mock_code_processor_class.return_value = mock_code_processor
            # Mock GitHub client
            mock_github_client = AsyncMock()
            mock_github_client.get_repository = AsyncMock(
                return_value={
                    "default_branch": "main",
                    "description": "Test repo",
                    "language": "Python",
                },
            )

            with patch.object(
                repository_scanner,
                "_get_github_client",
                return_value=mock_github_client,
            ):
                result = await repository_scanner.scan_repository(
                    repo_config,
                    force_full_scan=True,
                )

                assert result["repository_id"] == mock_repo_record.id
                assert result["files_scanned"] == 2
                assert result["full_scan"] is True

    @pytest.mark.asyncio
    async def test_scan_all_repositories_success(
        self,
        repository_scanner: RepositoryScanner,
        mock_settings: Any,
    ) -> None:
        """Test scanning all repositories successfully."""
        with patch.object(
            repository_scanner,
            "scan_repository",
            new=AsyncMock(return_value={"repository_id": 1, "files_scanned": 10}),
        ):
            results = await repository_scanner.scan_all_repositories()

            assert results["repositories_scanned"] == 2
            assert results["successful"] == 2
            assert results["failed"] == 0
            assert len(results["results"]) == 2

    @pytest.mark.asyncio
    async def test_scan_all_repositories_with_error(
        self,
        repository_scanner: RepositoryScanner,
        mock_settings: Any,
    ) -> None:
        """Test scanning repositories with one failure."""

        async def scan_side_effect(
            repo_config: RepositoryConfig, force_full_scan: bool = False
        ) -> dict[str, int]:
            if "test-repo2" in repo_config.url:
                raise RepositoryError("Failed to scan")
            return {"repository_id": 1, "files_scanned": 10}

        with patch.object(
            repository_scanner,
            "scan_repository",
            side_effect=scan_side_effect,
        ):
            results = await repository_scanner.scan_all_repositories()

            assert results["repositories_scanned"] == 2
            assert results["successful"] == 1
            assert results["failed"] == 1
            assert results["results"][1]["status"] == "error"

    @pytest.mark.asyncio
    async def test_setup_webhooks(
        self,
        repository_scanner: RepositoryScanner,
        mock_settings: Any,
    ) -> None:
        """Test setting up webhooks."""
        # Mock GitHub client
        mock_github_client = AsyncMock()
        mock_github_client.create_webhook = AsyncMock(
            return_value={"id": 12345, "active": True},
        )

        with patch.object(
            repository_scanner,
            "_get_github_client",
            return_value=mock_github_client,
        ):
            # Mock repository records
            repo1 = MagicMock(spec=Repository)
            repo2 = MagicMock(spec=Repository)

            results = [
                MagicMock(scalar_one_or_none=MagicMock(return_value=repo1)),
                MagicMock(scalar_one_or_none=MagicMock(return_value=repo2)),
            ]
            cast("Any", repository_scanner.db_session).execute.side_effect = results

            webhook_results = await repository_scanner.setup_webhooks()

            assert webhook_results["webhooks_created"] == 2
            assert webhook_results["failed"] == 0
            assert len(webhook_results["results"]) == 2
            assert mock_github_client.create_webhook.call_count == 2

    @pytest.mark.asyncio
    async def test_setup_webhooks_disabled(
        self,
        repository_scanner: RepositoryScanner,
        mock_settings: Any,
    ) -> None:
        """Test setup webhooks when disabled in config."""
        mock_settings.github.use_webhooks = False

        result = await repository_scanner.setup_webhooks()

        assert result["message"] == "Webhooks disabled in configuration"
