"""Tests for repository management tools."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Repository
from src.mcp_server.tools.repository_management import RepositoryManagementTools


@pytest.fixture
def mock_db_session():
    """Create mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_mcp():
    """Create mock FastMCP instance."""
    mcp = MagicMock(spec=FastMCP)
    mcp.tool = MagicMock(side_effect=lambda **kwargs: lambda func: func)
    return mcp


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings."""
    with patch("langchain_openai.OpenAIEmbeddings") as mock_class:
        mock_instance = MagicMock()
        mock_instance.aembed_query = AsyncMock(return_value=[0.1] * 1536)
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def repo_tools(mock_db_session, mock_mcp, mock_embeddings):
    """Create repository management tools fixture."""
    with (
        patch("src.embeddings.embedding_generator.settings") as mock_gen_settings,
        patch("src.embeddings.embedding_generator.OpenAIEmbeddings") as mock_gen_openai,
    ):
        # Configure settings
        mock_gen_settings.openai_api_key.get_secret_value.return_value = "test-key"
        mock_gen_settings.embeddings.model = "text-embedding-ada-002"

        # Use the mock embeddings fixture
        mock_gen_openai.return_value = mock_embeddings

        return RepositoryManagementTools(mock_db_session, mock_mcp)


class TestRepositoryManagementTools:
    """Tests for repository management tools."""

    @pytest.mark.asyncio
    async def test_register_tools(self, repo_tools, mock_mcp):
        """Test tool registration."""
        await repo_tools.register_tools()

        # Should register at least 7 tools
        assert mock_mcp.tool.call_count >= 7

        # Check tool names
        tool_names = [call[1]["name"] for call in mock_mcp.tool.call_args_list]
        expected_tools = [
            "add_repository",
            "list_repositories",
            "scan_repository",
            "update_repository",
            "remove_repository",
            "get_repository_stats",
            "sync_repository",
        ]
        for tool in expected_tools:
            assert tool in tool_names

    @pytest.mark.asyncio
    async def test_add_repository_new(self, repo_tools, mock_db_session):
        """Test adding a new repository."""
        # Mock no existing repository
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        # Mock scanner
        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(
            return_value={
                "repository_id": 1,
                "files_scanned": 50,
                "functions_found": 200,
                "classes_found": 30,
            }
        )

        with patch(
            "src.mcp_server.tools.repository_management.RepositoryScanner",
            return_value=mock_scanner,
        ):
            result = await repo_tools.add_repository(
                url="https://github.com/test/repo",
                name="test-repo",
                branch="main",
            )

        assert result["status"] == "success"
        assert result["repository"]["url"] == "https://github.com/test/repo"
        assert result["repository"]["name"] == "test-repo"
        assert result["scan_result"]["files_scanned"] == 50

        # Verify repository was added to session
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_repository_already_exists(self, repo_tools, mock_db_session):
        """Test adding a repository that already exists."""
        # Mock existing repository
        mock_repo = MagicMock(spec=Repository)
        mock_repo.id = 1
        mock_repo.github_url = "https://github.com/test/repo"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_repo

        mock_db_session.execute.return_value = mock_result

        result = await repo_tools.add_repository(
            url="https://github.com/test/repo",
            name="test-repo",
        )

        assert result["status"] == "error"
        assert "already exists" in result["error"]

    @pytest.mark.asyncio
    async def test_add_repository_with_access_token(self, repo_tools, mock_db_session):
        """Test adding a private repository with access token."""
        # Mock no existing repository
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        # Mock scanner
        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(
            return_value={
                "repository_id": 1,
                "files_scanned": 100,
                "private": True,
            }
        )

        with patch(
            "src.mcp_server.tools.repository_management.RepositoryScanner",
            return_value=mock_scanner,
        ):
            result = await repo_tools.add_repository(
                url="https://github.com/test/private-repo",
                name="private-repo",
                access_token="ghp_testtoken123",
            )

        assert result["status"] == "success"
        assert result["scan_result"]["private"] is True

        # Verify token was set
        added_repo = mock_db_session.add.call_args[0][0]
        assert added_repo.access_token == "ghp_testtoken123"

    @pytest.mark.asyncio
    async def test_list_repositories_empty(self, repo_tools, mock_db_session):
        """Test listing repositories when none exist."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db_session.execute.return_value = mock_result

        result = await repo_tools.list_repositories()

        assert result["repositories"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_list_repositories_with_stats(self, repo_tools, mock_db_session):
        """Test listing repositories with statistics."""
        # Mock repositories
        repos = []
        for i, (name, owner, last_synced) in enumerate(
            [
                ("repo1", "owner1", datetime.now(UTC)),
                ("repo2", "owner2", None),
                ("repo3", "owner3", datetime(2024, 1, 1, tzinfo=UTC)),
            ]
        ):
            repo = MagicMock(spec=Repository)
            repo.id = i + 1
            repo.name = name
            repo.owner = owner
            repo.github_url = f"https://github.com/{owner}/{name}"
            repo.default_branch = "main"
            repo.last_synced = last_synced
            repo.created_at = datetime(2023, 1, 1, tzinfo=UTC)
            repos.append(repo)

        repo_result = MagicMock()
        repo_result.scalars.return_value.all.return_value = repos

        # Mock stats for each repository
        stats_results = [
            MagicMock(
                __iter__=lambda self: iter(
                    [
                        (100, 50, 500, 2000, 10000)
                    ]  # files, modules, functions, classes, loc
                )
            ),
            MagicMock(__iter__=lambda self: iter([(50, 25, 250, 1000, 5000)])),
            MagicMock(__iter__=lambda self: iter([(0, 0, 0, 0, 0)])),  # Empty repo
        ]

        mock_db_session.execute.side_effect = [repo_result, *stats_results]

        result = await repo_tools.list_repositories(include_stats=True)

        assert result["total"] == 3
        assert len(result["repositories"]) == 3

        # Check first repository with stats
        repo1 = result["repositories"][0]
        assert repo1["name"] == "repo1"
        assert repo1["stats"]["file_count"] == 100
        assert repo1["stats"]["function_count"] == 500
        assert repo1["stats"]["total_lines"] == 10000
        assert repo1["last_synced"] is not None

        # Check repository with no sync
        repo2 = result["repositories"][1]
        assert repo2["last_synced"] is None

        # Check empty repository
        repo3 = result["repositories"][2]
        assert repo3["stats"]["file_count"] == 0

    @pytest.mark.asyncio
    async def test_update_repository_not_found(self, repo_tools, mock_db_session):
        """Test updating non-existent repository."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        result = await repo_tools.update_repository(
            repository_id=999,
            name="new-name",
        )

        assert result["status"] == "error"
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_update_repository_success(self, repo_tools, mock_db_session):
        """Test successfully updating repository."""
        # Mock existing repository
        mock_repo = MagicMock(spec=Repository)
        mock_repo.id = 1
        mock_repo.name = "old-name"
        mock_repo.default_branch = "main"
        mock_repo.access_token = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_repo

        mock_db_session.execute.return_value = mock_result

        result = await repo_tools.update_repository(
            repository_id=1,
            name="new-name",
            branch="develop",
            access_token="new-token",
        )

        assert result["status"] == "success"
        assert mock_repo.name == "new-name"
        assert mock_repo.default_branch == "develop"
        assert mock_repo.access_token == "new-token"

        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_repository_not_found(self, repo_tools, mock_db_session):
        """Test removing non-existent repository."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        result = await repo_tools.remove_repository(repository_id=999)

        assert result["status"] == "error"
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_remove_repository_success(self, repo_tools, mock_db_session):
        """Test successfully removing repository."""
        # Mock existing repository
        mock_repo = MagicMock(spec=Repository)
        mock_repo.id = 1
        mock_repo.name = "test-repo"
        mock_repo.github_url = "https://github.com/test/repo"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_repo

        mock_db_session.execute.return_value = mock_result

        result = await repo_tools.remove_repository(repository_id=1)

        assert result["status"] == "success"
        assert result["repository"]["name"] == "test-repo"

        mock_db_session.delete.assert_called_once_with(mock_repo)
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_repository_stats_not_found(self, repo_tools, mock_db_session):
        """Test getting stats for non-existent repository."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        result = await repo_tools.get_repository_stats(repository_id=999)

        assert result["status"] == "error"
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_get_repository_stats_detailed(self, repo_tools, mock_db_session):
        """Test getting detailed repository statistics."""
        # Mock repository
        mock_repo = MagicMock(spec=Repository)
        mock_repo.id = 1
        mock_repo.name = "test-repo"
        mock_repo.last_synced = datetime.now(UTC)

        repo_result = MagicMock()
        repo_result.scalar_one_or_none.return_value = mock_repo

        # Mock basic stats
        basic_stats_result = MagicMock()
        basic_stats_result.one.return_value = (150, 75, 600, 3000, 15000)

        # Mock language distribution
        lang_result = MagicMock()
        lang_result.__iter__ = MagicMock(
            return_value=iter(
                [
                    ("Python", 100, 10000),
                    ("JavaScript", 30, 3000),
                    ("TypeScript", 20, 2000),
                ]
            )
        )

        # Mock complexity stats
        complexity_result = MagicMock()
        complexity_result.one.return_value = (5.5, 25, 120)

        # Mock largest files
        largest_files_result = MagicMock()
        largest_files_result.__iter__ = MagicMock(
            return_value=iter(
                [
                    ("/src/large_module.py", 2000),
                    ("/src/data_processor.py", 1500),
                    ("/tests/test_integration.py", 1200),
                ]
            )
        )

        # Mock most complex functions
        complex_functions_result = MagicMock()
        complex_functions_result.__iter__ = MagicMock(
            return_value=iter(
                [
                    ("process_data", "/src/processor.py", 25),
                    ("calculate_metrics", "/src/analyzer.py", 22),
                    ("validate_input", "/src/validator.py", 20),
                ]
            )
        )

        mock_db_session.execute.side_effect = [
            repo_result,
            basic_stats_result,
            lang_result,
            complexity_result,
            largest_files_result,
            complex_functions_result,
        ]

        result = await repo_tools.get_repository_stats(repository_id=1)

        assert result["status"] == "success"
        assert result["repository"]["name"] == "test-repo"

        stats = result["stats"]
        assert stats["file_count"] == 150
        assert stats["module_count"] == 75
        assert stats["function_count"] == 600
        assert stats["class_count"] == 3000
        assert stats["total_lines"] == 15000

        # Check language distribution
        assert len(stats["language_distribution"]) == 3
        assert stats["language_distribution"][0]["language"] == "Python"
        assert stats["language_distribution"][0]["percentage"] == pytest.approx(
            66.67, 0.1
        )

        # Check complexity
        assert stats["avg_complexity"] == 5.5
        assert stats["max_complexity"] == 25

        # Check detailed lists
        assert len(stats["largest_files"]) == 3
        assert stats["largest_files"][0]["lines"] == 2000

        assert len(stats["most_complex_functions"]) == 3
        assert stats["most_complex_functions"][0]["complexity"] == 25

    @pytest.mark.asyncio
    async def test_scan_repository_not_found(self, repo_tools, mock_db_session):
        """Test scanning non-existent repository."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        result = await repo_tools.scan_repository(repository_id=999)

        assert result["status"] == "error"
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_scan_repository_full_scan(self, repo_tools, mock_db_session):
        """Test full repository scan."""
        # Mock repository
        mock_repo = MagicMock(spec=Repository)
        mock_repo.id = 1
        mock_repo.name = "test-repo"
        mock_repo.github_url = "https://github.com/test/repo"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_repo

        mock_db_session.execute.return_value = mock_result

        # Mock scanner
        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(
            return_value={
                "repository_id": 1,
                "files_scanned": 200,
                "files_updated": 150,
                "files_deleted": 10,
                "functions_found": 800,
                "classes_found": 100,
                "scan_time": 45.5,
            }
        )

        with patch(
            "src.mcp_server.tools.repository_management.RepositoryScanner",
            return_value=mock_scanner,
        ):
            result = await repo_tools.scan_repository(repository_id=1, full_scan=True)

        assert result["status"] == "success"
        assert result["repository"]["name"] == "test-repo"
        assert result["scan_result"]["files_scanned"] == 200
        assert result["scan_result"]["scan_time"] == 45.5

        # Verify full scan was requested
        mock_scanner.scan_repository.assert_called_once_with(
            mock_repo, force_full_scan=True
        )

    @pytest.mark.asyncio
    async def test_sync_repository_incremental(self, repo_tools, mock_db_session):
        """Test incremental repository sync."""
        # Mock repository with last sync time
        mock_repo = MagicMock(spec=Repository)
        mock_repo.id = 1
        mock_repo.name = "test-repo"
        mock_repo.last_synced = datetime(2024, 1, 1, tzinfo=UTC)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_repo

        mock_db_session.execute.return_value = mock_result

        # Mock git sync
        with patch(
            "src.mcp_server.tools.repository_management.GitSync"
        ) as mock_git_sync_class:
            mock_git_sync = MagicMock()
            mock_git_sync.sync_repository = AsyncMock(
                return_value={
                    "status": "success",
                    "commits_processed": 15,
                    "files_changed": 25,
                }
            )
            mock_git_sync_class.return_value = mock_git_sync

            result = await repo_tools.sync_repository(repository_id=1)

        assert result["status"] == "success"
        assert result["sync_result"]["commits_processed"] == 15
        assert result["sync_result"]["files_changed"] == 25

    @pytest.mark.asyncio
    async def test_add_repository_scanner_error(self, repo_tools, mock_db_session):
        """Test handling scanner error when adding repository."""
        # Mock no existing repository
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        # Mock scanner error
        mock_scanner = MagicMock()
        mock_scanner.scan_repository = AsyncMock(
            side_effect=Exception("Failed to clone repository")
        )

        with patch(
            "src.mcp_server.tools.repository_management.RepositoryScanner",
            return_value=mock_scanner,
        ):
            result = await repo_tools.add_repository(
                url="https://github.com/test/repo",
                name="test-repo",
            )

        assert result["status"] == "error"
        assert "scan failed" in result["error"]

        # Repository should be rolled back
        mock_db_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_repositories_filter_by_language(
        self, repo_tools, mock_db_session
    ):
        """Test listing repositories filtered by language."""
        # This would be a future enhancement - placeholder test
        repos = []
        for i, name in enumerate(["python-repo", "js-repo", "rust-repo"]):
            repo = MagicMock(spec=Repository)
            repo.id = i + 1
            repo.name = name
            repo.owner = "test"
            repo.github_url = f"https://github.com/test/{name}"
            repo.default_branch = "main"
            repo.last_synced = None
            repos.append(repo)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = repos

        mock_db_session.execute.return_value = mock_result

        result = await repo_tools.list_repositories()

        assert result["total"] == 3
        assert len(result["repositories"]) == 3
