"""Tests for Git repository synchronization."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import git
import pytest
from git.exc import GitCommandError, InvalidGitRepositoryError

from src.scanner.git_sync import GitSync
from src.utils.exceptions import RepositoryError


@pytest.fixture
def git_sync(tmp_path):
    """Create GitSync fixture with temporary storage."""
    with patch("src.scanner.git_sync.get_settings") as mock_settings:
        mock_settings.return_value.scanner.storage_path = tmp_path / "repos"
        mock_settings.return_value.scanner.exclude_patterns = [
            "__pycache__",
            "*.pyc",
            ".git",
            ".pytest_cache",
        ]
        sync = GitSync()
        yield sync


@pytest.fixture
def mock_repo():
    """Create mock git repository."""
    repo = MagicMock(spec=git.Repo)
    repo.working_dir = "/tmp/test_repo"
    repo.heads = {"main": MagicMock(), "develop": MagicMock()}
    repo.active_branch = MagicMock(name="main")
    repo.active_branch.name = "main"
    repo.remotes = MagicMock()
    repo.remotes.origin = MagicMock()
    return repo


class TestGitSync:
    """Tests for GitSync class."""

    def test_init(self, tmp_path) -> None:
        """Test GitSync initialization."""
        with patch("src.scanner.git_sync.settings") as mock_settings:
            mock_settings.scanner.root_paths = [str(tmp_path / "repos")]
            sync = GitSync()

            assert sync.storage_path == tmp_path / "repos"
            assert sync.storage_path.exists()

    def test_get_repo_path(self, git_sync) -> None:
        """Test repository path generation."""
        path = git_sync._get_repo_path("test-owner", "test-repo")
        assert path == git_sync.storage_path / "test-owner" / "test-repo"

    def testextract_owner_repo_https(self, git_sync) -> None:
        """Test extracting owner and repo from HTTPS URL."""
        owner, repo = git_sync.extract_owner_repo(
            "https://github.com/test-owner/test-repo",
        )
        assert owner == "test-owner"
        assert repo == "test-repo"

        # With .git suffix
        owner, repo = git_sync.extract_owner_repo(
            "https://github.com/test-owner/test-repo.git",
        )
        assert owner == "test-owner"
        assert repo == "test-repo"

    def testextract_owner_repo_ssh(self, git_sync) -> None:
        """Test extracting owner and repo from SSH URL."""
        owner, repo = git_sync.extract_owner_repo(
            "git@github.com:test-owner/test-repo.git",
        )
        assert owner == "test-owner"
        assert repo == "test-repo"

    def test_extract_owner_repo_invalid(self, git_sync) -> None:
        """Test extracting owner and repo from invalid URL."""
        from src.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Invalid URL"):
            git_sync.extract_owner_repo("https://example.com/test-repo")

        with pytest.raises(ValidationError, match="Invalid path"):
            git_sync.extract_owner_repo("https://github.com/invalid-path")

    @pytest.mark.asyncio
    async def test_clone_repository_success(self, git_sync, mock_repo) -> None:
        """Test successful repository cloning."""
        with patch("git.Repo.clone_from", return_value=mock_repo) as mock_clone:
            repo = await git_sync.clone_repository(
                "https://github.com/test-owner/test-repo",
                branch="main",
            )

            assert repo == mock_repo
            mock_clone.assert_called_once()

            # Check clone arguments
            call_args = mock_clone.call_args
            assert "test-owner/test-repo" in str(call_args[0][1])
            assert call_args[1]["branch"] == "main"
            assert call_args[1]["depth"] == 1

    @pytest.mark.asyncio
    async def test_clone_repository_with_token(self, git_sync, mock_repo) -> None:
        """Test repository cloning with access token."""
        with patch("git.Repo.clone_from", return_value=mock_repo) as mock_clone:
            await git_sync.clone_repository(
                "https://github.com/test-owner/test-repo",
                access_token="test_token",
            )

            # Check that token was inserted into URL
            call_args = mock_clone.call_args
            assert "test_token@github.com" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_clone_repository_already_exists(self, git_sync, mock_repo) -> None:
        """Test cloning when repository already exists."""
        repo_path = git_sync._get_repo_path("test-owner", "test-repo")
        repo_path.mkdir(parents=True)

        with (
            patch("git.Repo", return_value=mock_repo),
            patch.object(
                git_sync,
                "update_repository",
                return_value=mock_repo,
            ) as mock_update,
        ):
            repo = await git_sync.clone_repository(
                "https://github.com/test-owner/test-repo",
            )

            assert repo == mock_repo
            mock_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_clone_repository_git_error(self, git_sync) -> None:
        """Test repository cloning with git error."""
        with (
            patch(
                "git.Repo.clone_from",
                side_effect=GitCommandError("clone", "error"),
            ),
            pytest.raises(RepositoryError, match="Clone failed"),
        ):
            await git_sync.clone_repository(
                "https://github.com/test-owner/test-repo",
            )

    @pytest.mark.asyncio
    async def test_update_repository_success(self, git_sync, mock_repo) -> None:
        """Test successful repository update."""
        repo_path = git_sync._get_repo_path("test-owner", "test-repo")
        repo_path.mkdir(parents=True)

        with patch("git.Repo", return_value=mock_repo):
            repo = await git_sync.update_repository(
                "https://github.com/test-owner/test-repo",
                branch="main",
            )

            assert repo == mock_repo
            mock_repo.remotes.origin.fetch.assert_called_once()
            mock_repo.remotes.origin.pull.assert_called_once_with("main")

    @pytest.mark.asyncio
    async def test_update_repository_not_exists(self, git_sync, mock_repo) -> None:
        """Test updating non-existent repository."""
        with patch.object(
            git_sync,
            "clone_repository",
            return_value=mock_repo,
        ) as mock_clone:
            repo = await git_sync.update_repository(
                "https://github.com/test-owner/test-repo",
            )

            assert repo == mock_repo
            mock_clone.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_repository_invalid_repo(self, git_sync, mock_repo) -> None:
        """Test updating invalid repository directory."""
        repo_path = git_sync._get_repo_path("test-owner", "test-repo")
        repo_path.mkdir(parents=True)

        with patch("git.Repo", side_effect=InvalidGitRepositoryError):
            with patch.object(
                git_sync,
                "clone_repository",
                return_value=mock_repo,
            ) as mock_clone:
                repo = await git_sync.update_repository(
                    "https://github.com/test-owner/test-repo",
                )

                assert repo == mock_repo
                mock_clone.assert_called_once()
                assert not repo_path.exists()  # Should be removed

    def test_get_repository_exists(self, git_sync, mock_repo) -> None:
        """Test getting existing repository."""
        repo_path = git_sync._get_repo_path("test-owner", "test-repo")
        repo_path.mkdir(parents=True)

        with patch("git.Repo", return_value=mock_repo):
            repo = git_sync.get_repository("https://github.com/test-owner/test-repo")
            assert repo == mock_repo

    def test_get_repository_not_exists(self, git_sync) -> None:
        """Test getting non-existent repository."""
        repo = git_sync.get_repository("https://github.com/test-owner/test-repo")
        assert repo is None

    def test_get_file_hash(self, git_sync, tmp_path) -> None:
        """Test file hash calculation."""
        test_file = tmp_path / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        hash_value = git_sync.get_file_hash(test_file)

        # Verify hash
        expected_hash = hashlib.sha256(test_content).hexdigest()
        assert hash_value == expected_hash

    @pytest.mark.asyncio
    async def test_get_changed_files(self, git_sync, mock_repo) -> None:
        """Test getting changed files."""
        # Mock diff items
        diff_item1 = MagicMock()
        diff_item1.a_path = "src/old.py"
        diff_item1.b_path = "src/new.py"
        diff_item1.new_file = False
        diff_item1.deleted_file = False
        diff_item1.renamed_file = True

        diff_item2 = MagicMock()
        diff_item2.a_path = None
        diff_item2.b_path = "src/added.py"
        diff_item2.new_file = True
        diff_item2.deleted_file = False
        diff_item2.renamed_file = False

        mock_commit = MagicMock()
        mock_commit.diff.return_value = [diff_item1, diff_item2]
        mock_repo.commit.return_value = mock_commit
        mock_repo.head.commit = MagicMock()

        changed_files = await git_sync.get_changed_files(mock_repo, "abc123")

        assert len(changed_files) == 2
        # For renamed files, the key is the old path (a_path)
        assert changed_files["src/old.py"]["change_type"] == "renamed"
        assert changed_files["src/old.py"]["new_path"] == "src/new.py"
        assert changed_files["src/added.py"]["change_type"] == "added"

    @pytest.mark.asyncio
    async def test_scan_repository_files(self, git_sync, tmp_path) -> None:
        """Test scanning repository files."""
        # Create mock repository structure
        repo_path = tmp_path / "test_repo"
        src_path = repo_path / "src"
        src_path.mkdir(parents=True)

        # Create test files
        (src_path / "main.py").write_text("print('main')")
        (src_path / "utils.py").write_text("print('utils')")
        (src_path / "data.json").write_text('{"test": true}')

        # Create excluded directory
        pycache_path = src_path / "__pycache__"
        pycache_path.mkdir()
        (pycache_path / "main.cpython-39.pyc").write_text("compiled")

        mock_repo = MagicMock()
        mock_repo.working_dir = str(repo_path)

        files = await git_sync.scan_repository_files(mock_repo, {".py"})

        assert len(files) == 2
        file_paths = [f["path"] for f in files]
        assert "src/main.py" in file_paths
        assert "src/utils.py" in file_paths
        assert "src/data.json" not in file_paths
        assert "__pycache__" not in str(file_paths)

    def test_detect_language(self, git_sync) -> None:
        """Test language detection from file extension."""
        test_cases = [
            (Path("test.py"), "python"),
            (Path("test.js"), "javascript"),
            (Path("test.ts"), "typescript"),
            (Path("test.java"), "java"),
            (Path("test.cpp"), "cpp"),
            (Path("test.rs"), "rust"),
            (Path("test.unknown"), "unknown"),
        ]

        for file_path, expected_language in test_cases:
            assert git_sync._detect_language(file_path) == expected_language

    def test_get_commit_info(self, git_sync, mock_repo) -> None:
        """Test getting commit information."""
        mock_commit = MagicMock()
        mock_commit.hexsha = "abc123def456"
        mock_commit.message = "Test commit message\n"
        mock_commit.author.name = "Test Author"
        mock_commit.author.email = "test@example.com"
        mock_commit.committed_date = 1234567890
        mock_commit.stats.files = {"file1.py": {}, "file2.py": {}}
        mock_commit.stats.total = {"insertions": 10, "deletions": 5}

        mock_repo.commit.return_value = mock_commit

        info = git_sync.get_commit_info(mock_repo, "abc123")

        assert info["sha"] == "abc123def456"
        assert info["message"] == "Test commit message"
        assert info["author"] == "Test Author"
        assert info["author_email"] == "test@example.com"
        assert info["files_changed"] == ["file1.py", "file2.py"]
        assert info["additions"] == 10
        assert info["deletions"] == 5

    @pytest.mark.asyncio
    async def test_get_recent_commits(self, git_sync, mock_repo) -> None:
        """Test getting recent commits."""
        # Create mock commits
        commits = []
        for i in range(3):
            commit = MagicMock()
            commit.hexsha = f"commit{i}"
            commit.message = f"Commit {i}"
            commit.author.name = "Test Author"
            commit.author.email = "test@example.com"
            commit.committed_date = 1234567890 + i * 3600
            commit.stats.files = {}
            commit.stats.total = {"insertions": i, "deletions": 0}
            commits.append(commit)

        mock_repo.iter_commits.return_value = commits

        with patch.object(git_sync, "get_commit_info") as mock_get_info:
            mock_get_info.side_effect = lambda r, sha: {
                "sha": sha,
                "message": f"Commit {sha[-1]}",
            }

            recent_commits = await git_sync.get_recent_commits(
                mock_repo,
                branch="main",
                limit=10,
            )

            assert len(recent_commits) == 3
            assert recent_commits[0]["sha"] == "commit0"
            assert recent_commits[2]["sha"] == "commit2"
