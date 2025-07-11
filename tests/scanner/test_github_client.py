"""Tests for GitHub API client."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.scanner.github_client import GitHubClient
from src.utils.exceptions import GitHubError, RateLimitError


@pytest.fixture
def github_client():
    """Create GitHub client fixture."""
    return GitHubClient(access_token="test_token")


@pytest.fixture
def mock_response():
    """Create mock HTTP response."""
    response = MagicMock(spec=httpx.Response)
    response.headers = {
        "X-RateLimit-Remaining": "4999",
        "X-RateLimit-Reset": str(int(datetime.now(UTC).timestamp()) + 3600),
    }
    response.status_code = 200
    return response


class TestGitHubClient:
    """Tests for GitHubClient class."""

    @pytest.mark.asyncio
    async def test_context_manager(self, github_client) -> None:
        """Test async context manager."""
        async with github_client as client:
            assert client._client is not None
            assert isinstance(client._client, httpx.AsyncClient)

        # Client should be closed after context
        assert client._client is None or client._client.is_closed

    def test_get_headers_with_token(self) -> None:
        """Test header generation with access token."""
        client = GitHubClient(access_token="test_token")
        headers = client._get_headers()

        assert headers["Authorization"] == "token test_token"
        assert headers["Accept"] == "application/vnd.github.v3+json"
        assert "User-Agent" in headers

    def test_get_headers_without_token(self) -> None:
        """Test header generation without access token."""
        client = GitHubClient()
        headers = client._get_headers()

        assert "Authorization" not in headers
        assert headers["Accept"] == "application/vnd.github.v3+json"

    @pytest.mark.asyncio
    async def test_check_rate_limit_ok(self, github_client) -> None:
        """Test rate limit check when limit is ok."""
        github_client._rate_limit_remaining = 100

        # Should not wait
        await github_client._check_rate_limit()

    @pytest.mark.asyncio
    async def test_check_rate_limit_exhausted(self, github_client) -> None:
        """Test rate limit check when limit is exhausted."""
        github_client._rate_limit_remaining = 5
        github_client._rate_limit_reset = datetime.now(UTC).replace(
            second=datetime.now(tz=UTC).second + 1,
        )

        with patch("asyncio.sleep") as mock_sleep:
            await github_client._check_rate_limit()
            mock_sleep.assert_called_once()

    def test_update_rate_limit(self, github_client, mock_response) -> None:
        """Test rate limit update from response headers."""
        github_client._update_rate_limit(mock_response)

        assert github_client._rate_limit_remaining == 4999
        assert github_client._rate_limit_reset is not None

    @pytest.mark.asyncio
    async def test_request_success(self, github_client, mock_response) -> None:
        """Test successful API request."""
        mock_response.json.return_value = {"id": 123, "name": "test-repo"}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        # Mock the httpx.AsyncClient creation
        with patch("httpx.AsyncClient", return_value=mock_client):
            async with github_client:
                response = await github_client._request("GET", "/repos/test/repo")

                assert response == mock_response
                mock_client.request.assert_called_once_with(
                    "GET",
                    "https://api.github.com/repos/test/repo",
                )

    @pytest.mark.asyncio
    async def test_request_rate_limit_error(self, github_client) -> None:
        """Test rate limit error handling."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.headers = {
            "Retry-After": "60",
            "X-RateLimit-Limit": "5000",
        }

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        # Mock the httpx.AsyncClient creation
        with patch("httpx.AsyncClient", return_value=mock_client):
            async with github_client:
                with pytest.raises(RateLimitError) as exc_info:
                    await github_client._request("GET", "/test")

                assert exc_info.value.details.get("retry_after") == 60

    @pytest.mark.asyncio
    async def test_request_github_error(self, github_client) -> None:
        """Test GitHub API error handling."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.json.return_value = {"message": "Not Found"}
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        # Mock the httpx.AsyncClient creation
        with patch("httpx.AsyncClient", return_value=mock_client):
            async with github_client:
                with pytest.raises(GitHubError) as exc_info:
                    await github_client._request("GET", "/test")

                assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_repository(self, github_client, mock_response) -> None:
        """Test getting repository information."""
        repo_data = {
            "id": 123,
            "name": "test-repo",
            "default_branch": "main",
        }
        mock_response.json.return_value = repo_data

        with patch.object(github_client, "_request", return_value=mock_response):
            result = await github_client.get_repository("test-owner", "test-repo")

            assert result == repo_data
            github_client._request.assert_called_once_with(
                "GET",
                "/repos/test-owner/test-repo",
            )

    @pytest.mark.asyncio
    async def test_get_default_branch(self, github_client) -> None:
        """Test getting default branch."""
        with patch.object(
            github_client,
            "get_repository",
            return_value={"default_branch": "develop"},
        ):
            branch = await github_client.get_default_branch("test-owner", "test-repo")
            assert branch == "develop"

    @pytest.mark.asyncio
    async def test_get_commits(self, github_client, mock_response) -> None:
        """Test getting commits."""
        commits_data = [
            {"sha": "abc123", "message": "First commit"},
            {"sha": "def456", "message": "Second commit"},
        ]
        mock_response.json.return_value = commits_data
        mock_response.headers = {"X-RateLimit-Remaining": "4999"}

        with patch.object(github_client, "_request", return_value=mock_response):
            commits = await github_client.get_commits(
                "test-owner",
                "test-repo",
                branch="main",
                per_page=50,
            )

            assert commits == commits_data
            github_client._request.assert_called_with(
                "GET",
                "/repos/test-owner/test-repo/commits",
                params={
                    "sha": "main",
                    "per_page": 50,
                    "page": 1,
                },
            )

    @pytest.mark.asyncio
    async def test_get_commits_pagination(self, github_client) -> None:
        """Test getting commits with pagination."""
        # First page
        response1 = MagicMock(spec=httpx.Response)
        response1.status_code = 200
        response1.json.return_value = [{"sha": "abc123"}]
        response1.headers = {
            "X-RateLimit-Remaining": "4999",
            "Link": '<https://api.github.com/repos/test/repo/commits?page=2>; rel="next"',
        }

        # Second page
        response2 = MagicMock(spec=httpx.Response)
        response2.status_code = 200
        response2.json.return_value = [{"sha": "def456"}]
        response2.headers = {"X-RateLimit-Remaining": "4998"}

        with patch.object(
            github_client,
            "_request",
            side_effect=[response1, response2],
        ):
            commits = await github_client.get_commits("test-owner", "test-repo")

            assert len(commits) == 2
            assert commits[0]["sha"] == "abc123"
            assert commits[1]["sha"] == "def456"

    @pytest.mark.asyncio
    async def test_create_webhook(self, github_client, mock_response) -> None:
        """Test creating webhook."""
        webhook_data = {
            "id": 12345,
            "url": "https://example.com/webhook",
            "active": True,
        }
        mock_response.json.return_value = webhook_data

        with patch.object(github_client, "_request", return_value=mock_response):
            result = await github_client.create_webhook(
                "test-owner",
                "test-repo",
                "https://example.com/webhook",
                ["push", "pull_request"],
                secret="webhook_secret",
            )

            assert result == webhook_data
            github_client._request.assert_called_once()

            # Check request data
            call_args = github_client._request.call_args
            assert call_args[0] == ("POST", "/repos/test-owner/test-repo/hooks")
            assert call_args[1]["json"]["config"]["secret"] == "webhook_secret"
            assert call_args[1]["json"]["events"] == ["push", "pull_request"]

    @pytest.mark.asyncio
    async def test_get_file_content(self, github_client, mock_response) -> None:
        """Test getting file content."""
        file_data = {
            "name": "test.py",
            "path": "src/test.py",
            "content": "cHJpbnQoIkhlbGxvIFdvcmxkIik=",
        }
        mock_response.json.return_value = file_data

        with patch.object(github_client, "_request", return_value=mock_response):
            result = await github_client.get_file_content(
                "test-owner",
                "test-repo",
                "src/test.py",
                ref="main",
            )

            assert result == file_data
            github_client._request.assert_called_once_with(
                "GET",
                "/repos/test-owner/test-repo/contents/src/test.py",
                params={"ref": "main"},
            )

    @pytest.mark.asyncio
    async def test_get_changed_files(self, github_client, mock_response) -> None:
        """Test getting changed files between commits."""
        compare_data = {
            "files": [
                {
                    "filename": "src/main.py",
                    "status": "modified",
                    "additions": 10,
                    "deletions": 5,
                },
                {
                    "filename": "tests/test_main.py",
                    "status": "added",
                    "additions": 50,
                    "deletions": 0,
                },
            ],
        }
        mock_response.json.return_value = compare_data

        with patch.object(github_client, "_request", return_value=mock_response):
            files = await github_client.get_changed_files(
                "test-owner",
                "test-repo",
                "abc123",
                "def456",
            )

            assert files == compare_data["files"]
            github_client._request.assert_called_once_with(
                "GET",
                "/repos/test-owner/test-repo/compare/abc123...def456",
            )
