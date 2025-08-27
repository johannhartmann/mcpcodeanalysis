"""GitHub API client for repository monitoring."""

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.logger import get_logger
from src.utils.exceptions import GitHubError, RateLimitError

logger = get_logger(__name__)

# Constants
MIN_RATE_LIMIT_THRESHOLD = 10
HTTP_TOO_MANY_REQUESTS = 429
HTTP_ERROR_THRESHOLD = 400


class GitHubClient:
    """Asynchronous GitHub API client."""

    def __init__(self, access_token: str | None = None) -> None:
        self.access_token = access_token
        # Using global settings from src.config
        self.base_url = "https://api.github.com"
        self._client: httpx.AsyncClient | None = None
        self._rate_limit_remaining = 5000  # Default GitHub API rate limit
        self._rate_limit_reset: datetime | None = None

    async def __aenter__(self) -> "GitHubClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            headers=self._get_headers(),
            timeout=30.0,
        )
        return self

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "MCP-Code-Analysis-Server/0.1.0",
        }
        if self.access_token:
            headers["Authorization"] = f"token {self.access_token}"
        return headers

    async def _check_rate_limit(self) -> None:
        """Check and handle rate limiting."""
        if self._rate_limit_remaining <= MIN_RATE_LIMIT_THRESHOLD and (
            self._rate_limit_reset and datetime.now(UTC) < self._rate_limit_reset
        ):
            wait_seconds = (self._rate_limit_reset - datetime.now(UTC)).total_seconds()
            logger.warning(
                "Rate limit nearly exhausted, waiting",
                remaining=self._rate_limit_remaining,
                reset_in_seconds=wait_seconds,
            )
            await asyncio.sleep(wait_seconds)

    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure the HTTPX AsyncClient is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers=self._get_headers(),
                timeout=30.0,
            )
        return self._client

    def _update_rate_limit(self, response: httpx.Response) -> None:
        """Update rate limit info from response headers."""
        if "X-RateLimit-Remaining" in response.headers:
            self._rate_limit_remaining = int(response.headers["X-RateLimit-Remaining"])
        if "X-RateLimit-Reset" in response.headers:
            self._rate_limit_reset = datetime.fromtimestamp(
                int(response.headers["X-RateLimit-Reset"]),
                tz=UTC,
            )

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make an API request with retry logic."""
        await self._check_rate_limit()

        url = f"{self.base_url}{endpoint}"
        client = self._ensure_client()
        response = await client.request(method, url, **kwargs)

        self._update_rate_limit(response)

        if response.status_code == HTTP_TOO_MANY_REQUESTS:
            retry_after = int(response.headers.get("Retry-After", 60))
            msg = "Rate limit hit"
            raise RateLimitError(msg, retry_after=retry_after)

        if response.status_code >= HTTP_ERROR_THRESHOLD:
            error_data = {}
            with contextlib.suppress(Exception):
                error_data = response.json()

            msg = "API error"
            raise GitHubError(
                msg,
                status_code=response.status_code,
                github_error=error_data,
            )

        return response

    async def get_repository(self, owner: str, repo: str) -> dict[str, Any]:
        """Get repository information."""
        logger.info("Fetching repository info", owner=owner, repo=repo)
        response = await self._request("GET", f"/repos/{owner}/{repo}")
        result: dict[str, Any] = response.json()
        return result

    async def get_default_branch(self, owner: str, repo: str) -> str:
        """Get repository's default branch."""
        repo_info = await self.get_repository(owner, repo)
        default_branch: str = repo_info.get("default_branch", "main")
        return default_branch

    async def get_commits(
        self,
        owner: str,
        repo: str,
        branch: str = "main",
        since: datetime | None = None,
        until: datetime | None = None,
        per_page: int = 100,
    ) -> list[dict[str, Any]]:
        """Get commits from a repository."""
        logger.info(
            "Fetching commits",
            owner=owner,
            repo=repo,
            branch=branch,
            since=since,
            until=until,
        )

        params = {
            "sha": branch,
            "per_page": per_page,
        }
        if since:
            params["since"] = since.isoformat()
        if until:
            params["until"] = until.isoformat()

        commits = []
        page = 1

        while True:
            params["page"] = page
            response = await self._request(
                "GET",
                f"/repos/{owner}/{repo}/commits",
                params=params,
            )

            page_commits = response.json()
            if not page_commits:
                break

            commits.extend(page_commits)

            # Check if there are more pages
            if (
                "Link" not in response.headers
                or 'rel="next"' not in response.headers["Link"]
            ):
                break

            page += 1

        return commits

    async def get_commit_details(
        self,
        owner: str,
        repo: str,
        sha: str,
    ) -> dict[str, Any]:
        """Get detailed information about a specific commit."""
        logger.debug("Fetching commit details", owner=owner, repo=repo, sha=sha)
        response = await self._request("GET", f"/repos/{owner}/{repo}/commits/{sha}")
        result: dict[str, Any] = response.json()
        return result

    async def get_tree(
        self,
        owner: str,
        repo: str,
        tree_sha: str,
        *,
        recursive: bool = True,
    ) -> dict[str, Any]:
        """Get repository tree structure."""
        logger.debug("Fetching tree", owner=owner, repo=repo, tree_sha=tree_sha)
        params = {"recursive": 1} if recursive else {}
        response = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/git/trees/{tree_sha}",
            params=params,
        )
        result: dict[str, Any] = response.json()
        return result

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str | None = None,
    ) -> dict[str, Any]:
        """Get file content from repository."""
        logger.debug(
            "Fetching file content",
            owner=owner,
            repo=repo,
            path=path,
            ref=ref,
        )
        params = {"ref": ref} if ref else {}
        response = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/contents/{path}",
            params=params,
        )
        result: dict[str, Any] = response.json()
        return result

    async def create_webhook(
        self,
        owner: str,
        repo: str,
        url: str,
        events: list[str],
        secret: str | None = None,
    ) -> dict[str, Any]:
        """Create a webhook for repository events."""
        logger.info("Creating webhook", owner=owner, repo=repo, url=url, events=events)

        config = {"url": url, "content_type": "json"}
        if secret:
            config["secret"] = secret

        data = {
            "name": "web",
            "active": True,
            "events": events,
            "config": config,
        }

        response = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/hooks",
            json=data,
        )
        result: dict[str, Any] = response.json()
        return result

    async def delete_webhook(self, owner: str, repo: str, hook_id: int) -> None:
        """Delete a webhook."""
        logger.info("Deleting webhook", owner=owner, repo=repo, hook_id=hook_id)
        await self._request("DELETE", f"/repos/{owner}/{repo}/hooks/{hook_id}")

    async def get_rate_limit(self) -> dict[str, Any]:
        """Get current rate limit status."""
        response = await self._request("GET", "/rate_limit")
        result: dict[str, Any] = response.json()
        return result

    async def get_changed_files(
        self,
        owner: str,
        repo: str,
        base_sha: str,
        head_sha: str,
    ) -> list[dict[str, Any]]:
        """Get files changed between two commits."""
        logger.info(
            "Getting changed files",
            owner=owner,
            repo=repo,
            base_sha=base_sha,
            head_sha=head_sha,
        )

        response = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/compare/{base_sha}...{head_sha}",
        )

        data: dict[str, Any] = response.json()
        files: list[dict[str, Any]] = data.get("files", [])
        return files
