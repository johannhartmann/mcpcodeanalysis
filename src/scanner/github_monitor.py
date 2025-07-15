"""GitHub repository monitoring for the MCP Code Analysis Server."""

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from src.config import settings
from src.logger import get_logger
from src.utils.exceptions import GitHubError

logger = get_logger(__name__)

# Constants
MIN_URL_PARTS = 2
HTTP_NOT_FOUND = 404
HTTP_UNAUTHORIZED = 401
HTTP_UNPROCESSABLE_ENTITY = 422
COMMITS_PER_PAGE = 100
MAX_PAGES = 10  # Max 1000 commits


class GitHubMonitor:
    """Monitor GitHub repositories for changes."""

    def __init__(self) -> None:
        self.github_config = getattr(settings, "github", {})
        self.repositories = settings.repositories
        self.client = httpx.AsyncClient(
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "MCP-Code-Analysis-Server",
            },
            timeout=30.0,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def get_repository_info(
        self,
        repo_url: str,
        token: str | None = None,
    ) -> dict[str, Any]:
        """Get repository information from GitHub."""
        # Parse GitHub URL
        parts = repo_url.rstrip("/").split("/")
        if len(parts) < MIN_URL_PARTS:
            msg = "Invalid URL"
            raise GitHubError(msg)

        owner = parts[-2]
        repo = parts[-1].replace(".git", "")

        # Set auth header if token provided
        headers = {}
        if token:
            headers["Authorization"] = f"token {token}"

        # Get repository info
        url = f"https://api.github.com/repos/{owner}/{repo}"

        try:
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            return {
                "owner": owner,
                "name": repo,
                "default_branch": data.get("default_branch", "main"),
                "clone_url": data.get("clone_url", repo_url),
                "private": data.get("private", False),
                "size": data.get("size", 0),
                "language": data.get("language"),
                "description": data.get("description"),
                "topics": data.get("topics", []),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "pushed_at": data.get("pushed_at"),
            }
        except httpx.HTTPStatusError as e:
            if e.response.status_code == HTTP_NOT_FOUND:
                msg = "Not found"
                raise GitHubError(msg) from e
            if e.response.status_code == HTTP_UNAUTHORIZED:
                msg = "Auth failed"
                raise GitHubError(msg) from e
            msg = "API error"
            raise GitHubError(msg) from e
        except (AttributeError, KeyError, ValueError, TypeError) as e:
            msg = "Fetch failed"
            raise GitHubError(msg) from e

    async def get_commits_since(
        self,
        owner: str,
        repo: str,
        since: datetime,
        branch: str = "main",
        token: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get commits since a specific date."""
        headers = {}
        if token:
            headers["Authorization"] = f"token {token}"

        # Format date for GitHub API
        since_str = since.isoformat() + "Z"

        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params: dict[str, str | int] = {
            "sha": branch,
            "since": since_str,
            "per_page": 100,
        }

        commits = []
        page = 1

        try:
            while True:
                params["page"] = page
                response = await self.client.get(url, headers=headers, params=params)
                response.raise_for_status()

                page_commits = response.json()
                if not page_commits:
                    break

                for commit_data in page_commits:
                    commits.append(
                        {
                            "sha": commit_data["sha"],
                            "message": commit_data["commit"]["message"],
                            "author": commit_data["commit"]["author"]["name"],
                            "author_email": commit_data["commit"]["author"]["email"],
                            "timestamp": datetime.fromisoformat(
                                commit_data["commit"]["author"]["date"].replace(
                                    "Z",
                                    "+00:00",
                                ),
                            ),
                            "url": commit_data["html_url"],
                        },
                    )

                # Check if there are more pages
                if len(page_commits) < COMMITS_PER_PAGE:
                    break

                page += 1

                # Rate limit check
                if page > MAX_PAGES:  # Max 1000 commits
                    logger.warning(
                        "Too many commits for %s/%s, stopping at 1000",
                        owner,
                        repo,
                    )
                    break

            return commits

        except httpx.HTTPStatusError as e:
            msg = "Get commits failed"
            raise GitHubError(msg) from e
        except (AttributeError, KeyError, ValueError, TypeError) as e:
            msg = "Fetch error"
            raise GitHubError(msg) from e

    async def get_commit_files(
        self,
        owner: str,
        repo: str,
        sha: str,
        token: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get files changed in a specific commit."""
        headers = {}
        if token:
            headers["Authorization"] = f"token {token}"

        url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"

        try:
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            files = []

            for file_data in data.get("files", []):
                files.append(
                    {
                        "filename": file_data["filename"],
                        "status": file_data["status"],  # added, removed, modified
                        "additions": file_data.get("additions", 0),
                        "deletions": file_data.get("deletions", 0),
                        "changes": file_data.get("changes", 0),
                        "patch": file_data.get("patch"),
                    },
                )

            return files

        except httpx.HTTPStatusError as e:
            msg = "Get files failed"
            raise GitHubError(msg) from e
        except (AttributeError, KeyError, ValueError, TypeError) as e:
            msg = "Fetch error"
            raise GitHubError(msg) from e

    async def check_rate_limit(self, token: str | None = None) -> dict[str, Any]:
        """Check GitHub API rate limit."""
        headers = {}
        if token:
            headers["Authorization"] = f"token {token}"

        url = "https://api.github.com/rate_limit"

        try:
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            core_limit = data["resources"]["core"]

            return {
                "limit": core_limit["limit"],
                "remaining": core_limit["remaining"],
                "reset": datetime.fromtimestamp(core_limit["reset"], tz=UTC),
                "used": core_limit["limit"] - core_limit["remaining"],
            }

        except Exception:
            logger.exception("Failed to check rate limit: %s")
            return {
                "limit": 60,  # Default for unauthenticated
                "remaining": 0,
                "reset": datetime.now(tz=UTC) + timedelta(hours=1),
                "used": 60,
            }

    async def setup_webhook(
        self,
        owner: str,
        repo: str,
        webhook_url: str,
        token: str,
        events: list[str] | None = None,
    ) -> dict[str, Any]:
        """Set up a webhook for repository events."""
        if events is None:
            events = ["push", "pull_request", "release"]

        headers = {"Authorization": f"token {token}"}

        url = f"https://api.github.com/repos/{owner}/{repo}/hooks"

        payload = {
            "name": "web",
            "active": True,
            "events": events,
            "config": {
                "url": webhook_url,
                "content_type": "json",
                "secret": self.github_config.get("webhook_secret"),
            },
        }

        try:
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            result: dict[str, Any] = response.json()
            return result

        except httpx.HTTPStatusError as e:
            if e.response.status_code == HTTP_UNPROCESSABLE_ENTITY:
                # Webhook might already exist
                logger.warning("Webhook may already exist for %s/%s", owner, repo)
                return {"status": "exists"}
            msg = "Create failed"
            raise GitHubError(msg) from e
        except (AttributeError, KeyError, ValueError, TypeError) as e:
            msg = "Create error"
            raise GitHubError(msg) from e

    async def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature."""
        if not self.github_config.get("webhook_secret"):
            return True  # No secret configured

        import hashlib
        import hmac

        expected_signature = (
            "sha256="
            + hmac.new(
                self.github_config.get("webhook_secret").encode(),
                payload,
                hashlib.sha256,
            ).hexdigest()
        )

        return hmac.compare_digest(expected_signature, signature)
