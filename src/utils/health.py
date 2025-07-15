"""Health check utilities for MCP Code Analysis Server."""

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import get_database_url, settings
from src.logger import get_logger
from src.utils.version import get_package_version

logger = get_logger(__name__)

# Constants
HTTP_OK = 200
HTTP_UNAUTHORIZED = 401
DISK_CRITICAL_THRESHOLD = 90


class HealthStatus:
    """Health check status constants."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck:
    """Base health check class."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def check(self) -> dict[str, Any]:
        """Perform health check."""
        start_time = datetime.now(tz=UTC)
        try:
            result = await self._perform_check()
            status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
            details = result if isinstance(result, dict) else {}
        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.exception("Health check failed: %s", self.name)
            status = HealthStatus.UNHEALTHY
            details = {"error": str(e)}

        end_time = datetime.now(tz=UTC)
        duration_ms = (end_time - start_time).total_seconds() * 1000

        return {
            "name": self.name,
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "timestamp": end_time.isoformat(),
            "details": details,
        }

    async def _perform_check(self) -> Any:
        """Perform the actual health check (to be implemented by subclasses)."""
        raise NotImplementedError


class DatabaseHealthCheck(HealthCheck):
    """Database connectivity health check."""

    def __init__(self) -> None:
        super().__init__("database")
        # Using global settings from src.config

    async def _perform_check(self) -> dict[str, Any]:
        """Check database connectivity and pgvector extension."""
        engine = create_async_engine(
            get_database_url(),
            pool_size=1,
            max_overflow=0,
        )

        try:
            async with engine.begin() as conn:
                # Check basic connectivity
                result: Any = await conn.execute(text("SELECT 1"))
                result.scalar()

                # Check pgvector extension
                vector_result: Any = await conn.execute(
                    text(
                        "SELECT installed_version FROM pg_available_extensions WHERE name = 'vector'",
                    ),
                )
                vector_version = vector_result.scalar()

                # Get database size
                size_result: Any = await conn.execute(
                    text("SELECT pg_database_size(current_database())"),
                )
                db_size_bytes = size_result.scalar()

                # Get table counts
                tables_result: Any = await conn.execute(
                    text(
                        """
                        SELECT
                            (SELECT COUNT(*) FROM repositories) as repositories,
                            (SELECT COUNT(*) FROM files) as files,
                            (SELECT COUNT(*) FROM code_embeddings) as embeddings
                    """,
                    ),
                )
                counts = tables_result.first()

                return {
                    "connected": True,
                    "pgvector_version": vector_version,
                    "database_size_mb": round(db_size_bytes / (1024 * 1024), 2),
                    "table_counts": {
                        "repositories": counts.repositories if counts else 0,
                        "files": counts.files if counts else 0,
                        "embeddings": counts.embeddings if counts else 0,
                    },
                }
        finally:
            await engine.dispose()


class GitHubHealthCheck(HealthCheck):
    """GitHub API connectivity health check."""

    def __init__(self) -> None:
        super().__init__("github")
        # Using global settings from src.config

    async def _perform_check(self) -> dict[str, Any]:
        """Check GitHub API connectivity and rate limits."""
        async with httpx.AsyncClient() as client:
            # Check API status
            response = await client.get(
                "https://api.github.com/rate_limit",
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=10.0,
            )

            if response.status_code != HTTP_OK:
                return {"connected": False, "status_code": response.status_code}

            data = response.json()
            rate_limit = data.get("rate", {})

            return {
                "connected": True,
                "rate_limit": {
                    "limit": rate_limit.get("limit", 0),
                    "remaining": rate_limit.get("remaining", 0),
                    "reset": (
                        datetime.fromtimestamp(
                            rate_limit.get("reset", 0), tz=UTC
                        ).isoformat()
                        if rate_limit.get("reset")
                        else None
                    ),
                },
            }


class OpenAIHealthCheck(HealthCheck):
    """OpenAI API connectivity health check."""

    def __init__(self) -> None:
        super().__init__("openai")
        # Using global settings from src.config

    async def _perform_check(self) -> dict[str, Any]:
        """Check OpenAI API connectivity."""
        async with httpx.AsyncClient() as client:
            # Use models endpoint as a simple connectivity check
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}",
                },
                timeout=10.0,
            )

            if response.status_code == HTTP_UNAUTHORIZED:
                return {"connected": False, "error": "Invalid API key"}
            if response.status_code != HTTP_OK:
                return {"connected": False, "status_code": response.status_code}

            data = response.json()
            embedding_models = [
                model["id"]
                for model in data.get("data", [])
                if "embedding" in model["id"]
            ]

            return {
                "connected": True,
                "embedding_models_available": embedding_models,
                "configured_model": settings.embeddings.model,
                "model_available": settings.embeddings.model in embedding_models,
            }


class DiskSpaceHealthCheck(HealthCheck):
    """Disk space health check."""

    def __init__(self) -> None:
        super().__init__("disk_space")
        # Using global settings from src.config

    async def _perform_check(self) -> dict[str, Any]:
        """Check available disk space."""
        import shutil

        paths_to_check = {
            "repositories": (
                settings.scanner.root_paths[0] if settings.scanner.root_paths else "."
            ),
            "cache": settings.embeddings.cache_dir,
            "logs": Path(settings.logging.file_path).parent,
        }

        results = {}
        for name, path_value in paths_to_check.items():
            path = Path(path_value) if isinstance(path_value, str) else path_value
            if path.exists():
                stat = shutil.disk_usage(path)
                results[name] = {
                    "total_gb": round(stat.total / (1024**3), 2),
                    "used_gb": round(stat.used / (1024**3), 2),
                    "free_gb": round(stat.free / (1024**3), 2),
                    "used_percent": round((stat.used / stat.total) * 100, 2),
                }

        # Check if any disk is critically low (< 10% free)
        critical = any(
            disk["used_percent"] > DISK_CRITICAL_THRESHOLD for disk in results.values()
        )

        return {
            "paths": results,
            "critical_space": critical,
        }


class HealthCheckManager:
    """Manages all health checks."""

    def __init__(self) -> None:
        self.checks: list[HealthCheck] = [
            DatabaseHealthCheck(),
            GitHubHealthCheck(),
            OpenAIHealthCheck(),
            DiskSpaceHealthCheck(),
        ]

    async def check_all(self) -> dict[str, Any]:
        """Run all health checks."""
        start_time = datetime.now(tz=UTC)

        # Run all checks concurrently
        check_results = await asyncio.gather(
            *[check.check() for check in self.checks],
            return_exceptions=True,
        )

        # Process results
        checks: list[dict[str, Any]] = []
        overall_status = HealthStatus.HEALTHY

        for result in check_results:
            if isinstance(result, Exception):
                logger.exception("Health check failed")
                checks.append(
                    {
                        "name": "unknown",
                        "status": HealthStatus.UNHEALTHY,
                        "error": str(result),
                    },
                )
                overall_status = HealthStatus.UNHEALTHY
            else:
                # Type narrowing - result is dict[str, Any] here, not Exception
                result_dict = result  # type: dict[str, Any]
                checks.append(result_dict)
                if result_dict["status"] == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif (
                    result_dict["status"] == HealthStatus.DEGRADED
                    and overall_status == HealthStatus.HEALTHY
                ):
                    overall_status = HealthStatus.DEGRADED

        end_time = datetime.now(tz=UTC)
        duration_ms = (end_time - start_time).total_seconds() * 1000

        return {
            "status": overall_status,
            "timestamp": end_time.isoformat(),
            "duration_ms": round(duration_ms, 2),
            "checks": checks,
            "version": get_package_version(),
            "environment": {
                "debug": getattr(settings, "debug", False),
            },
        }


# Global health check manager
_health_manager: HealthCheckManager | None = None


def get_health_manager() -> HealthCheckManager:
    """Get health check manager singleton."""
    global _health_manager  # noqa: PLW0603
    if _health_manager is None:
        _health_manager = HealthCheckManager()
    return _health_manager
