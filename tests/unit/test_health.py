"""Tests for health check utilities."""

from typing import Any, Never, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils.health import (
    DatabaseHealthCheck,
    DiskSpaceHealthCheck,
    GitHubHealthCheck,
    HealthCheck,
    HealthCheckManager,
    HealthStatus,
    OpenAIHealthCheck,
    get_health_manager,
)


class TestHealthStatus:
    """Test HealthStatus constants."""

    def test_status_values(self) -> None:
        """Test health status values."""
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNHEALTHY == "unhealthy"


class TestHealthCheck:
    """Test base HealthCheck class."""

    class TestCheck(HealthCheck):
        """Test implementation of HealthCheck."""

        def __init__(self, name: str) -> None:
            super().__init__(name)

        async def _perform_check(self) -> dict[str, Any]:
            return {"test": "data"}

    @pytest.mark.asyncio
    async def test_successful_check(self) -> None:
        """Test successful health check."""
        check = self.TestCheck("test_check")
        result = await check.check()

        assert result["name"] == "test_check"
        assert result["status"] == HealthStatus.HEALTHY
        assert result["details"]["test"] == "data"
        assert "duration_ms" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_failed_check(self) -> None:
        """Test failed health check."""

        class FailingCheck(HealthCheck):
            async def _perform_check(self) -> Never:
                raise ValueError("Test error")

        check = FailingCheck("failing_check")
        result = await check.check()

        assert result["name"] == "failing_check"
        assert result["status"] == HealthStatus.UNHEALTHY
        assert result["details"]["error"] == "Test error"

    @pytest.mark.asyncio
    async def test_boolean_result(self) -> None:
        """Test health check with boolean result."""

        class BooleanCheck(HealthCheck):
            async def _perform_check(self) -> bool:
                return False

        check = BooleanCheck("bool_check")
        result = await check.check()

        assert result["status"] == HealthStatus.UNHEALTHY
        assert result["details"] == {}


class TestDatabaseHealthCheck:
    """Test DatabaseHealthCheck."""

    @pytest.mark.asyncio
    async def test_database_healthy(self) -> None:
        """Test healthy database check."""
        with patch("src.utils.health.create_async_engine") as mock_engine:
            # Mock the engine and connection
            mock_conn = AsyncMock()
            mock_engine.return_value.begin.return_value.__aenter__.return_value = (
                mock_conn
            )
            mock_engine.return_value.dispose = AsyncMock()

            # Mock query results
            mock_conn.execute.side_effect = [
                AsyncMock(scalar=MagicMock(return_value=1)),  # SELECT 1
                AsyncMock(scalar=MagicMock(return_value="0.2.5")),  # pgvector version
                AsyncMock(scalar=MagicMock(return_value=1048576)),  # database size
                AsyncMock(
                    first=MagicMock(
                        return_value=MagicMock(
                            repositories=5,
                            files=100,
                            embeddings=500,
                        ),
                    ),
                ),  # table counts
            ]

            check = DatabaseHealthCheck()
            result = await check._perform_check()

            assert result["connected"] is True
            assert result["pgvector_version"] == "0.2.5"
            assert result["database_size_mb"] == 1.0
            assert result["table_counts"]["repositories"] == 5
            assert result["table_counts"]["files"] == 100
            assert result["table_counts"]["embeddings"] == 500

    @pytest.mark.asyncio
    async def test_database_connection_error(self) -> None:
        """Test database connection error."""
        with patch("src.utils.health.create_async_engine") as mock_engine:
            mock_engine.return_value.begin.side_effect = Exception("Connection failed")
            mock_engine.return_value.dispose = AsyncMock()

            check = DatabaseHealthCheck()

            with pytest.raises(Exception, match="Connection failed"):
                await check._perform_check()


class TestGitHubHealthCheck:
    """Test GitHubHealthCheck."""

    @pytest.mark.asyncio
    async def test_github_healthy(self) -> None:
        """Test healthy GitHub API check."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "rate": {
                    "limit": 5000,
                    "remaining": 4999,
                    "reset": 1234567890,
                },
            }

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            check = GitHubHealthCheck()
            result = await check._perform_check()

            assert result["connected"] is True
            assert result["rate_limit"]["limit"] == 5000
            assert result["rate_limit"]["remaining"] == 4999
            assert "reset" in result["rate_limit"]

    @pytest.mark.asyncio
    async def test_github_api_error(self) -> None:
        """Test GitHub API error."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 503

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            check = GitHubHealthCheck()
            result = await check._perform_check()

            assert result["connected"] is False
            assert result["status_code"] == 503


class TestOpenAIHealthCheck:
    """Test OpenAIHealthCheck."""

    @pytest.mark.asyncio
    async def test_openai_healthy(self) -> None:
        """Test healthy OpenAI API check."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"id": "text-embedding-ada-002"},
                    {"id": "text-embedding-3-small"},
                    {"id": "gpt-4"},
                ],
            }

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            check = OpenAIHealthCheck()
            result = await check._perform_check()

            assert result["connected"] is True
            assert "text-embedding-ada-002" in result["embedding_models_available"]
            assert result["model_available"] is True

    @pytest.mark.asyncio
    async def test_openai_invalid_key(self) -> None:
        """Test OpenAI API with invalid key."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 401

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            check = OpenAIHealthCheck()
            result = await check._perform_check()

            assert result["connected"] is False
            assert result["error"] == "Invalid API key"


class TestDiskSpaceHealthCheck:
    """Test DiskSpaceHealthCheck."""

    @pytest.mark.asyncio
    async def test_disk_space_healthy(self) -> None:
        """Test healthy disk space check."""
        with patch("shutil.disk_usage") as mock_disk_usage:
            # Mock disk usage (total=100GB, used=50GB, free=50GB)
            mock_disk_usage.return_value = MagicMock(
                total=100 * 1024**3,
                used=50 * 1024**3,
                free=50 * 1024**3,
            )

            with patch("pathlib.Path.exists", return_value=True):
                check = DiskSpaceHealthCheck()
                result = await check._perform_check()

                assert "paths" in result
                assert result["critical_space"] is False

                # Check one of the paths
                for path_info in result["paths"].values():
                    assert path_info["total_gb"] == 100.0
                    assert path_info["used_gb"] == 50.0
                    assert path_info["free_gb"] == 50.0
                    assert path_info["used_percent"] == 50.0

    @pytest.mark.asyncio
    async def test_disk_space_critical(self) -> None:
        """Test critical disk space."""
        with patch("shutil.disk_usage") as mock_disk_usage:
            # Mock disk usage (total=100GB, used=95GB, free=5GB)
            mock_disk_usage.return_value = MagicMock(
                total=100 * 1024**3,
                used=95 * 1024**3,
                free=5 * 1024**3,
            )

            with patch("pathlib.Path.exists", return_value=True):
                check = DiskSpaceHealthCheck()
                result = await check._perform_check()

                assert result["critical_space"] is True


class TestHealthCheckManager:
    """Test HealthCheckManager."""

    @pytest.mark.asyncio
    async def test_check_all_healthy(self) -> None:
        """Test all checks healthy."""
        manager = HealthCheckManager()

        # Mock all checks to return healthy
        for check in manager.checks:
            cast("Any", check).check = AsyncMock(
                return_value={
                    "name": check.name,
                    "status": HealthStatus.HEALTHY,
                    "details": {},
                },
            )

        result = await manager.check_all()

        assert result["status"] == HealthStatus.HEALTHY
        assert len(result["checks"]) == len(manager.checks)
        assert "timestamp" in result
        assert "duration_ms" in result
        assert result["version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_check_some_unhealthy(self) -> None:
        """Test some checks unhealthy."""
        manager = HealthCheckManager()

        # Mock checks with mixed results
        for i, check in enumerate(manager.checks):
            if i == 0:
                status = HealthStatus.UNHEALTHY
            elif i == 1:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            cast("Any", check).check = AsyncMock(
                return_value={
                    "name": check.name,
                    "status": status,
                    "details": {},
                },
            )

        result = await manager.check_all()

        assert result["status"] == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_check_exception_handling(self) -> None:
        """Test exception handling in checks."""
        manager = HealthCheckManager()

        # Make first check raise exception
        cast("Any", manager.checks[0]).check = AsyncMock(
            side_effect=Exception("Check failed")
        )

        # Others are healthy
        for check in manager.checks[1:]:
            cast("Any", check).check = AsyncMock(
                return_value={
                    "name": check.name,
                    "status": HealthStatus.HEALTHY,
                    "details": {},
                },
            )

        result = await manager.check_all()

        assert result["status"] == HealthStatus.UNHEALTHY
        # Should have error check
        error_checks = [c for c in result["checks"] if "error" in c]
        assert len(error_checks) == 1


class TestGetHealthManager:
    """Test get_health_manager function."""

    def test_singleton(self) -> None:
        """Test health manager singleton."""
        manager1 = get_health_manager()
        manager2 = get_health_manager()

        assert manager1 is manager2
        assert isinstance(manager1, HealthCheckManager)
