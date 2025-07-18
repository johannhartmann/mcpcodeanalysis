"""Tests for configuration models."""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from src.models import (
    DatabaseConfig,
    QueryConfig,
    RepositoryConfig,
    ScannerConfig,
)


class TestRepositoryConfig:
    """Test RepositoryConfig model."""

    def test_valid_github_url(self) -> None:
        """Test valid GitHub URL."""
        config = RepositoryConfig(
            url="https://github.com/owner/repo",
            branch="main",
        )
        assert config.url == "https://github.com/owner/repo"
        assert config.branch == "main"
        assert config.access_token is None

    def test_github_ssh_url(self) -> None:
        """Test GitHub SSH URL."""
        config = RepositoryConfig(
            url="git@github.com:owner/repo.git",
        )
        assert config.url == "git@github.com:owner/repo.git"

    def test_invalid_url(self) -> None:
        """Test invalid repository URL."""
        with pytest.raises(ValidationError) as exc_info:
            RepositoryConfig(url="https://gitlab.com/owner/repo")

        assert "Invalid GitHub URL" in str(exc_info.value)

    def test_with_access_token(self) -> None:
        """Test repository with access token."""
        config = RepositoryConfig(
            url="https://github.com/owner/private-repo",
            access_token=SecretStr("ghp_test_token"),
        )
        assert config.access_token.get_secret_value() == "ghp_test_token"


class TestScannerConfig:
    """Test ScannerConfig model."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = ScannerConfig()
        assert config.sync_interval == 300
        assert config.storage_path == Path("./repositories")
        assert "__pycache__" in config.exclude_patterns

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = ScannerConfig(
            sync_interval=60,
            storage_path=Path("/storage"),
            exclude_patterns=["build", "dist"],
        )
        assert config.sync_interval == 60
        assert config.storage_path == Path("/storage")
        assert config.exclude_patterns == ["build", "dist"]

    def test_sync_interval_validation(self) -> None:
        """Test sync interval validation."""
        with pytest.raises(ValidationError):
            ScannerConfig(sync_interval=5)  # Too short


class TestDatabaseConfig:
    """Test DatabaseConfig model."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = DatabaseConfig(password=SecretStr("test_password"))
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "code_analysis"
        assert config.user == "codeanalyzer"
        assert config.pool_size == 10
        assert config.vector_dimension == 1536

    def test_port_validation(self) -> None:
        """Test port number validation."""
        with pytest.raises(ValidationError):
            DatabaseConfig(password=SecretStr("test"), port=70000)  # Invalid port


class TestQueryConfig:
    """Test QueryConfig model."""

    def test_ranking_weights_default(self) -> None:
        """Test default ranking weights."""
        config = QueryConfig()
        assert config.ranking_weights.semantic_similarity == 0.6
        assert config.ranking_weights.keyword_match == 0.2
        assert config.ranking_weights.recency == 0.1
        assert config.ranking_weights.popularity == 0.1

    def test_ranking_weights_validation(self) -> None:
        """Test ranking weights validation."""
        from src.models import RankingWeights

        # Valid weights that sum to 1.0
        weights = RankingWeights(
            semantic_similarity=0.5,
            keyword_match=0.3,
            recency=0.1,
            popularity=0.1,
        )
        config = QueryConfig(ranking_weights=weights)

        total = (
            config.ranking_weights.semantic_similarity
            + config.ranking_weights.keyword_match
            + config.ranking_weights.recency
            + config.ranking_weights.popularity
        )
        assert abs(total - 1.0) < 0.001

    def test_similarity_threshold_validation(self) -> None:
        """Test similarity threshold validation."""
        with pytest.raises(ValidationError):
            QueryConfig(similarity_threshold=1.5)  # Out of range


# Tests for Dynaconf settings can be added here if needed
class TestDynaconfSettings:
    """Test Dynaconf settings integration."""

    def test_settings_load(self, test_settings):
        """Test that settings load correctly."""
        assert test_settings is not None
        assert hasattr(test_settings, "database")
        assert hasattr(test_settings, "scanner")

    def test_database_url_generation(self, test_settings):
        """Test database URL generation."""
        from src.config import get_database_url

        url = get_database_url()
        assert url.startswith("postgresql://")
        assert "test_code_analysis" in url
