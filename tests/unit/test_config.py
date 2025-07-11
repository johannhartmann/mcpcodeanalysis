"""Tests for configuration management."""

import tempfile
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
        assert "node_modules" in config.exclude_patterns

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = ScannerConfig(
            sync_interval=600,
            storage_path=Path("/custom/path"),
            exclude_patterns=["*.tmp", "build/"],
        )
        assert config.sync_interval == 600
        assert config.storage_path == Path("/custom/path")
        assert config.exclude_patterns == ["*.tmp", "build/"]

    def test_sync_interval_validation(self) -> None:
        """Test sync interval validation."""
        with pytest.raises(ValidationError):
            ScannerConfig(sync_interval=30)  # Less than minimum


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

        # Sum should be 1.0
        total = (
            config.ranking_weights.semantic_similarity
            + config.ranking_weights.keyword_match
            + config.ranking_weights.recency
            + config.ranking_weights.popularity
        )
        assert abs(total - 1.0) < 1e-3

    def test_similarity_threshold_validation(self) -> None:
        """Test similarity threshold validation."""
        with pytest.raises(ValidationError):
            QueryConfig(similarity_threshold=1.5)  # Out of range

    # Tests for the old Settings class have been removed
    # since we're now using Dynaconf for configuration management
    """Test Settings model."""

    def test_from_yaml(self, test_config_file, monkeypatch) -> None:
        """Test loading settings from YAML."""
        # This test is disabled as we're now using Dynaconf
        pytest.skip("Settings class replaced with Dynaconf")

    def test_env_var_expansion(self, monkeypatch) -> None:
        """Test environment variable expansion in YAML."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
repositories:
  - url: https://github.com/test/repo
    access_token: ${GITHUB_TOKEN}
""",
            )
            config_path = Path(f.name)

        try:
            # This test is disabled as we're now using Dynaconf
            pytest.skip("Settings class replaced with Dynaconf")
        finally:
            config_path.unlink()

    def test_get_database_url(self) -> None:
        """Test database URL generation."""
        # This test is now handled by Dynaconf
        pytest.skip("Database URL generation handled by Dynaconf")

    def test_get_database_url_from_config(self, monkeypatch, tmp_path) -> None:
        """Test database URL from configuration."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        # Clear POSTGRES_PASSWORD from environment to test database config password
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

        # Create a temporary directory and change to it to avoid loading .env
        monkeypatch.chdir(tmp_path)

        # This test is disabled as we're now using Dynaconf
        pytest.skip("Settings class replaced with Dynaconf")

    def test_validate_config(self) -> None:
        """Test configuration validation."""
        # This test is now handled by Dynaconf validators
        pytest.skip("Configuration validation handled by Dynaconf")

    def test_validate_ranking_weights(self) -> None:
        """Test ranking weights validation."""
        # Invalid weights that don't sum to 1.0
        from src.models import RankingWeights

        with pytest.raises(ValueError, match="Ranking weights must sum to 1.0"):
            RankingWeights(
                semantic_similarity=0.5,
                keyword_match=0.3,
                recency=0.1,
                popularity=0.2,  # Sum = 1.1
            )


class TestSettingsSingleton:
    """Test settings singleton functionality."""

    def test_get_settings(self, test_config_file, monkeypatch) -> None:
        """Test get_settings singleton."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("CONFIG_PATH", str(test_config_file))

        # This test is disabled as we're now using Dynaconf
        pytest.skip("get_settings replaced with Dynaconf")

        # Skipping as get_settings is removed

    def test_reload_settings(self, test_config_file, monkeypatch) -> None:
        """Test reload_settings."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("CONFIG_PATH", str(test_config_file))

        # This test is disabled as we're now using Dynaconf
        pytest.skip("get_settings and reload_settings replaced with Dynaconf")
