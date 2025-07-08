"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from src.mcp_server.config import (
    DatabaseConfig,
    GitHubConfig,
    LoggingConfig,
    MCPConfig,
    QueryConfig,
    RepositoryConfig,
    ScannerConfig,
    Settings,
    get_settings,
    reload_settings,
)


class TestRepositoryConfig:
    """Test RepositoryConfig model."""
    
    def test_valid_github_url(self):
        """Test valid GitHub URL."""
        config = RepositoryConfig(
            url="https://github.com/owner/repo",
            branch="main",
        )
        assert config.url == "https://github.com/owner/repo"
        assert config.branch == "main"
        assert config.access_token is None
    
    def test_github_ssh_url(self):
        """Test GitHub SSH URL."""
        config = RepositoryConfig(
            url="git@github.com:owner/repo.git",
        )
        assert config.url == "git@github.com:owner/repo.git"
    
    def test_invalid_url(self):
        """Test invalid repository URL."""
        with pytest.raises(ValidationError) as exc_info:
            RepositoryConfig(url="https://gitlab.com/owner/repo")
        
        assert "Invalid GitHub URL" in str(exc_info.value)
    
    def test_with_access_token(self):
        """Test repository with access token."""
        config = RepositoryConfig(
            url="https://github.com/owner/private-repo",
            access_token=SecretStr("ghp_test_token"),
        )
        assert config.access_token.get_secret_value() == "ghp_test_token"


class TestScannerConfig:
    """Test ScannerConfig model."""
    
    def test_defaults(self):
        """Test default values."""
        config = ScannerConfig()
        assert config.sync_interval == 300
        assert config.storage_path == Path("./repositories")
        assert "__pycache__" in config.exclude_patterns
        assert "node_modules" in config.exclude_patterns
    
    def test_custom_values(self):
        """Test custom values."""
        config = ScannerConfig(
            sync_interval=600,
            storage_path=Path("/custom/path"),
            exclude_patterns=["*.tmp", "build/"],
        )
        assert config.sync_interval == 600
        assert config.storage_path == Path("/custom/path")
        assert config.exclude_patterns == ["*.tmp", "build/"]
    
    def test_sync_interval_validation(self):
        """Test sync interval validation."""
        with pytest.raises(ValidationError):
            ScannerConfig(sync_interval=30)  # Less than minimum


class TestDatabaseConfig:
    """Test DatabaseConfig model."""
    
    def test_defaults(self):
        """Test default values."""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "code_analysis"
        assert config.user == "codeanalyzer"
        assert config.pool_size == 10
        assert config.vector_dimension == 1536
    
    def test_port_validation(self):
        """Test port number validation."""
        with pytest.raises(ValidationError):
            DatabaseConfig(port=70000)  # Invalid port


class TestQueryConfig:
    """Test QueryConfig model."""
    
    def test_ranking_weights_default(self):
        """Test default ranking weights."""
        config = QueryConfig()
        assert config.ranking_weights["semantic_similarity"] == 0.6
        assert config.ranking_weights["keyword_match"] == 0.2
        assert config.ranking_weights["recency"] == 0.1
        assert config.ranking_weights["popularity"] == 0.1
        
        # Sum should be 1.0
        assert sum(config.ranking_weights.values()) == 1.0
    
    def test_similarity_threshold_validation(self):
        """Test similarity threshold validation."""
        with pytest.raises(ValidationError):
            QueryConfig(similarity_threshold=1.5)  # Out of range


class TestSettings:
    """Test Settings model."""
    
    def test_from_yaml(self, test_config_file, monkeypatch):
        """Test loading settings from YAML."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        
        settings = Settings.from_yaml(test_config_file)
        
        assert len(settings.repositories) == 2
        assert settings.repositories[0].url == "https://github.com/test/repo1"
        assert settings.scanner.sync_interval == 60
        assert settings.embeddings.model == "text-embedding-ada-002"
        assert settings.openai_api_key.get_secret_value() == "sk-test"
    
    def test_env_var_expansion(self, monkeypatch):
        """Test environment variable expansion in YAML."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
repositories:
  - url: https://github.com/test/repo
    access_token: ${GITHUB_TOKEN}
""")
            config_path = Path(f.name)
        
        try:
            settings = Settings.from_yaml(config_path)
            assert settings.repositories[0].access_token == "ghp_test"
        finally:
            config_path.unlink()
    
    def test_get_database_url(self, test_settings):
        """Test database URL generation."""
        # With DATABASE_URL env var
        url = test_settings.get_database_url()
        assert url.startswith("postgresql://")
        assert "test_code_analysis" in url
    
    def test_get_database_url_from_config(self, monkeypatch):
        """Test database URL from configuration."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        
        settings = Settings(
            openai_api_key=SecretStr("sk-test"),
            database=DatabaseConfig(
                host="db.example.com",
                port=5433,
                database="mydb",
                user="myuser",
                password=SecretStr("mypass"),
            )
        )
        
        url = settings.get_database_url()
        assert url == "postgresql://myuser:mypass@db.example.com:5433/mydb"
    
    def test_validate_config(self, test_settings, tmp_path, monkeypatch):
        """Test configuration validation."""
        # Set paths to temp directory
        test_settings.scanner.storage_path = tmp_path / "repos"
        test_settings.embeddings.cache_dir = tmp_path / "cache"
        test_settings.logging.file_path = tmp_path / "logs" / "test.log"
        
        test_settings.validate_config()
        
        # Check directories were created
        assert test_settings.scanner.storage_path.exists()
        assert test_settings.embeddings.cache_dir.exists()
        assert test_settings.logging.file_path.parent.exists()
    
    def test_validate_ranking_weights(self, test_settings):
        """Test ranking weights validation."""
        # Invalid weights that don't sum to 1.0
        test_settings.query.ranking_weights = {
            "semantic_similarity": 0.5,
            "keyword_match": 0.3,
            "recency": 0.1,
            "popularity": 0.2,  # Sum = 1.1
        }
        
        with pytest.raises(ValueError) as exc_info:
            test_settings.validate_config()
        
        assert "Ranking weights must sum to 1.0" in str(exc_info.value)


class TestSettingsSingleton:
    """Test settings singleton functionality."""
    
    def test_get_settings(self, test_config_file, monkeypatch):
        """Test get_settings singleton."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("CONFIG_PATH", str(test_config_file))
        
        # Clear singleton
        from src.mcp_server import config
        config._settings = None
        
        # First call creates instance
        settings1 = get_settings()
        assert settings1 is not None
        
        # Second call returns same instance
        settings2 = get_settings()
        assert settings2 is settings1
    
    def test_reload_settings(self, test_config_file, monkeypatch):
        """Test reload_settings."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("CONFIG_PATH", str(test_config_file))
        
        # Clear singleton
        from src.mcp_server import config
        config._settings = None
        
        # Get initial settings
        settings1 = get_settings()
        
        # Reload settings
        settings2 = reload_settings(test_config_file)
        
        # Should be different instances
        assert settings2 is not settings1
        
        # get_settings should now return the new instance
        settings3 = get_settings()
        assert settings3 is settings2