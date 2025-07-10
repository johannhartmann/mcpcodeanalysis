"""Configuration management for MCP Code Analysis Server."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RepositoryConfig(BaseModel):
    """Configuration for a GitHub repository."""

    url: str
    branch: str | None = None
    access_token: SecretStr | None = None
    enable_domain_analysis: bool = Field(
        default=False,
        description="Enable domain-driven analysis during indexing",
    )

    @field_validator("url")
    @classmethod
    def validate_github_url(cls, v: str) -> str:
        """Validate GitHub URL format."""
        if not v.startswith(("https://github.com/", "git@github.com:")):
            msg = f"Invalid GitHub URL: {v}"
            raise ValueError(msg)
        return v


class ScannerConfig(BaseModel):
    """Scanner configuration."""

    sync_interval: int = Field(
        default=300,
        ge=60,
        description="Sync interval in seconds",
    )
    webhook_secret: SecretStr | None = None
    storage_path: Path = Path("./repositories")
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "__pycache__",
            "*.pyc",
            ".git",
            "venv",
            ".env",
            "node_modules",
        ],
    )


class ParserConfig(BaseModel):
    """Parser configuration."""

    languages: list[str] = Field(default_factory=lambda: ["python"])
    chunk_size: int = Field(default=100, ge=10, le=1000)


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration."""

    model: str = "text-embedding-ada-002"
    batch_size: int = Field(default=100, ge=1, le=500)
    use_cache: bool = True
    cache_dir: Path = Path(".embeddings_cache")
    max_tokens: int = Field(default=8000, ge=100, le=8191)
    generate_interpreted: bool = True


class DomainAnalysisConfig(BaseModel):
    """Domain-driven analysis configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable domain analysis by default",
    )
    chunk_size: int = Field(default=1000, ge=500, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=500)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    max_entities_per_file: int = Field(default=50, ge=1, le=200)
    leiden_resolution: float = Field(default=1.0, ge=0.1, le=10.0)
    min_context_size: int = Field(default=3, ge=1, le=20)


class DatabaseConfig(BaseModel):
    """Database configuration."""

    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = "code_analysis"
    user: str = "codeanalyzer"
    password: SecretStr | None = None
    pool_size: int = Field(default=10, ge=1, le=50)
    max_overflow: int = Field(default=20, ge=0, le=100)
    vector_dimension: int = 1536
    index_lists: int = 100

    @property
    def url(self) -> str:
        """Get database connection URL."""
        password_str = self.password.get_secret_value() if self.password else ""
        if password_str:
            return f"postgresql+asyncpg://{self.user}:{password_str}@{self.host}:{self.port}/{self.database}"
        return (
            f"postgresql+asyncpg://{self.user}@{self.host}:{self.port}/{self.database}"
        )


class MCPConfig(BaseModel):
    """MCP server configuration."""

    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    rate_limit_enabled: bool = False
    rate_limit_per_minute: int = Field(default=60, ge=1)
    request_timeout: int = Field(default=30, ge=1, le=300)


class GitHubConfig(BaseModel):
    """GitHub integration configuration."""

    api_rate_limit: int = Field(default=5000, ge=1)
    webhook_endpoint: str = "/webhooks/github"
    use_webhooks: bool = True
    poll_interval: int = Field(default=300, ge=60)


class QueryConfig(BaseModel):
    """Query processing configuration."""

    default_limit: int = Field(default=10, ge=1, le=100)
    max_limit: int = Field(default=100, ge=1, le=1000)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    include_context: bool = True
    context_lines: int = Field(default=3, ge=0, le=10)
    ranking_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "semantic_similarity": 0.6,
            "keyword_match": 0.2,
            "recency": 0.1,
            "popularity": 0.1,
        },
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"
    file_enabled: bool = True
    file_path: Path = Path("logs/mcp-server.log")
    file_rotation: str = "daily"
    file_retention_days: int = Field(default=7, ge=1)
    console_enabled: bool = True
    console_colorized: bool = True


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment variables
    openai_api_key: SecretStr | None = None
    database_url: str | None = None
    postgres_password: SecretStr | None = None
    api_key: SecretStr | None = None
    debug: bool = False

    # Configuration sections
    repositories: list[RepositoryConfig] = Field(default_factory=list)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    domain_analysis: DomainAnalysisConfig = Field(default_factory=DomainAnalysisConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    github: GitHubConfig = Field(default_factory=GitHubConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Settings":
        """Load settings from YAML configuration file."""
        if not config_path.exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        with config_path.open() as f:
            config_data = yaml.safe_load(f)

        # Expand environment variables in configuration
        config_data = cls._expand_env_vars(config_data)

        # Load environment variables
        env_settings = cls()

        # Merge configuration by recreating with proper types
        if config_data:
            # Update the dictionary used for initialization
            init_data = {}

            # Get values from environment first
            for field_name in cls.model_fields:
                if hasattr(env_settings, field_name):
                    init_data[field_name] = getattr(env_settings, field_name)

            # Override with config file values
            init_data.update(config_data)

            # Create new instance with merged data
            return cls(**init_data)

        return env_settings

    @staticmethod
    def _expand_env_vars(config: Any) -> Any:
        """Recursively expand environment variables in configuration."""
        if isinstance(config, dict):
            return {k: Settings._expand_env_vars(v) for k, v in config.items()}
        if isinstance(config, list):
            return [Settings._expand_env_vars(item) for item in config]
        if isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        return config

    def get_database_url(self) -> str:
        """Get database URL."""
        if self.database_url:
            return self.database_url

        password = self.postgres_password or self.database.password
        if password:
            password_str = (
                password.get_secret_value()
                if hasattr(password, "get_secret_value")
                else str(password)
            )
        else:
            password_str = ""  # nosec B105 - Empty password is valid for local dev

        return (
            f"postgresql://{self.database.user}:{password_str}"
            f"@{self.database.host}:{self.database.port}/{self.database.database}"
        )

    def validate_config(self) -> None:
        """Validate configuration settings."""
        # Ensure storage path exists
        self.scanner.storage_path.mkdir(parents=True, exist_ok=True)

        # Ensure cache directory exists
        if self.embeddings.use_cache:
            self.embeddings.cache_dir.mkdir(parents=True, exist_ok=True)

        # Ensure log directory exists
        if self.logging.file_enabled:
            self.logging.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate ranking weights sum to 1.0
        WEIGHT_SUM_TOLERANCE = 0.001
        weight_sum = sum(self.query.ranking_weights.values())
        if abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
            msg = f"Ranking weights must sum to 1.0, got {weight_sum}"
            raise ValueError(msg)


class SettingsManager:
    """Manages the singleton settings instance."""

    def __init__(self):
        self._settings: Settings | None = None

    def get(self) -> Settings:
        """Get application settings singleton."""
        if self._settings is None:
            config_path = Path(os.getenv("CONFIG_PATH", "config.yaml"))
            if config_path.exists():
                self._settings = Settings.from_yaml(config_path)
            else:
                self._settings = Settings()
            self._settings.validate_config()
        return self._settings

    def reload(self, config_path: Path | None = None) -> Settings:
        """Reload settings from configuration file."""
        if config_path is None:
            config_path = Path(os.getenv("CONFIG_PATH", "config.yaml"))
        self._settings = Settings.from_yaml(config_path) if config_path.exists() else Settings()
        self._settings.validate_config()
        return self._settings


# Global settings manager instance
_settings_manager = SettingsManager()


def get_settings() -> Settings:
    """Get application settings singleton."""
    return _settings_manager.get()


def reload_settings(config_path: Path | None = None) -> Settings:
    """Reload settings from configuration file."""
    return _settings_manager.reload(config_path)
