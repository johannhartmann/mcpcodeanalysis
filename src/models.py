"""Configuration models for MCP Code Analysis Server."""

import re
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


class RepositoryConfig(BaseModel):
    """Configuration for a GitHub repository."""

    url: str = Field(..., description="GitHub repository URL")
    branch: str | None = Field(
        None, description="Branch to track (defaults to default branch)"
    )
    access_token: SecretStr | None = Field(
        None, description="GitHub access token for private repos"
    )
    enable_domain_analysis: bool = Field(
        False, description="Enable domain analysis for this repository"
    )

    @field_validator("url")
    @classmethod
    def validate_github_url(cls, v: str) -> str:
        """Validate GitHub URL format."""
        # Accept both HTTPS and SSH formats, and file:// for local repos
        patterns = [
            r"^https://github\.com/[\w-]+/[\w.-]+/?$",
            r"^git@github\.com:[\w-]+/[\w.-]+\.git$",
            r"^git@github\.com:[\w-]+/[\w.-]+$",
            r"^file:///.+$",  # Accept file:// URLs for local repos
        ]

        if not any(re.match(pattern, v) for pattern in patterns):
            msg = f"Invalid repository URL: {v}"
            raise ValueError(msg)

        return v.rstrip("/")  # Remove trailing slash if present


class ScannerConfig(BaseModel):
    """Scanner configuration."""

    sync_interval: int = Field(
        300, ge=60, description="Repository sync interval in seconds"
    )
    storage_path: Path = Field(
        Path("./repositories"), description="Local storage path for repositories"
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "__pycache__",
            "*.pyc",
            ".git",
            ".venv",
            "venv",
            "env",
            "node_modules",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "*.egg-info",
            "dist",
            "build",
            ".coverage",
            "htmlcov",
        ],
        description="Patterns to exclude from scanning",
    )
    max_file_size_mb: int = Field(
        10, ge=1, le=100, description="Maximum file size to process in MB"
    )
    use_git: bool = Field(True, description="Use Git for change tracking")
    git_branch: str = Field("main", description="Default Git branch")


class ParserConfig(BaseModel):
    """Parser configuration."""

    languages: list[str] = Field(["python"], description="Languages to parse")
    chunk_size: int = Field(
        100, ge=10, le=1000, description="Chunk size for large files"
    )
    max_depth: int = Field(
        10, ge=1, le=50, description="Maximum depth for nested structures"
    )
    extract_docstrings: bool = Field(True, description="Extract docstrings")
    extract_type_hints: bool = Field(True, description="Extract type hints")


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration."""

    model: str = Field("text-embedding-ada-002", description="OpenAI embedding model")
    batch_size: int = Field(
        100, ge=1, le=1000, description="Batch size for embedding generation"
    )
    max_tokens: int = Field(
        8000, ge=100, le=10000, description="Maximum tokens per chunk"
    )
    use_cache: bool = Field(True, description="Cache embeddings locally")
    cache_dir: Path = Field(Path(".embeddings_cache"), description="Cache directory")
    generate_interpreted: bool = Field(
        True, description="Generate interpreted embeddings"
    )


class DatabaseConfig(BaseModel):
    """Database configuration."""

    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, ge=1, le=65535, description="Database port")
    database: str = Field("code_analysis", description="Database name")
    user: str = Field("codeanalyzer", description="Database user")
    password: SecretStr = Field(..., description="Database password")
    pool_size: int = Field(10, ge=1, description="Connection pool size")
    max_overflow: int = Field(20, ge=0, description="Maximum overflow connections")
    vector_dimension: int = Field(1536, description="Vector dimension for pgvector")
    index_lists: int = Field(100, description="IVFFlat index parameter")


class MCPConfig(BaseModel):
    """MCP server configuration."""

    host: str = Field("127.0.0.1", description="Server host")
    port: int = Field(8080, ge=1, le=65535, description="Server port")
    allow_origins: list[str] = Field(["*"], description="CORS allowed origins")
    rate_limit_enabled: bool = Field(False, description="Enable rate limiting")
    rate_limit_per_minute: int = Field(60, ge=1, description="Rate limit per minute")
    request_timeout: int = Field(
        30, ge=1, le=300, description="Request timeout in seconds"
    )


class RankingWeights(BaseModel):
    """Query ranking weights."""

    semantic_similarity: float = Field(0.6, ge=0.0, le=1.0)
    keyword_match: float = Field(0.2, ge=0.0, le=1.0)
    recency: float = Field(0.1, ge=0.0, le=1.0)
    popularity: float = Field(0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sum(self) -> "RankingWeights":
        """Validate that weights sum to 1.0."""
        total = (
            self.semantic_similarity
            + self.keyword_match
            + self.recency
            + self.popularity
        )
        if abs(total - 1.0) > 1e-3:
            msg = f"Ranking weights must sum to 1.0, got {total}"
            raise ValueError(msg)
        return self


class QueryConfig(BaseModel):
    """Query configuration."""

    default_limit: int = Field(
        10, ge=1, le=100, description="Default search result limit"
    )
    max_limit: int = Field(
        100, ge=1, le=1000, description="Maximum search result limit"
    )
    similarity_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Similarity threshold"
    )
    include_context: bool = Field(True, description="Include file context in results")
    context_lines: int = Field(3, ge=0, description="Number of context lines")
    ranking_weights: RankingWeights = Field(
        default_factory=lambda: RankingWeights(
            semantic_similarity=0.6, keyword_match=0.2, recency=0.1, popularity=0.1
        )
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    format: str = Field("json", pattern="^(json|text)$")
    file_enabled: bool = Field(True, description="Enable file logging")
    file_path: Path = Field(Path("logs/mcp-server.log"), description="Log file path")
    file_rotation: str = Field("daily", description="Log rotation schedule")
    file_retention_days: int = Field(7, ge=1, description="Log retention in days")
    console_enabled: bool = Field(True, description="Enable console logging")
    console_colorized: bool = Field(True, description="Colorize console output")


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    metrics_enabled: bool = Field(False, description="Enable metrics collection")
    metrics_port: int = Field(9090, ge=1, le=65535, description="Metrics port")
    health_check_enabled: bool = Field(True, description="Enable health check endpoint")
    health_check_path: str = Field("/health", description="Health check path")
    profiling_enabled: bool = Field(False, description="Enable performance profiling")
    profiling_path: Path = Field(Path("profiles/"), description="Profiling data path")
