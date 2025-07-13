"""Configuration management using Dynaconf."""

from pathlib import Path

from dynaconf import Dynaconf, Validator

# Initialize Dynaconf with validators
settings = Dynaconf(
    envvar_prefix="MCP",
    settings_files=["settings.toml", ".secrets.toml"],
    environments=True,
    load_dotenv=True,
    dotenv_path=".env",
    merge_enabled=True,
    validators=[
        # Database validators
        Validator("database.host", must_exist=True),
        Validator("database.port", must_exist=True, gte=1, lte=65535),
        Validator("database.database", must_exist=True),
        Validator("database.user", must_exist=True),
        Validator("database.password", must_exist=True),
        # OpenAI validators (only required in production)
        Validator(
            "OPENAI_API_KEY",
            must_exist=True,
            when=Validator("ENV_FOR_DYNACONF", eq="production"),
        ),
        # Scanner validators
        Validator("scanner.max_file_size_mb", gte=1, lte=100),
        Validator("scanner.root_paths", must_exist=True, is_type_of=list),
        # Parser validators
        Validator("parser.chunk_size", gte=10, lte=1000),
        Validator("parser.max_depth", gte=1, lte=50),
        # Embeddings validators
        Validator("embeddings.batch_size", gte=1, lte=1000),
        Validator("embeddings.max_tokens", gte=100, lte=10000),
        # Query validators
        Validator("query.similarity_threshold", gte=0.0, lte=1.0),
        Validator("query.default_limit", gte=1, lte=100),
        Validator("query.max_limit", gte=1, lte=1000),
        # Validate ranking weights sum to 1.0
        Validator(
            "query.ranking_weights",
            condition=lambda v: abs(sum(v.values()) - 1.0) < 1e-3,
            messages={"condition": "Ranking weights must sum to 1.0"},
        ),
        # MCP server validators
        Validator("mcp.port", gte=1, lte=65535),
        Validator("mcp.request_timeout", gte=1, lte=300),
        # Logging validators
        Validator(
            "logging.level", is_in=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        ),
        Validator("logging.format", is_in=["json", "text"]),
    ],
)


# Note: Removed property decorator as it was incorrectly used at module level


# Helper function to get database URL
def get_database_url() -> str:
    """Get database URL from settings."""
    import os

    # Allow override for local testing
    if database_url := os.getenv("DATABASE_URL"):
        return database_url
    db = settings.database
    password = db.password
    return (
        f"postgresql+asyncpg://{db.user}:{password}@{db.host}:{db.port}/{db.database}"
    )


# Ensure directories exist
def ensure_directories() -> None:
    """Ensure required directories exist."""
    if settings.embeddings.use_cache:
        Path(settings.embeddings.cache_dir).mkdir(parents=True, exist_ok=True)

    if settings.logging.file_enabled:
        log_path = Path(settings.logging.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    if settings.monitoring.profiling_enabled:
        Path(settings.monitoring.profiling_path).mkdir(parents=True, exist_ok=True)


# Don't create directories on import - let applications handle it
