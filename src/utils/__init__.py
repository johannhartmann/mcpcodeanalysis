"""Utility modules for MCP Code Analysis Server."""

from src.logger import get_logger, setup_logging
from src.utils.exceptions import TimeoutError  # noqa: A004
from src.utils.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    DatabaseError,
    EmbeddingError,
    GitHubError,
    MCPError,
    NotFoundError,
    OpenAIError,
    ParsingError,
    QueryError,
    RateLimitError,
    RepositoryError,
    ValidationError,
    VectorSearchError,
    WebhookError,
)

__all__ = [
    "AuthenticationError",
    "AuthorizationError",
    "ConfigurationError",
    "DatabaseError",
    "EmbeddingError",
    "GitHubError",
    # Exceptions
    "MCPError",
    "NotFoundError",
    "OpenAIError",
    "ParsingError",
    "QueryError",
    "RateLimitError",
    "RepositoryError",
    "TimeoutError",
    "ValidationError",
    "VectorSearchError",
    "WebhookError",
    # Logging
    "get_logger",
    "setup_logging",
]
