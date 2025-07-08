"""Custom exceptions for MCP Code Analysis Server."""

from typing import Any, Dict, Optional


class MCPError(Exception):
    """Base exception for MCP server errors."""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}


class ConfigurationError(MCPError):
    """Configuration related errors."""
    pass


class RepositoryError(MCPError):
    """Repository operation errors."""
    pass


class GitHubError(RepositoryError):
    """GitHub API related errors."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        github_error: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        if github_error:
            self.details["github_error"] = github_error


class ParserError(MCPError):
    """Alias for ParsingError for backward compatibility."""
    pass


class ParsingError(MCPError):
    """Code parsing errors."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        language: Optional[str] = None,
    ):
        super().__init__(message)
        if file_path:
            self.details["file_path"] = file_path
        if line_number:
            self.details["line_number"] = line_number
        if language:
            self.details["language"] = language


class EmbeddingError(MCPError):
    """Embedding generation errors."""
    pass


class OpenAIError(EmbeddingError):
    """OpenAI API related errors."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        openai_error: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        if openai_error:
            self.details["openai_error"] = openai_error


class DatabaseError(MCPError):
    """Database operation errors."""
    pass


class QueryError(MCPError):
    """Query processing errors."""
    pass


class VectorSearchError(QueryError):
    """Vector similarity search errors."""
    pass


class RateLimitError(MCPError):
    """Rate limiting errors."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        limit: Optional[int] = None,
        remaining: Optional[int] = None,
    ):
        super().__init__(message)
        if retry_after:
            self.details["retry_after"] = retry_after
        if limit:
            self.details["limit"] = limit
        if remaining:
            self.details["remaining"] = remaining


class AuthenticationError(MCPError):
    """Authentication related errors."""
    pass


class AuthorizationError(MCPError):
    """Authorization related errors."""
    pass


class ValidationError(MCPError):
    """Data validation errors."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        errors: Optional[list] = None,
    ):
        super().__init__(message)
        if field:
            self.details["field"] = field
        if value is not None:
            self.details["value"] = value
        if errors:
            self.details["errors"] = errors


class NotFoundError(MCPError):
    """Resource not found errors."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
    ):
        super().__init__(message)
        if resource_type:
            self.details["resource_type"] = resource_type
        if resource_id:
            self.details["resource_id"] = resource_id


class TimeoutError(MCPError):
    """Operation timeout errors."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        super().__init__(message)
        if operation:
            self.details["operation"] = operation
        if timeout_seconds:
            self.details["timeout_seconds"] = timeout_seconds


class WebhookError(MCPError):
    """Webhook processing errors."""
    
    def __init__(
        self,
        message: str,
        event_type: Optional[str] = None,
        delivery_id: Optional[str] = None,
    ):
        super().__init__(message)
        if event_type:
            self.details["event_type"] = event_type
        if delivery_id:
            self.details["delivery_id"] = delivery_id