"""Tests for custom exceptions."""

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
    TimeoutError,
    ValidationError,
    VectorSearchError,
    WebhookError,
)


class TestMCPError:
    """Test base MCPError."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        error = MCPError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.code == "MCPError"
        assert error.details == {}

    def test_error_with_code(self) -> None:
        """Test error with custom code."""
        error = MCPError("Test error", code="TEST_001")
        assert error.code == "TEST_001"

    def test_error_with_details(self) -> None:
        """Test error with details."""
        details = {"field": "username", "reason": "invalid"}
        error = MCPError("Validation failed", details=details)
        assert error.details == details


class TestGitHubError:
    """Test GitHubError."""

    def test_github_error(self) -> None:
        """Test GitHub error creation."""
        error = GitHubError(
            "API rate limit exceeded",
            status_code=429,
            github_error={"message": "API rate limit exceeded"},
        )
        assert error.message == "API rate limit exceeded"
        assert error.status_code == 429
        assert error.details["github_error"]["message"] == "API rate limit exceeded"

    def test_github_error_inheritance(self) -> None:
        """Test GitHubError inheritance."""
        error = GitHubError("Test error")
        assert isinstance(error, RepositoryError)
        assert isinstance(error, MCPError)


class TestParsingError:
    """Test ParsingError."""

    def test_parsing_error_minimal(self) -> None:
        """Test parsing error with minimal info."""
        error = ParsingError("Syntax error")
        assert error.message == "Syntax error"
        assert error.details == {}

    def test_parsing_error_full(self) -> None:
        """Test parsing error with full info."""
        error = ParsingError(
            "Unexpected token",
            file_path="/src/main.py",
            line_number=42,
            language="python",
        )
        assert error.details["file_path"] == "/src/main.py"
        assert error.details["line_number"] == 42
        assert error.details["language"] == "python"


class TestOpenAIError:
    """Test OpenAIError."""

    def test_openai_error(self) -> None:
        """Test OpenAI error creation."""
        error = OpenAIError(
            "Invalid API key",
            status_code=401,
            openai_error={"error": {"message": "Invalid API key"}},
        )
        assert error.status_code == 401
        assert error.details["openai_error"]["error"]["message"] == "Invalid API key"

    def test_openai_error_inheritance(self) -> None:
        """Test OpenAIError inheritance."""
        error = OpenAIError("Test error")
        assert isinstance(error, EmbeddingError)
        assert isinstance(error, MCPError)


class TestRateLimitError:
    """Test RateLimitError."""

    def test_rate_limit_error_minimal(self) -> None:
        """Test rate limit error with minimal info."""
        error = RateLimitError("Rate limit exceeded")
        assert error.message == "Rate limit exceeded"
        assert error.details == {}

    def test_rate_limit_error_full(self) -> None:
        """Test rate limit error with full info."""
        error = RateLimitError(
            "API rate limit exceeded",
            retry_after=60,
            limit=100,
            remaining=0,
        )
        assert error.details["retry_after"] == 60
        assert error.details["limit"] == 100
        assert error.details["remaining"] == 0


class TestValidationError:
    """Test ValidationError."""

    def test_validation_error_minimal(self) -> None:
        """Test validation error with minimal info."""
        error = ValidationError("Invalid input")
        assert error.message == "Invalid input"
        assert error.details == {}

    def test_validation_error_with_field(self) -> None:
        """Test validation error with field info."""
        error = ValidationError(
            "Invalid email format",
            field="email",
            value="invalid@",
        )
        assert error.details["field"] == "email"
        assert error.details["value"] == "invalid@"

    def test_validation_error_with_errors_list(self) -> None:
        """Test validation error with errors list."""
        errors = [
            {"field": "username", "message": "Too short"},
            {"field": "password", "message": "Too weak"},
        ]
        error = ValidationError("Multiple validation errors", errors=errors)
        assert error.details["errors"] == errors


class TestNotFoundError:
    """Test NotFoundError."""

    def test_not_found_error_minimal(self) -> None:
        """Test not found error with minimal info."""
        error = NotFoundError("Resource not found")
        assert error.message == "Resource not found"
        assert error.details == {}

    def test_not_found_error_full(self) -> None:
        """Test not found error with full info."""
        error = NotFoundError(
            "Repository not found",
            resource_type="repository",
            resource_id="123",
        )
        assert error.details["resource_type"] == "repository"
        assert error.details["resource_id"] == "123"


class TestTimeoutError:
    """Test TimeoutError."""

    def test_timeout_error_minimal(self) -> None:
        """Test timeout error with minimal info."""
        error = TimeoutError("Operation timed out")
        assert error.message == "Operation timed out"
        assert error.details == {}

    def test_timeout_error_full(self) -> None:
        """Test timeout error with full info."""
        error = TimeoutError(
            "Database query timeout",
            operation="vector_search",
            timeout_seconds=30.0,
        )
        assert error.details["operation"] == "vector_search"
        assert error.details["timeout_seconds"] == 30.0


class TestWebhookError:
    """Test WebhookError."""

    def test_webhook_error_minimal(self) -> None:
        """Test webhook error with minimal info."""
        error = WebhookError("Webhook processing failed")
        assert error.message == "Webhook processing failed"
        assert error.details == {}

    def test_webhook_error_full(self) -> None:
        """Test webhook error with full info."""
        error = WebhookError(
            "Invalid webhook signature",
            event_type="push",
            delivery_id="12345-67890",
        )
        assert error.details["event_type"] == "push"
        assert error.details["delivery_id"] == "12345-67890"


class TestErrorHierarchy:
    """Test error class hierarchy."""

    def test_all_errors_inherit_from_mcp_error(self) -> None:
        """Test that all custom errors inherit from MCPError."""
        error_classes = [
            ConfigurationError,
            RepositoryError,
            GitHubError,
            ParsingError,
            EmbeddingError,
            OpenAIError,
            DatabaseError,
            QueryError,
            VectorSearchError,
            RateLimitError,
            AuthenticationError,
            AuthorizationError,
            ValidationError,
            NotFoundError,
            TimeoutError,
            WebhookError,
        ]

        for error_class in error_classes:
            error = error_class("Test")
            assert isinstance(error, MCPError)

    def test_specific_inheritance(self) -> None:
        """Test specific inheritance relationships."""
        # GitHubError inherits from RepositoryError
        assert issubclass(GitHubError, RepositoryError)

        # OpenAIError inherits from EmbeddingError
        assert issubclass(OpenAIError, EmbeddingError)

        # VectorSearchError inherits from QueryError
        assert issubclass(VectorSearchError, QueryError)
