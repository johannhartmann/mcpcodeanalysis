"""Tests for MCP tools utility functions and helpers."""

# mypy: disallow-any-generics=False


from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Function
from src.mcp_server.tools.utils import (
    format_file_path,
    format_function_signature,
    format_timestamp,
    get_entity_by_type_and_id,
    get_file_content_safe,
    paginate_results,
    parse_entity_reference,
    validate_entity_type,
    validate_file_path,
)


@pytest.fixture
def mock_db_session() -> AsyncMock:
    """Create mock database session."""
    return AsyncMock(spec=AsyncSession)


class TestToolUtils:
    """Tests for tool utility functions."""

    def test_format_file_path_absolute(self) -> None:
        """Test formatting absolute file paths."""
        assert format_file_path("/src/models/user.py") == "/src/models/user.py"
        assert (
            format_file_path("/home/user/project/file.py")
            == "/home/user/project/file.py"
        )

    def test_format_file_path_relative(self) -> None:
        """Test formatting relative file paths."""
        assert format_file_path("src/models/user.py") == "src/models/user.py"
        assert format_file_path("./src/file.py") == "src/file.py"
        assert format_file_path("../parent/file.py") == "../parent/file.py"

    def test_format_file_path_with_repo_root(self) -> None:
        """Test formatting file paths with repository root."""
        assert format_file_path("/repo/src/file.py", repo_root="/repo") == "src/file.py"
        assert (
            format_file_path(
                "/home/user/project/src/file.py", repo_root="/home/user/project"
            )
            == "src/file.py"
        )

    def test_format_function_signature(self) -> None:
        """Test formatting function signatures."""
        # Simple function
        assert (
            format_function_signature("test_func", parameters=None, return_type=None)
            == "test_func()"
        )

        # With parameters
        assert (
            format_function_signature(
                "test_func", parameters='["arg1", "arg2"]', return_type=None
            )
            == "test_func(arg1, arg2)"
        )

        # With return type
        assert (
            format_function_signature("test_func", parameters=None, return_type="str")
            == "test_func() -> str"
        )

        # Complete signature
        assert (
            format_function_signature(
                "process_data",
                parameters='["data: List[str]", "config: Dict[str, Any]"]',
                return_type="Optional[Result]",
            )
            == "process_data(data: List[str], config: Dict[str, Any]) -> Optional[Result]"
        )

    def test_format_function_signature_with_defaults(self) -> None:
        """Test formatting function signatures with default values."""
        assert (
            format_function_signature(
                "create_user",
                parameters='["name: str", "age: int = 18", "active: bool = True"]',
                return_type="User",
            )
            == "create_user(name: str, age: int = 18, active: bool = True) -> User"
        )

    def test_format_timestamp(self) -> None:
        """Test formatting timestamps."""
        # Recent timestamp (less than a minute)
        now = datetime.now(UTC)
        assert format_timestamp(now) == "just now"

        # Minutes ago
        minutes_ago = datetime.now(UTC).replace(minute=datetime.now(UTC).minute - 5)
        result = format_timestamp(minutes_ago)
        assert "5 minutes ago" in result or "4 minutes ago" in result

        # Hours ago
        from datetime import timedelta

        hours_ago = datetime.now(UTC) - timedelta(hours=3)
        assert "3 hours ago" in format_timestamp(hours_ago)

        # Days ago
        days_ago = datetime.now(UTC) - timedelta(days=2)
        assert "2 days ago" in format_timestamp(days_ago)

        # Specific date for old timestamps
        old_date = datetime(2023, 1, 15, 10, 30, tzinfo=UTC)
        assert format_timestamp(old_date) == "2023-01-15 10:30"

    def test_validate_entity_type(self) -> None:
        """Test entity type validation."""
        valid_types = ["function", "class", "module", "file"]
        for entity_type in valid_types:
            assert validate_entity_type(entity_type) is True

        invalid_types = ["method", "variable", "package", ""]
        for entity_type in invalid_types:
            assert validate_entity_type(entity_type) is False

    def test_validate_file_path(self) -> None:
        """Test file path validation."""
        # Valid paths
        valid_paths = [
            "/src/models/user.py",
            "src/utils/helpers.py",
            "./tests/test_user.py",
            "../lib/core.py",
            "/home/user/project/file.py",
        ]
        for path in valid_paths:
            assert validate_file_path(path) is True

        # Invalid paths
        invalid_paths = [
            "",
            " ",
            "file.py\x00",  # Null byte
            "file\nname.py",  # Newline
            "../../../../../../etc/passwd",  # Path traversal attempt
        ]
        for path in invalid_paths:
            assert validate_file_path(path) is False

    def test_parse_entity_reference(self) -> None:
        """Test parsing entity references."""
        # Function reference
        result = parse_entity_reference("function:process_data")
        assert result == ("function", "process_data", None)

        # Class reference
        result = parse_entity_reference("class:UserModel")
        assert result == ("class", "UserModel", None)

        # Module reference
        result = parse_entity_reference("module:src.models")
        assert result == ("module", "src.models", None)

        # With file context
        result = parse_entity_reference("function:save@/src/models/user.py")
        assert result == ("function", "save", "/src/models/user.py")

        # Invalid references
        assert parse_entity_reference("invalid") == (None, None, None)
        assert parse_entity_reference("") == (None, None, None)
        assert parse_entity_reference("function:") == (None, None, None)

    def test_paginate_results(self) -> None:
        """Test result pagination."""
        items = list(range(100))

        # First page
        page1 = paginate_results(items, page=1, page_size=10)
        assert page1["items"] == list(range(10))
        assert page1["total"] == 100
        assert page1["page"] == 1
        assert page1["pages"] == 10
        assert page1["has_next"] is True
        assert page1["has_prev"] is False

        # Middle page
        page5 = paginate_results(items, page=5, page_size=10)
        assert page5["items"] == list(range(40, 50))
        assert page5["page"] == 5
        assert page5["has_next"] is True
        assert page5["has_prev"] is True

        # Last page
        page10 = paginate_results(items, page=10, page_size=10)
        assert page10["items"] == list(range(90, 100))
        assert page10["has_next"] is False
        assert page10["has_prev"] is True

        # Page beyond range
        page_invalid = paginate_results(items, page=20, page_size=10)
        assert page_invalid["items"] == []
        assert page_invalid["page"] == 20

    def test_paginate_results_custom_size(self) -> None:
        """Test pagination with custom page sizes."""
        items = list(range(50))

        # Large page size
        large_page = paginate_results(items, page=1, page_size=25)
        assert len(large_page["items"]) == 25
        assert large_page["pages"] == 2

        # Small page size
        small_page = paginate_results(items, page=3, page_size=5)
        assert len(small_page["items"]) == 5
        assert small_page["items"] == list(range(10, 15))
        assert small_page["pages"] == 10

    @pytest.mark.asyncio
    async def test_get_entity_by_type_and_id_function(
        self, mock_db_session: AsyncMock
    ) -> None:
        """Test getting entity by type and ID for functions."""
        # Mock function
        mock_function = MagicMock(spec=Function)
        mock_function.id = 10
        mock_function.name = "test_function"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_function

        mock_db_session.execute.return_value = mock_result

        result = await get_entity_by_type_and_id(mock_db_session, "function", 10)

        assert result == mock_function

    @pytest.mark.asyncio
    async def test_get_entity_by_type_and_id_not_found(
        self, mock_db_session: AsyncMock
    ) -> None:
        """Test getting entity that doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_db_session.execute.return_value = mock_result

        result = await get_entity_by_type_and_id(mock_db_session, "function", 999)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_entity_by_type_and_id_invalid_type(
        self, mock_db_session: AsyncMock
    ) -> None:
        """Test getting entity with invalid type."""
        with pytest.raises(ValueError, match="Invalid entity type"):
            await get_entity_by_type_and_id(mock_db_session, "invalid_type", 10)

    @pytest.mark.asyncio
    async def test_get_file_content_safe(self) -> None:
        """Test safely reading file content."""
        # Mock successful file read via Path.read_text used by implementation
        with patch(
            "pathlib.Path.read_text", return_value="def hello():\n    return 'world'"
        ) as mock_read:
            content = await get_file_content_safe("/src/test.py")

            assert content == "def hello():\n    return 'world'"
            assert mock_read.called

    @pytest.mark.asyncio
    async def test_get_file_content_safe_not_found(self) -> None:
        """Test reading non-existent file."""
        with patch("builtins.open", side_effect=FileNotFoundError()):
            content = await get_file_content_safe("/nonexistent.py")
            assert content is None

    @pytest.mark.asyncio
    async def test_get_file_content_safe_permission_error(self) -> None:
        """Test reading file with permission error."""
        with patch("builtins.open", side_effect=PermissionError()):
            content = await get_file_content_safe("/restricted.py")
            assert content is None

    @pytest.mark.asyncio
    async def test_get_file_content_safe_max_size(self) -> None:
        """Test reading file with size limit."""
        large_content = "x" * 2_000_000  # 2MB

        # Patch Path.read_text to return a large content blob, implementation slices it
        with patch("pathlib.Path.read_text", return_value=large_content):
            content = await get_file_content_safe("/large.py", max_size=1_000_000)

            assert content is not None
            assert len(content) == 1_000_000
            assert content == large_content[:1_000_000]

    def test_format_error_response(self) -> None:
        """Test formatting error responses."""
        from src.mcp_server.tools.utils import format_error_response

        # Simple error
        result = format_error_response("File not found")
        assert result["error"] == "File not found"
        assert result["status"] == "error"

        # With details
        result = format_error_response(
            "Invalid input",
            details={"field": "name", "reason": "too short"},
        )
        assert result["error"] == "Invalid input"
        assert result["details"]["field"] == "name"

        # With error code
        result = format_error_response("Permission denied", code="PERMISSION_DENIED")
        assert result["code"] == "PERMISSION_DENIED"

    def test_sanitize_output(self) -> None:
        """Test output sanitization."""
        from src.mcp_server.tools.utils import sanitize_output

        # Remove sensitive data
        data = {
            "name": "test",
            "password": "secret123",
            "api_key": "sk-123456",
            "token": "ghp_abc123",
            "normal_field": "visible",
        }

        sanitized = sanitize_output(data)
        assert sanitized["name"] == "test"
        assert sanitized["password"] == "[REDACTED]"
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["token"] == "[REDACTED]"
        assert sanitized["normal_field"] == "visible"

    def test_calculate_similarity_score(self) -> None:
        """Test similarity score calculation."""
        from src.mcp_server.tools.utils import calculate_similarity_score

        # Identical strings
        assert calculate_similarity_score("hello", "hello") == 1.0

        # Completely different
        assert calculate_similarity_score("hello", "xyz") < 0.3

        # Similar strings
        score = calculate_similarity_score("hello world", "hello word")
        assert 0.8 < score < 1.0

        # Case insensitive option
        assert calculate_similarity_score("Hello", "hello", case_sensitive=False) == 1.0

    def test_parse_code_location(self) -> None:
        """Test parsing code location strings."""
        from src.mcp_server.tools.utils import parse_code_location

        # File and line
        result = parse_code_location("/src/models/user.py:42")
        assert result == ("/src/models/user.py", 42, None)

        # File, line, and column
        result = parse_code_location("/src/utils/helpers.py:10:5")
        assert result == ("/src/utils/helpers.py", 10, 5)

        # Just file
        result = parse_code_location("/src/main.py")
        assert result == ("/src/main.py", None, None)

        # Invalid format
        assert parse_code_location("invalid:location:format:extra") == (
            None,
            None,
            None,
        )
