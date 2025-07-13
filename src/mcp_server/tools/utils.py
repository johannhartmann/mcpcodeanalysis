"""Utility functions for MCP tools."""

import math
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Class, File, Function, Module
from src.logger import get_logger

logger = get_logger(__name__)


def format_file_path(path: str, repo_root: str | None = None) -> str:
    """Format file path for display.

    Args:
        path: File path to format
        repo_root: Repository root to strip from absolute paths

    Returns:
        Formatted file path
    """
    if not path:
        return path

    # Normalize path
    path = path.replace("\\", "/")

    # Remove leading ./
    if path.startswith("./"):
        path = path[2:]

    # Strip repository root if provided
    if repo_root and path.startswith(repo_root):
        path = path[len(repo_root) :].lstrip("/")

    return path


def format_function_signature(
    name: str,
    parameters: str | None = None,
    return_type: str | None = None,
) -> str:
    """Format function signature for display.

    Args:
        name: Function name
        parameters: JSON string of parameters or None
        return_type: Return type annotation or None

    Returns:
        Formatted function signature
    """
    # Parse parameters
    if parameters:
        try:
            import json

            params = json.loads(parameters)
            param_str = ", ".join(params)
        except:
            param_str = ""
    else:
        param_str = ""

    # Build signature
    signature = f"{name}({param_str})"

    # Add return type if present
    if return_type:
        signature += f" -> {return_type}"

    return signature


def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for human-readable display.

    Args:
        timestamp: Datetime to format

    Returns:
        Human-readable time string
    """
    if not timestamp:
        return "never"

    # Ensure timezone awareness
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    delta = now - timestamp

    # Just now
    if delta.total_seconds() < 60:
        return "just now"

    # Minutes
    minutes = int(delta.total_seconds() / 60)
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"

    # Hours
    hours = int(minutes / 60)
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"

    # Days
    days = int(hours / 24)
    if days < 30:
        return f"{days} day{'s' if days != 1 else ''} ago"

    # Older - show actual date
    return timestamp.strftime("%Y-%m-%d %H:%M")


def validate_entity_type(entity_type: str) -> bool:
    """Validate entity type.

    Args:
        entity_type: Entity type to validate

    Returns:
        True if valid, False otherwise
    """
    valid_types = {"function", "class", "module", "file"}
    return entity_type in valid_types


def validate_file_path(path: str) -> bool:
    """Validate file path for safety.

    Args:
        path: File path to validate

    Returns:
        True if valid and safe, False otherwise
    """
    if not path or not path.strip():
        return False

    # Check for null bytes
    if "\x00" in path:
        return False

    # Check for newlines
    if "\n" in path or "\r" in path:
        return False

    # Check for excessive path traversal
    if path.count("..") > 3:
        return False

    return True


def parse_entity_reference(reference: str) -> tuple[str | None, str | None, str | None]:
    """Parse entity reference string.

    Format: "type:name[@file]"

    Args:
        reference: Entity reference string

    Returns:
        Tuple of (entity_type, entity_name, file_path)
    """
    if not reference:
        return None, None, None

    # Parse type:name
    if ":" not in reference:
        return None, None, None

    entity_type, rest = reference.split(":", 1)

    if not rest:
        return None, None, None

    # Parse optional @file
    if "@" in rest:
        entity_name, file_path = rest.split("@", 1)
    else:
        entity_name = rest
        file_path = None

    return entity_type, entity_name, file_path


def paginate_results(
    items: list[Any],
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """Paginate a list of results.

    Args:
        items: List of items to paginate
        page: Page number (1-based)
        page_size: Items per page

    Returns:
        Dictionary with paginated results
    """
    total = len(items)
    pages = math.ceil(total / page_size) if page_size > 0 else 1

    # Calculate slice indices
    start = (page - 1) * page_size
    end = start + page_size

    # Get page items
    page_items = items[start:end]

    return {
        "items": page_items,
        "total": total,
        "page": page,
        "pages": pages,
        "page_size": page_size,
        "has_next": page < pages,
        "has_prev": page > 1,
    }


async def get_entity_by_type_and_id(
    db_session: AsyncSession,
    entity_type: str,
    entity_id: int,
) -> Function | Class | Module | File | None:
    """Get entity by type and ID.

    Args:
        db_session: Database session
        entity_type: Type of entity
        entity_id: Entity ID

    Returns:
        Entity object or None
    """
    if not validate_entity_type(entity_type):
        raise ValueError(f"Invalid entity type: {entity_type}")

    # Map types to models
    type_map = {
        "function": Function,
        "class": Class,
        "module": Module,
        "file": File,
    }

    model = type_map[entity_type]

    # Query entity
    result = await db_session.execute(select(model).where(model.id == entity_id))

    return result.scalar_one_or_none()


async def get_file_content_safe(
    file_path: str,
    max_size: int = 1_000_000,  # 1MB default
) -> str | None:
    """Safely read file content with size limit.

    Args:
        file_path: Path to file
        max_size: Maximum file size to read

    Returns:
        File content or None if error
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read(max_size)
            return content
    except (FileNotFoundError, PermissionError, OSError):
        return None
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None


def format_error_response(
    message: str,
    code: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Format error response for consistency.

    Args:
        message: Error message
        code: Optional error code
        details: Optional error details

    Returns:
        Formatted error response
    """
    response = {
        "status": "error",
        "error": message,
    }

    if code:
        response["code"] = code

    if details:
        response["details"] = details

    return response


def sanitize_output(data: dict[str, Any]) -> dict[str, Any]:
    """Sanitize output by removing sensitive data.

    Args:
        data: Data dictionary to sanitize

    Returns:
        Sanitized data
    """
    sensitive_keys = {
        "password",
        "secret",
        "token",
        "key",
        "api_key",
        "access_token",
        "refresh_token",
        "private_key",
    }

    sanitized = {}
    for key, value in data.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            sanitized[key] = "[REDACTED]"
        else:
            sanitized[key] = value

    return sanitized


def calculate_similarity_score(
    text1: str,
    text2: str,
    case_sensitive: bool = True,
) -> float:
    """Calculate simple similarity score between two strings.

    Args:
        text1: First text
        text2: Second text
        case_sensitive: Whether to consider case

    Returns:
        Similarity score between 0 and 1
    """
    if not case_sensitive:
        text1 = text1.lower()
        text2 = text2.lower()

    if text1 == text2:
        return 1.0

    # Simple character-based similarity
    longer = max(len(text1), len(text2))
    if longer == 0:
        return 1.0

    # Count matching characters
    matches = sum(c1 == c2 for c1, c2 in zip(text1, text2))

    # Basic similarity score
    score = matches / longer

    return score


def parse_code_location(location: str) -> tuple[str | None, int | None, int | None]:
    """Parse code location string.

    Format: "file:line:column" or "file:line" or "file"

    Args:
        location: Location string

    Returns:
        Tuple of (file_path, line, column)
    """
    if not location:
        return None, None, None

    parts = location.split(":")

    if len(parts) == 1:
        return parts[0], None, None
    elif len(parts) == 2:
        try:
            return parts[0], int(parts[1]), None
        except ValueError:
            return None, None, None
    elif len(parts) == 3:
        try:
            return parts[0], int(parts[1]), int(parts[2])
        except ValueError:
            return None, None, None
    else:
        return None, None, None
