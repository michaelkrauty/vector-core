"""Validation utilities for MCP tool parameters."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..errors import ErrorCode

# Default bounds for limit validation
DEFAULT_MIN_LIMIT = 1
DEFAULT_MAX_LIMIT = 100000  # Effectively unlimited for maximum recall


def validate_limit(
    limit: int | None,
    default: int = 10,
    minimum: int = DEFAULT_MIN_LIMIT,
    maximum: int = DEFAULT_MAX_LIMIT,
) -> int:
    """
    Validate and clamp a limit parameter to safe bounds.

    This provides consistent limit validation across all MCP servers,
    preventing unbounded queries and ensuring sensible defaults.

    Args:
        limit: The limit value to validate (can be None)
        default: Default value to use if limit is None or <= 0
        minimum: Minimum allowed value (default: 1)
        maximum: Maximum allowed value (default: 100000)

    Returns:
        Validated limit clamped to [minimum, maximum]

    Examples:
        >>> validate_limit(50)
        50
        >>> validate_limit(None)
        10
        >>> validate_limit(0)
        10
        >>> validate_limit(500)
        500
        >>> validate_limit(5, default=20, maximum=50)
        5
    """
    if limit is None or limit <= 0:
        return default
    return max(minimum, min(maximum, limit))


def validate_uuid_string(uuid_str: str) -> bool:
    """
    Validate that a string is a valid UUID format.

    Args:
        uuid_str: String to validate

    Returns:
        True if valid UUID format, False otherwise
    """
    from uuid import UUID

    try:
        UUID(uuid_str)
        return True
    except (ValueError, TypeError):
        return False


def parse_uuid_or_none(uuid_str: str) -> "UUID | None":
    """
    Parse a UUID string, returning None if invalid.

    Useful for MCP tool parameters that accept UUID strings.

    Args:
        uuid_str: String to parse

    Returns:
        UUID if valid, None otherwise
    """
    from uuid import UUID

    try:
        return UUID(uuid_str)
    except (ValueError, TypeError):
        return None


def validate_directory_path(path: str) -> Path | dict:
    """
    Validate that a path exists and is a directory.

    Returns the resolved Path if valid, or an error_response dict if invalid.
    This reduces boilerplate in MCP tool handlers that validate directory paths.

    Args:
        path: Path string to validate

    Returns:
        Path: The resolved Path object if valid
        dict: An error_response dict if invalid (check with is_error_response())

    Example:
        >>> result = validate_directory_path("/home/user/project")
        >>> if isinstance(result, dict):  # It's an error
        ...     return result
        >>> # result is now a valid Path
        >>> abs_path = str(result)
    """
    from ..errors import ErrorCode, error_response

    resolved = Path(path).resolve()
    if not resolved.exists():
        return error_response(ErrorCode.FILE_NOT_FOUND, f"Path does not exist: {path}")
    if not resolved.is_dir():
        return error_response(ErrorCode.VALIDATION_FAILED, f"Path is not a directory: {path}")
    return resolved


def validate_file_path(path: str) -> Path | dict:
    """
    Validate that a path exists and is a file.

    Returns the resolved Path if valid, or an error_response dict if invalid.
    This reduces boilerplate in MCP tool handlers that validate file paths.

    Args:
        path: Path string to validate

    Returns:
        Path: The resolved Path object if valid
        dict: An error_response dict if invalid (check with is_error_response())

    Example:
        >>> result = validate_file_path("/home/user/file.txt")
        >>> if isinstance(result, dict):  # It's an error
        ...     return result
        >>> # result is now a valid Path to a file
    """
    from ..errors import ErrorCode, error_response

    resolved = Path(path).resolve()
    if not resolved.exists():
        return error_response(ErrorCode.FILE_NOT_FOUND, f"File does not exist: {path}")
    if not resolved.is_file():
        return error_response(ErrorCode.VALIDATION_FAILED, f"Path is not a file: {path}")
    return resolved
