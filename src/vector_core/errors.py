"""Error handling utilities for MCP servers.

This module provides:
1. ErrorCode enum for structured API error responses
2. error_response() helper for creating consistent error dicts
3. ErrorCollector for batch operations that need to report multiple issues

API Error Codes Usage:
    from vector_core.errors import ErrorCode, error_response

    @mcp.tool()
    async def my_tool(note_id: str) -> dict:
        try:
            uuid = UUID(note_id)
        except ValueError:
            return error_response(ErrorCode.INVALID_UUID, f"Invalid UUID: {note_id}")

        note = store.read(uuid)
        if not note:
            return error_response(ErrorCode.NOT_FOUND, f"Note not found: {note_id}")

        return note.model_dump(mode="json")

Batch Error Collection Usage:
    collector = ErrorCollector()

    for file in files:
        try:
            process(file)
        except FileNotFoundError as e:
            collector.add_file_access_error(file, e)
        except UnicodeDecodeError as e:
            collector.add_encoding_error(file, e)

    if collector.has_errors:
        print(collector.format_summary())
"""

import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# =============================================================================
# API Error Codes - for structured tool responses
# =============================================================================


class ErrorCode(str, Enum):
    """Structured error codes for MCP tool responses.

    Using string enums for JSON serialization compatibility.
    """

    # Validation errors
    INVALID_UUID = "invalid_uuid"
    INVALID_INPUT = "invalid_input"
    VALIDATION_FAILED = "validation_failed"

    # Not found errors
    NOT_FOUND = "not_found"
    COLLECTION_NOT_FOUND = "collection_not_found"
    NOTE_NOT_FOUND = "note_not_found"
    FACT_NOT_FOUND = "fact_not_found"
    GLOSSARY_NOT_FOUND = "glossary_not_found"

    # Service availability errors
    SERVICE_UNAVAILABLE = "service_unavailable"
    EMBEDDING_SERVICE_ERROR = "embedding_service_error"
    QDRANT_ERROR = "qdrant_error"
    DATABASE_ERROR = "database_error"

    # Operation errors
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"

    # Resource errors
    FILE_NOT_FOUND = "file_not_found"
    FILE_ACCESS_ERROR = "file_access_error"
    ENCODING_ERROR = "encoding_error"
    PARSE_ERROR = "parse_error"

    # Internal errors
    INTERNAL_ERROR = "internal_error"
    NOT_IMPLEMENTED = "not_implemented"


def error_response(
    code: ErrorCode,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a standardized error response dict for MCP tools.

    Args:
        code: The error code enum value
        message: Human-readable error message
        details: Optional additional details (e.g., field-level errors)

    Returns:
        Dict with error_code, message, and optionally details

    Example:
        return error_response(
            ErrorCode.NOT_FOUND,
            f"Note not found: {note_id}",
            details={"searched_id": note_id}
        )
    """
    response: dict[str, Any] = {
        "error_code": code.value,
        "message": message,
    }
    if details:
        response["details"] = details
    return response


def is_error_response(response: dict[str, Any]) -> bool:
    """Check if a response dict is an error response.

    Args:
        response: Dict to check

    Returns:
        True if response contains error_code key
    """
    return "error_code" in response


def format_error(response: dict[str, Any]) -> str:
    """Format an error_response dict as a human-readable string.

    Use this in MCP tools that return str but need to handle error_response dicts
    from validation helpers like validate_directory_path().

    Args:
        response: An error_response dict with error_code, message, and optional details

    Returns:
        Formatted error string

    Example:
        result = validate_directory_path(path)
        if isinstance(result, dict):
            return format_error(result)  # Returns "Error: Path does not exist: /foo"
    """
    message = response.get("message", "Unknown error")
    details = response.get("details")

    if details:
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        return f"Error: {message} ({detail_str})"
    return f"Error: {message}"


# =============================================================================
# Exception Classes - for specific error handling
# =============================================================================


class DatabaseError(Exception):
    """Base exception for database operations.

    Use this for general database errors that don't fit a more specific category.
    """

    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails.

    Attributes:
        db_path: Path to the database that failed to connect
        operation: The operation that was attempted
        original: The original exception that caused this error
    """

    def __init__(
        self,
        db_path: str,
        operation: str = "connect",
        original: Exception | None = None,
    ):
        self.db_path = db_path
        self.operation = operation
        self.original = original
        super().__init__(f"Database connection failed during {operation}: {db_path}")


class DatabaseIntegrityError(DatabaseError):
    """Raised when database integrity constraint is violated.

    Attributes:
        message: Description of the integrity violation
        constraint: Name of the violated constraint (if known)
    """

    def __init__(self, message: str, constraint: str | None = None):
        self.constraint = constraint
        super().__init__(message)


class DatabaseLockError(DatabaseError):
    """Raised when database lock cannot be acquired.

    Attributes:
        db_path: Path to the locked database
        timeout: The timeout value that was exceeded
    """

    def __init__(self, db_path: str, timeout: float):
        self.db_path = db_path
        self.timeout = timeout
        super().__init__(f"Database locked: {db_path} (timeout: {timeout}s)")


class StorageError(Exception):
    """Base exception for vector storage operations.

    Use this for Qdrant or other vector database errors.
    """

    pass


class CollectionError(StorageError):
    """Raised for collection-level errors (create, delete, not found).

    Attributes:
        collection: Name of the collection
        operation: The operation that failed
    """

    def __init__(self, collection: str, operation: str, message: str | None = None):
        self.collection = collection
        self.operation = operation
        msg = message or f"Collection {operation} failed: {collection}"
        super().__init__(msg)


class PointOperationError(StorageError):
    """Raised when point upsert/delete fails.

    Attributes:
        collection: Name of the collection
        operation: The operation that failed (upsert, delete, etc.)
        point_count: Number of points involved in the failed operation
    """

    def __init__(self, collection: str, operation: str, point_count: int = 0):
        self.collection = collection
        self.operation = operation
        self.point_count = point_count
        super().__init__(
            f"Point {operation} failed on {collection} ({point_count} points)"
        )


# =============================================================================
# Batch Error Collection - for indexing operations
# =============================================================================


class ErrorSeverity(Enum):
    """Severity level of errors."""

    WARNING = "warning"  # Non-fatal, operation continued
    ERROR = "error"  # Fatal for this file, skipped
    CRITICAL = "critical"  # Fatal for operation


class ErrorCategory(Enum):
    """Category of errors for grouping."""

    FILE_ACCESS = "file_access"
    PARSE_ERROR = "parse_error"
    ENCODING = "encoding"
    EMBEDDING = "embedding"
    STORAGE = "storage"
    UNKNOWN = "unknown"


@dataclass
class IndexingError:
    """A single error encountered during indexing."""

    path: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    exception_type: str | None = None  # e.g., "FileNotFoundError"
    exception_message: str | None = None  # e.g., "No such file or directory"
    traceback_str: str | None = None  # Full traceback for debugging


@dataclass
class ErrorCollector:
    """Collects errors during indexing for summary reporting."""

    errors: list[IndexingError] = field(default_factory=list)
    max_errors: int = 100  # Limit stored errors to avoid memory issues
    _truncated_count: int = field(default=0, repr=False)
    capture_tracebacks: bool = True  # Whether to capture full tracebacks

    def add(
        self,
        path: str | Path,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        exception: Exception | None = None,
    ) -> None:
        """Add an error to the collection."""
        if len(self.errors) >= self.max_errors:
            self._truncated_count += 1
            return

        # Extract exception context
        exception_type: str | None = None
        exception_message: str | None = None
        traceback_str: str | None = None

        if exception is not None:
            exception_type = type(exception).__name__
            exception_message = str(exception)
            if self.capture_tracebacks:
                # Capture traceback at the point of error collection
                traceback_str = "".join(
                    traceback.format_exception(type(exception), exception, exception.__traceback__)
                )

        self.errors.append(
            IndexingError(
                path=str(path),
                message=message,
                severity=severity,
                category=category,
                exception_type=exception_type,
                exception_message=exception_message,
                traceback_str=traceback_str,
            )
        )

    @property
    def truncated_count(self) -> int:
        """Number of errors that were not stored due to max_errors limit."""
        return self._truncated_count

    def add_file_access_error(
        self, path: str | Path, exception: Exception | None = None
    ) -> None:
        """Convenience method for file access errors."""
        self.add(
            path=path,
            message="Failed to access file",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.FILE_ACCESS,
            exception=exception,
        )

    def add_encoding_error(
        self, path: str | Path, exception: Exception | None = None
    ) -> None:
        """Convenience method for encoding errors."""
        self.add(
            path=path,
            message="Failed to decode file (encoding issue)",
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.ENCODING,
            exception=exception,
        )

    def add_parse_error(
        self, path: str | Path, exception: Exception | None = None
    ) -> None:
        """Convenience method for parsing errors."""
        self.add(
            path=path,
            message="Failed to parse file",
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.PARSE_ERROR,
            exception=exception,
        )

    def add_embedding_error(
        self, path: str | Path, exception: Exception | None = None
    ) -> None:
        """Convenience method for embedding errors."""
        self.add(
            path=path,
            message="Failed to generate embeddings",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.EMBEDDING,
            exception=exception,
        )

    def add_storage_error(
        self, path: str | Path, exception: Exception | None = None
    ) -> None:
        """Convenience method for storage errors."""
        self.add(
            path=path,
            message="Failed to store in database",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.STORAGE,
            exception=exception,
        )

    @property
    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self.errors) > 0

    @property
    def error_count(self) -> int:
        """Total number of errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Number of warnings."""
        return sum(1 for e in self.errors if e.severity == ErrorSeverity.WARNING)

    @property
    def critical_count(self) -> int:
        """Number of critical errors."""
        return sum(1 for e in self.errors if e.severity == ErrorSeverity.CRITICAL)

    def by_category(self) -> dict[ErrorCategory, list[IndexingError]]:
        """Group errors by category."""
        result: dict[ErrorCategory, list[IndexingError]] = {}
        for error in self.errors:
            if error.category not in result:
                result[error.category] = []
            result[error.category].append(error)
        return result

    def format_summary(self, include_exceptions: bool = False) -> str:
        """Format a human-readable summary of errors.

        Args:
            include_exceptions: If True, include exception type and message in output
        """
        if not self.errors:
            return ""

        total = len(self.errors) + self._truncated_count
        lines = [f"Encountered {total} issues during indexing:"]

        by_cat = self.by_category()
        for category, errors in sorted(by_cat.items(), key=lambda x: len(x[1]), reverse=True):
            lines.append(f"\n  {category.value} ({len(errors)}):")
            # Show first 3 of each category
            for error in errors[:3]:
                error_line = f"    - {error.path}: {error.message}"
                if include_exceptions and error.exception_type:
                    error_line += f" [{error.exception_type}: {error.exception_message}]"
                lines.append(error_line)
            if len(errors) > 3:
                lines.append(f"    ... and {len(errors) - 3} more")

        if self._truncated_count > 0:
            lines.append(f"\n(Showing {len(self.errors)} of {total} errors; {self._truncated_count} omitted)")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all collected errors."""
        self.errors.clear()
        self._truncated_count = 0
