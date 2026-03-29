"""Datetime parsing utilities for consistent handling across MCP servers."""

from datetime import UTC, datetime

# Default timestamp for missing/invalid dates (Unix epoch in ISO format)
DEFAULT_TIMESTAMP = "1970-01-01T00:00:00+00:00"
DEFAULT_DATETIME = datetime(1970, 1, 1, tzinfo=UTC)


def parse_iso_datetime(
    value: str | None,
    default: datetime | None = None,
) -> datetime:
    """
    Parse an ISO 8601 datetime string with robust fallback handling.

    Handles common variations:
    - Standard ISO: "2024-01-15T10:30:00+00:00"
    - Zulu suffix: "2024-01-15T10:30:00Z"
    - Missing timezone: "2024-01-15T10:30:00" (assumes UTC)
    - None/empty: returns default

    Args:
        value: ISO datetime string or None
        default: Fallback datetime if parsing fails. Defaults to DEFAULT_DATETIME.

    Returns:
        Parsed datetime (always timezone-aware, UTC)

    Examples:
        >>> parse_iso_datetime("2024-01-15T10:30:00Z")
        datetime(2024, 1, 15, 10, 30, tzinfo=UTC)

        >>> parse_iso_datetime(None)
        datetime(1970, 1, 1, 0, 0, tzinfo=UTC)

        >>> parse_iso_datetime("invalid", default=datetime.now(UTC))
        <current datetime>
    """
    if default is None:
        default = DEFAULT_DATETIME

    if not value:
        return default

    try:
        # Normalize Zulu suffix to +00:00 for fromisoformat compatibility
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)

        # Ensure timezone-aware (assume UTC if naive)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)

        return dt
    except (ValueError, TypeError):
        return default


def parse_payload_timestamps(
    payload: dict,
    created_key: str = "created",
    modified_key: str = "modified",
) -> tuple[datetime, datetime]:
    """
    Extract created and modified timestamps from a payload dict.

    Common pattern in search results where modified falls back to created.

    Args:
        payload: Dictionary containing timestamp fields
        created_key: Key for created timestamp
        modified_key: Key for modified timestamp

    Returns:
        Tuple of (created, modified) datetimes

    Examples:
        >>> payload = {"created": "2024-01-15T10:00:00Z", "modified": "2024-01-16T12:00:00Z"}
        >>> parse_payload_timestamps(payload)
        (datetime(2024, 1, 15, 10, 0, tzinfo=UTC), datetime(2024, 1, 16, 12, 0, tzinfo=UTC))

        >>> payload = {"created": "2024-01-15T10:00:00Z"}  # No modified
        >>> created, modified = parse_payload_timestamps(payload)
        >>> created == modified
        True
    """
    created = parse_iso_datetime(payload.get(created_key))
    modified_raw = payload.get(modified_key)

    if modified_raw:
        modified = parse_iso_datetime(modified_raw, default=created)
    else:
        modified = created

    return created, modified


def now_utc() -> datetime:
    """Get current UTC datetime (convenience wrapper)."""
    return datetime.now(UTC)
