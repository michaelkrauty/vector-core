"""XDG-compliant path resolution for containers and diverse environments.

This module provides consistent path resolution following the XDG Base Directory
Specification, with fallbacks for containers, restricted environments, and
explicit overrides via environment variables.

Priority order for each path type:
1. Explicit environment variable (VECTOR_*_DIR)
2. XDG environment variable (XDG_*_HOME)
3. Default ~/.cache or ~/.local/share
4. Fallback to /tmp if home not writable

Usage:
    from vector_core.paths import get_cache_dir

    cache_path = get_cache_dir() / "embeddings"

Note: For persistent data (glossary.db, facts.db), use settings.shared_data_dir
instead. It is controlled via the VECTOR_SHARED_DATA_DIR environment variable.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_cache_dir() -> Path:
    """
    Get the cache directory for vector-core (reconstructible data).

    Resolution order:
    1. VECTOR_CACHE_DIR environment variable (explicit override)
    2. XDG_CACHE_HOME/vector-core (XDG specification)
    3. ~/.cache/vector-core (default)
    4. /tmp/vector-core-cache (fallback if home not writable)

    Cache directory is for data that can be regenerated:
    - Embedding cache (sqlite)
    - Search result cache
    - TF-IDF vocabulary snapshots

    Returns:
        Path to cache directory (created if doesn't exist)
    """
    # 1. Explicit override
    if explicit := os.environ.get("VECTOR_CACHE_DIR"):
        path = Path(explicit)
        logger.debug(f"Using explicit cache dir: {path}")
        return _ensure_dir(path)

    # 2. XDG specification
    if xdg := os.environ.get("XDG_CACHE_HOME"):
        path = Path(xdg) / "vector-core"
        logger.debug(f"Using XDG cache dir: {path}")
        return _ensure_dir(path)

    # 3. Default
    default = Path.home() / ".cache" / "vector-core"
    try:
        return _ensure_dir(default)
    except (OSError, PermissionError) as e:
        # 4. Fallback for containers/restricted environments
        fallback = Path("/tmp/vector-core-cache")
        logger.warning(
            f"Cannot use default cache dir {default}: {e}. "
            f"Falling back to {fallback}"
        )
        return _ensure_dir(fallback)



@lru_cache(maxsize=1)
def get_lock_dir() -> Path:
    """
    Get the directory for lock files.

    Uses cache directory subdirectory since locks are transient.

    Returns:
        Path to locks directory
    """
    return _ensure_dir(get_cache_dir() / "locks")


def _ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists and is writable.

    Args:
        path: Directory path to ensure

    Returns:
        The same path (for chaining)

    Raises:
        OSError: If directory cannot be created
        PermissionError: If directory is not writable
    """
    path.mkdir(parents=True, exist_ok=True)

    # Verify writable
    if not os.access(path, os.W_OK):
        raise PermissionError(f"Directory not writable: {path}")

    return path


def clear_path_cache() -> None:
    """
    Clear the cached path values.

    Useful for testing or when environment variables change.
    """
    get_cache_dir.cache_clear()
    get_lock_dir.cache_clear()
