"""File discovery with .gitignore support."""

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pathspec

from vector_core.settings import settings
from vector_core.utils.hashing import hash_content

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredFile:
    """Information about a discovered file."""

    path: Path  # Absolute path
    relative_path: str  # Relative to root
    size: int  # Size in bytes
    mtime: float  # Modification time (seconds since epoch)
    content_hash: str | None = None  # Lazy computed


@dataclass
class FileMetadata:
    """Lightweight file metadata for fast change detection."""

    relative_path: str
    size: int
    mtime: float
    content_hash: str


def _load_gitignore(root: Path) -> pathspec.PathSpec | None:
    """Load .gitignore patterns from directory."""
    gitignore_path = root / ".gitignore"
    if not gitignore_path.exists():
        return None

    with open(gitignore_path, encoding="utf-8", errors="ignore") as f:
        patterns = f.read().splitlines()

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


class FileDiscovery:
    """
    Discovers files in a directory respecting .gitignore.

    Features:
    - Gitignore-aware file walking
    - Extension filtering
    - Size filtering
    - Content hash computation
    """

    # Directories to always exclude
    ALWAYS_EXCLUDE: set[str] = {
        ".git", ".svn", ".hg", "node_modules", "__pycache__",
        ".venv", "venv", ".tox", ".pytest_cache", ".mypy_cache",
        "dist", "build", ".next", ".nuxt", "target", "coverage",
    }

    def __init__(
        self,
        extensions: set[str] | None = None,
        max_size_kb: int | None = None,
        respect_gitignore: bool = True,
        exclude_patterns: list[str] | None = None,
    ):
        """
        Initialize file discovery.

        Args:
            extensions: File extensions to include (e.g., {".py", ".md"}).
                       If None, includes all files.
            max_size_kb: Max file size in KB. Default from settings.
            respect_gitignore: Whether to honor .gitignore. Default True.
            exclude_patterns: Additional gitignore-style patterns to exclude.
        """
        self.extensions = extensions
        self.max_size_bytes = (max_size_kb or settings.max_file_size_kb) * 1024
        self.respect_gitignore = respect_gitignore
        self.exclude_patterns = exclude_patterns

    def discover(  # noqa: PLR0912
        self,
        root: Path | str,
        include_content: bool = False,
    ) -> Iterator[DiscoveredFile]:
        """
        Discover files in a directory.

        Args:
            root: Root directory to scan
            include_content: Whether to read and hash file content

        Yields:
            DiscoveredFile for each discovered file
        """
        root = Path(root).resolve()

        # Load .gitignore
        gitignore = _load_gitignore(root) if self.respect_gitignore else None

        # Build additional exclude spec
        exclude_spec = None
        if self.exclude_patterns:
            exclude_spec = pathspec.PathSpec.from_lines(
                "gitwildmatch", self.exclude_patterns
            )

        # followlinks=False prevents infinite loops from symlinks
        for dirpath, dirs, files in os.walk(root, followlinks=False):
            current_path = Path(dirpath)
            rel_root = current_path.relative_to(root)

            # Filter directories in-place (skip excluded dirs and symlinks)
            dirs[:] = [
                d for d in dirs
                if d not in self.ALWAYS_EXCLUDE
                and not d.startswith(".")
                and not (current_path / d).is_symlink()
            ]

            for filename in files:
                file_path = current_path / filename
                rel_path = str(rel_root / filename)

                # Skip hidden files and symlinks
                if filename.startswith("."):
                    continue
                if file_path.is_symlink():
                    continue

                # Check extension if filter is set
                if self.extensions and file_path.suffix.lower() not in self.extensions:
                    continue

                # Check gitignore
                if gitignore and gitignore.match_file(rel_path):
                    continue

                # Check exclude patterns
                if exclude_spec and exclude_spec.match_file(rel_path):
                    continue

                # Check file size
                try:
                    stat = file_path.stat()
                    if stat.st_size > self.max_size_bytes:
                        continue
                    if stat.st_size == 0:
                        continue
                except OSError:
                    continue

                # Compute content hash if requested
                content_hash = None
                if include_content:
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        content_hash = hash_content(content)
                    except OSError as e:
                        # File access error (permissions, deleted, etc.)
                        logger.debug(f"File access error for {rel_path}: {e}")
                        continue
                    except UnicodeDecodeError as e:
                        # Encoding issue despite errors="ignore" (rare)
                        logger.debug(f"Encoding error for {rel_path}: {e}")
                        continue

                yield DiscoveredFile(
                    path=file_path,
                    relative_path=rel_path,
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    content_hash=content_hash,
                )

    def scan_metadata(
        self,
        root: Path | str,
    ) -> Iterator[tuple[str, float, int]]:
        """
        Fast scan returning only (rel_path, mtime, size) for each file.

        This is much faster than discover() as it doesn't read file content.
        Use for quick change detection before committing to full file reads.

        Args:
            root: Root directory to scan

        Yields:
            Tuple of (relative_path, mtime, size_bytes)
        """
        root = Path(root).resolve()

        # Load .gitignore
        gitignore = _load_gitignore(root) if self.respect_gitignore else None

        # Build additional exclude spec
        exclude_spec = None
        if self.exclude_patterns:
            exclude_spec = pathspec.PathSpec.from_lines(
                "gitwildmatch", self.exclude_patterns
            )

        for dirpath, dirs, files in os.walk(root, followlinks=False):
            current_path = Path(dirpath)
            rel_root = current_path.relative_to(root)

            # Filter directories
            dirs[:] = [
                d for d in dirs
                if d not in self.ALWAYS_EXCLUDE
                and not d.startswith(".")
                and not (current_path / d).is_symlink()
            ]

            for filename in files:
                file_path = current_path / filename
                rel_path = str(rel_root / filename)

                # Skip hidden files and symlinks
                if filename.startswith("."):
                    continue
                if file_path.is_symlink():
                    continue

                # Check extension if filter is set
                if self.extensions and file_path.suffix.lower() not in self.extensions:
                    continue

                # Check gitignore
                if gitignore and gitignore.match_file(rel_path):
                    continue

                # Check exclude patterns
                if exclude_spec and exclude_spec.match_file(rel_path):
                    continue

                # Get file stats
                try:
                    stat = file_path.stat()
                    if stat.st_size > self.max_size_bytes or stat.st_size == 0:
                        continue
                except OSError:
                    continue

                yield (rel_path, stat.st_mtime, stat.st_size)


def get_file_hash(file_path: Path) -> str | None:
    """Get hash of file content without loading full content into memory."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        return hash_content(content)
    except OSError as e:
        # File access error (permissions, deleted, locked, etc.)
        logger.debug(f"File access error hashing {file_path}: {e}")
        return None
    except UnicodeDecodeError as e:
        # Encoding issue despite errors="ignore" (rare)
        logger.debug(f"Encoding error hashing {file_path}: {e}")
        return None


def read_file_content(file_path: Path) -> tuple[str, str, int] | None:
    """
    Read file and return (content, hash, line_count).

    Returns None if file cannot be read.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        return (content, hash_content(content), content.count("\n") + 1)
    except OSError as e:
        # File access error (permissions, deleted, locked, etc.)
        logger.debug(f"File access error reading {file_path}: {e}")
        return None
    except UnicodeDecodeError as e:
        # Encoding issue despite errors="ignore" (rare)
        logger.debug(f"Encoding error reading {file_path}: {e}")
        return None
