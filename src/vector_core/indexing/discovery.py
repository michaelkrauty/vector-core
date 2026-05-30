"""File discovery with nested .gitignore support."""

import logging
import os
from collections.abc import Iterable, Iterator
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


def _load_ignore_spec(path: Path) -> pathspec.GitIgnoreSpec | None:
    """Parse a gitignore-syntax file into a spec, or None if absent/unreadable."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return pathspec.GitIgnoreSpec.from_lines(f.read().splitlines())
    except OSError:
        return None


def _load_gitignore(root: Path) -> pathspec.GitIgnoreSpec | None:
    """Load root-level .gitignore patterns (kept for backwards compatibility)."""
    return _load_ignore_spec(root / ".gitignore")


def _is_ignored(
    abs_path: Path,
    rel_to_root: str,
    specs: list[tuple[Path, pathspec.GitIgnoreSpec]],
    exclude_spec: pathspec.GitIgnoreSpec | None,
) -> bool:
    """Decide whether a file is excluded by the accumulated ignore specs.

    Matching happens at file level (preserving historical behavior): the programmatic
    ``exclude_spec`` wins outright, then the deepest matching ignore file decides
    (``!`` negations re-include the file), falling back to shallower files and finally
    ``.git/info/exclude``. Each spec is matched relative to its own base directory, so
    a directory pattern such as ``build/`` still matches files nested beneath it.
    """
    if exclude_spec is not None and exclude_spec.match_file(rel_to_root):
        return True
    # Deepest spec first: the most specific ignore file takes precedence.
    for base, spec in reversed(specs):
        try:
            rel = str(abs_path.relative_to(base))
        except ValueError:
            continue
        result = spec.check_file(rel)
        if result.include is not None:
            # include=True -> matched an ignore pattern; include=False -> negation.
            return result.include
    return False


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
        ignore_filenames: str | Iterable[str] = (".gitignore",),
    ):
        """
        Initialize file discovery.

        Args:
            extensions: File extensions to include (e.g., {".py", ".md"}).
                       If None, includes all files.
            max_size_kb: Max file size in KB. Default from settings.
            respect_gitignore: Whether to honor ignore files. Default True. When
                True, ``ignore_filenames`` (at every directory level) and
                ``.git/info/exclude`` are honored; when False, none are read.
            exclude_patterns: Additional gitignore-style patterns to exclude.
            ignore_filenames: Name(s) of gitignore-syntax ignore files honored at every
                directory level. Accepts a single string or an iterable of names.
                Defaults to ``(".gitignore",)``; consumers may add their own, e.g.
                ``(".gitignore", ".myignore")``.
        """
        self.extensions = extensions
        self.max_size_bytes = (max_size_kb or settings.max_file_size_kb) * 1024
        self.respect_gitignore = respect_gitignore
        self.exclude_patterns = exclude_patterns
        if isinstance(ignore_filenames, str):
            ignore_filenames = (ignore_filenames,)
        self.ignore_filenames = tuple(ignore_filenames)

    def _dir_ignore_specs(
        self, directory: Path
    ) -> list[tuple[Path, pathspec.GitIgnoreSpec]]:
        """Ignore specs defined directly in ``directory`` (e.g. its own .gitignore)."""
        if not self.respect_gitignore:
            return []
        specs: list[tuple[Path, pathspec.GitIgnoreSpec]] = []
        for name in self.ignore_filenames:
            spec = _load_ignore_spec(directory / name)
            if spec is not None:
                specs.append((directory, spec))
        return specs

    def _root_ignore_specs(
        self, root: Path
    ) -> list[tuple[Path, pathspec.GitIgnoreSpec]]:
        """Root-level specs: ``.git/info/exclude`` (lowest precedence), then ignore files.

        The user's global ``core.excludesFile`` is intentionally not consulted, so that
        discovery stays reproducible and independent of per-machine git configuration.
        """
        if not self.respect_gitignore:
            return []
        specs: list[tuple[Path, pathspec.GitIgnoreSpec]] = []
        info_exclude = _load_ignore_spec(root / ".git" / "info" / "exclude")
        if info_exclude is not None:
            specs.append((root, info_exclude))
        specs.extend(self._dir_ignore_specs(root))
        return specs

    def _walk(  # noqa: PLR0912
        self, root: Path
    ) -> Iterator[tuple[Path, str, os.stat_result]]:
        """Walk ``root`` yielding ``(abs_path, rel_path, stat)`` for indexable files.

        Shared by :meth:`discover` and :meth:`scan_metadata` so the two cannot drift
        apart. Directories are pruned only by name (always-excluded dirs, dot
        directories, symlinks); ignore files are matched at file level using specs
        accumulated from the root down to each file, with git's "deeper overrides
        shallower" precedence. Also handles hidden-file and symlink skipping,
        extension filtering, and size limits.
        """
        exclude_spec = (
            pathspec.GitIgnoreSpec.from_lines(self.exclude_patterns)
            if self.exclude_patterns
            else None
        )
        # Accumulated specs per directory, ordered root -> deep (shared by reference
        # when a directory adds none of its own).
        specs_by_dir: dict[Path, list[tuple[Path, pathspec.GitIgnoreSpec]]] = {}

        # followlinks=False prevents infinite loops from symlinks pointing to parents.
        for dirpath, dirs, files in os.walk(root, followlinks=False):
            current_path = Path(dirpath)
            rel_root = current_path.relative_to(root)

            if current_path == root:
                current_specs = self._root_ignore_specs(root)
            else:
                parent_specs = specs_by_dir.get(current_path.parent, [])
                local = self._dir_ignore_specs(current_path)
                current_specs = parent_specs + local if local else parent_specs
            specs_by_dir[current_path] = current_specs

            # Prune directories by name only (always-excluded, dot dirs, symlinks).
            # Ignore files are NOT used to prune directories: matching happens at file
            # level, preserving historical behavior (a deeper "!dir/keep" can still
            # re-include a file under an otherwise-ignored directory).
            dirs[:] = [
                d
                for d in dirs
                if d not in self.ALWAYS_EXCLUDE
                and not d.startswith(".")
                and not (current_path / d).is_symlink()
            ]

            for filename in files:
                if filename.startswith("."):
                    continue
                file_path = current_path / filename
                if file_path.is_symlink():
                    continue
                if self.extensions and file_path.suffix.lower() not in self.extensions:
                    continue
                rel_path = str(rel_root / filename)
                if _is_ignored(file_path, rel_path, current_specs, exclude_spec):
                    continue
                try:
                    stat = file_path.stat()
                except OSError:
                    continue
                if stat.st_size == 0 or stat.st_size > self.max_size_bytes:
                    continue
                yield file_path, rel_path, stat

    def discover(
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
        for file_path, rel_path, stat in self._walk(root):
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
        for _file_path, rel_path, stat in self._walk(root):
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
