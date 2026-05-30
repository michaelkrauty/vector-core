"""Tests for file discovery."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from vector_core.indexing.discovery import (
    DiscoveredFile,
    FileDiscovery,
    FileMetadata,
    _load_gitignore,
    get_file_hash,
    hash_content,
    read_file_content,
)


class TestDiscoveredFile:
    """Tests for DiscoveredFile dataclass."""

    def test_basic_creation(self, tmp_path):
        """Create DiscoveredFile with required fields."""
        discovered = DiscoveredFile(
            path=tmp_path / "test.py",
            relative_path="test.py",
            size=100,
            mtime=1234567890.0,
        )

        assert discovered.path == tmp_path / "test.py"
        assert discovered.relative_path == "test.py"
        assert discovered.size == 100
        assert discovered.mtime == 1234567890.0
        assert discovered.content_hash is None

    def test_with_content_hash(self, tmp_path):
        """Create DiscoveredFile with content hash."""
        discovered = DiscoveredFile(
            path=tmp_path / "test.py",
            relative_path="test.py",
            size=100,
            mtime=1234567890.0,
            content_hash="abc123",
        )

        assert discovered.content_hash == "abc123"


class TestFileMetadata:
    """Tests for FileMetadata dataclass."""

    def test_creation(self):
        """Create FileMetadata with all fields."""
        meta = FileMetadata(
            relative_path="src/main.py",
            size=500,
            mtime=1234567890.0,
            content_hash="hash123",
        )

        assert meta.relative_path == "src/main.py"
        assert meta.size == 500
        assert meta.mtime == 1234567890.0
        assert meta.content_hash == "hash123"


class TestHashContent:
    """Tests for content hashing."""

    def test_deterministic(self):
        """Same content produces same hash."""
        content = "Hello, world!"
        assert hash_content(content) == hash_content(content)

    def test_different_content(self):
        """Different content produces different hashes."""
        assert hash_content("text1") != hash_content("text2")

    def test_sha256_format(self):
        """Hash is valid SHA256 hex string."""
        result = hash_content("test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestLoadGitignore:
    """Tests for gitignore loading."""

    def test_loads_gitignore(self, tmp_path):
        """Loads patterns from .gitignore."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n__pycache__/\n")

        spec = _load_gitignore(tmp_path)

        assert spec is not None
        assert spec.match_file("test.pyc")
        assert spec.match_file("__pycache__/cache.py")

    def test_no_gitignore(self, tmp_path):
        """Returns None if no .gitignore."""
        spec = _load_gitignore(tmp_path)

        assert spec is None


class TestFileDiscoveryInit:
    """Tests for FileDiscovery initialization."""

    def test_default_settings(self, monkeypatch):
        """Uses settings defaults."""
        mock_settings = MagicMock()
        mock_settings.max_file_size_kb = 512

        monkeypatch.setattr("vector_core.indexing.discovery.settings", mock_settings)

        discovery = FileDiscovery()

        assert discovery.extensions is None
        assert discovery.max_size_bytes == 512 * 1024
        assert discovery.respect_gitignore is True

    def test_custom_settings(self):
        """Accepts custom settings."""
        discovery = FileDiscovery(
            extensions={".py", ".js"},
            max_size_kb=256,
            respect_gitignore=False,
            exclude_patterns=["*test*"],
        )

        assert discovery.extensions == {".py", ".js"}
        assert discovery.max_size_bytes == 256 * 1024
        assert discovery.respect_gitignore is False
        assert discovery.exclude_patterns == ["*test*"]


class TestFileDiscoveryDiscover:
    """Tests for file discovery."""

    def test_discovers_files(self, tmp_path):
        """Discovers files in directory."""
        (tmp_path / "file1.py").write_text("print('hello')")
        (tmp_path / "file2.py").write_text("print('world')")

        discovery = FileDiscovery()
        files = list(discovery.discover(tmp_path))

        assert len(files) == 2
        paths = [f.relative_path for f in files]
        assert "file1.py" in paths
        assert "file2.py" in paths

    def test_recursive_discovery(self, tmp_path):
        """Discovers files in subdirectories."""
        (tmp_path / "root.py").write_text("root")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").write_text("nested")

        discovery = FileDiscovery()
        files = list(discovery.discover(tmp_path))

        paths = [f.relative_path for f in files]
        assert "root.py" in paths
        assert "subdir/nested.py" in paths

    def test_extension_filter(self, tmp_path):
        """Filters by extension."""
        (tmp_path / "file.py").write_text("python")
        (tmp_path / "file.js").write_text("javascript")
        (tmp_path / "file.txt").write_text("text")

        discovery = FileDiscovery(extensions={".py", ".txt"})
        files = list(discovery.discover(tmp_path))

        paths = [f.relative_path for f in files]
        assert "file.py" in paths
        assert "file.txt" in paths
        assert "file.js" not in paths

    def test_skips_hidden_files(self, tmp_path):
        """Skips hidden files (starting with .)."""
        (tmp_path / "visible.py").write_text("visible")
        (tmp_path / ".hidden.py").write_text("hidden")

        discovery = FileDiscovery()
        files = list(discovery.discover(tmp_path))

        paths = [f.relative_path for f in files]
        assert "visible.py" in paths
        assert ".hidden.py" not in paths

    def test_skips_excluded_directories(self, tmp_path):
        """Skips always-excluded directories."""
        (tmp_path / "main.py").write_text("main")
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "dep.js").write_text("dep")
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "cache.pyc").write_text("cache")

        discovery = FileDiscovery()
        files = list(discovery.discover(tmp_path))

        paths = [f.relative_path for f in files]
        assert "main.py" in paths
        assert not any("node_modules" in p for p in paths)
        assert not any("__pycache__" in p for p in paths)

    def test_respects_gitignore(self, tmp_path):
        """Respects .gitignore patterns."""
        (tmp_path / ".gitignore").write_text("*.log\nbuild/\n")
        (tmp_path / "main.py").write_text("main")
        (tmp_path / "debug.log").write_text("log")
        build = tmp_path / "build"
        build.mkdir()
        (build / "output.js").write_text("output")

        discovery = FileDiscovery(respect_gitignore=True)
        files = list(discovery.discover(tmp_path))

        paths = [f.relative_path for f in files]
        assert "main.py" in paths
        assert "debug.log" not in paths
        assert not any("build" in p for p in paths)

    def test_ignores_gitignore_when_disabled(self, tmp_path):
        """Ignores .gitignore when disabled."""
        (tmp_path / ".gitignore").write_text("*.log\n")
        (tmp_path / "main.py").write_text("main")
        (tmp_path / "debug.log").write_text("log")

        discovery = FileDiscovery(respect_gitignore=False)
        files = list(discovery.discover(tmp_path))

        paths = [f.relative_path for f in files]
        assert "debug.log" in paths

    def test_exclude_patterns(self, tmp_path):
        """Applies additional exclude patterns."""
        (tmp_path / "main.py").write_text("main")
        (tmp_path / "test_main.py").write_text("test")
        (tmp_path / "main_test.py").write_text("test")

        discovery = FileDiscovery(exclude_patterns=["*test*"])
        files = list(discovery.discover(tmp_path))

        paths = [f.relative_path for f in files]
        assert "main.py" in paths
        assert "test_main.py" not in paths
        assert "main_test.py" not in paths

    def test_size_limit(self, tmp_path):
        """Skips files exceeding size limit."""
        (tmp_path / "small.txt").write_text("small")
        (tmp_path / "large.txt").write_text("x" * 10000)

        discovery = FileDiscovery(max_size_kb=1)  # 1KB limit
        files = list(discovery.discover(tmp_path))

        paths = [f.relative_path for f in files]
        assert "small.txt" in paths
        assert "large.txt" not in paths

    def test_skips_empty_files(self, tmp_path):
        """Skips empty files."""
        (tmp_path / "content.py").write_text("content")
        (tmp_path / "empty.py").write_text("")

        discovery = FileDiscovery()
        files = list(discovery.discover(tmp_path))

        paths = [f.relative_path for f in files]
        assert "content.py" in paths
        assert "empty.py" not in paths

    def test_includes_content_hash(self, tmp_path):
        """Includes content hash when requested."""
        (tmp_path / "file.py").write_text("content")

        discovery = FileDiscovery()
        files = list(discovery.discover(tmp_path, include_content=True))

        assert len(files) == 1
        assert files[0].content_hash is not None
        assert len(files[0].content_hash) == 64

    def test_no_content_hash_by_default(self, tmp_path):
        """No content hash by default."""
        (tmp_path / "file.py").write_text("content")

        discovery = FileDiscovery()
        files = list(discovery.discover(tmp_path, include_content=False))

        assert len(files) == 1
        assert files[0].content_hash is None


class TestScanMetadata:
    """Tests for fast metadata scanning."""

    def test_returns_metadata_tuple(self, tmp_path):
        """Returns (rel_path, mtime, size) tuples."""
        (tmp_path / "file.py").write_text("content")

        discovery = FileDiscovery()
        results = list(discovery.scan_metadata(tmp_path))

        assert len(results) == 1
        rel_path, mtime, size = results[0]
        assert rel_path == "file.py"
        assert mtime > 0
        assert size == 7  # len("content")

    def test_faster_than_discover(self, tmp_path):
        """scan_metadata doesn't read file content."""
        # Create some files
        for i in range(5):
            (tmp_path / f"file{i}.py").write_text(f"content {i}")

        discovery = FileDiscovery()

        # scan_metadata should work without reading content
        results = list(discovery.scan_metadata(tmp_path))
        assert len(results) == 5


class TestGetFileHash:
    """Tests for get_file_hash function."""

    def test_returns_hash(self, tmp_path):
        """Returns hash of file content."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        result = get_file_hash(file_path)

        assert result is not None
        assert result == hash_content("content")

    def test_nonexistent_file(self, tmp_path):
        """Returns None for nonexistent file."""
        result = get_file_hash(tmp_path / "missing.txt")

        assert result is None


class TestReadFileContent:
    """Tests for read_file_content function."""

    def test_returns_content_hash_lines(self, tmp_path):
        """Returns (content, hash, line_count)."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("line1\nline2\nline3")

        result = read_file_content(file_path)

        assert result is not None
        content, file_hash, line_count = result
        assert content == "line1\nline2\nline3"
        assert file_hash == hash_content(content)
        assert line_count == 3

    def test_single_line(self, tmp_path):
        """Handles single line file."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("single line")

        result = read_file_content(file_path)

        assert result is not None
        _, _, line_count = result
        assert line_count == 1

    def test_nonexistent_file(self, tmp_path):
        """Returns None for nonexistent file."""
        result = read_file_content(tmp_path / "missing.txt")

        assert result is None


class TestSymlinkHandling:
    """Tests for symlink handling."""

    def test_skips_symlink_files(self, tmp_path):
        """Skips symlinked files."""
        real_file = tmp_path / "real.py"
        real_file.write_text("content")
        link_file = tmp_path / "link.py"
        link_file.symlink_to(real_file)

        discovery = FileDiscovery()
        files = list(discovery.discover(tmp_path))

        paths = [f.relative_path for f in files]
        assert "real.py" in paths
        assert "link.py" not in paths

    def test_skips_symlink_directories(self, tmp_path):
        """Skips symlinked directories."""
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()
        (real_dir / "file.py").write_text("content")
        link_dir = tmp_path / "link_dir"
        link_dir.symlink_to(real_dir)

        discovery = FileDiscovery()
        files = list(discovery.discover(tmp_path))

        paths = [f.relative_path for f in files]
        assert "real_dir/file.py" in paths
        # Should not have link_dir files (would cause duplication)
        assert not any("link_dir" in p for p in paths)


class TestDiscoverErrorHandling:
    """Tests for error handling in discover()."""

    def test_handles_stat_oserror(self, tmp_path):
        """Skips files that raise OSError during stat (lines 161-162)."""
        (tmp_path / "good.py").write_text("content")
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("content")

        discovery = FileDiscovery()

        # Track stat calls per file to only fail on second call (actual stat, not is_symlink)
        stat_calls = {}
        original_stat = Path.stat

        def mock_stat(self, *args, **kwargs):
            if self.name == "bad.py":
                stat_calls[self.name] = stat_calls.get(self.name, 0) + 1
                # First call is from is_symlink(), fail on second call (actual stat)
                if stat_calls[self.name] > 1:
                    raise OSError("Permission denied")
            return original_stat(self, *args, **kwargs)

        with patch.object(Path, "stat", mock_stat):
            files = list(discovery.discover(tmp_path))

        paths = [f.relative_path for f in files]
        assert "good.py" in paths
        # bad.py should be skipped due to OSError on second stat call

    def test_handles_read_exception(self, tmp_path):
        """Skips files that raise Exception during read (lines 170-171)."""
        (tmp_path / "good.py").write_text("content")
        (tmp_path / "bad.py").write_text("content")

        discovery = FileDiscovery()

        # Mock read_text to fail for specific file
        original_read = Path.read_text

        def mock_read(self, *args, **kwargs):
            if self.name == "bad.py":
                raise PermissionError("Cannot read file")
            return original_read(self, *args, **kwargs)

        with patch.object(Path, "read_text", mock_read):
            # Request content to trigger read
            files = list(discovery.discover(tmp_path, include_content=True))

        paths = [f.relative_path for f in files]
        assert "good.py" in paths
        # bad.py should be skipped due to read error


class TestScanMetadataEdgeCases:
    """Tests for scan_metadata edge cases."""

    def test_skips_hidden_files(self, tmp_path):
        """scan_metadata skips hidden files (line 227)."""
        (tmp_path / "visible.py").write_text("content")
        (tmp_path / ".hidden.py").write_text("hidden")

        discovery = FileDiscovery()
        results = list(discovery.scan_metadata(tmp_path))

        paths = [r[0] for r in results]
        assert "visible.py" in paths
        assert ".hidden.py" not in paths

    def test_skips_symlinks(self, tmp_path):
        """scan_metadata skips symlinked files (line 229)."""
        real_file = tmp_path / "real.py"
        real_file.write_text("content")
        link_file = tmp_path / "link.py"
        link_file.symlink_to(real_file)

        discovery = FileDiscovery()
        results = list(discovery.scan_metadata(tmp_path))

        paths = [r[0] for r in results]
        assert "real.py" in paths
        assert "link.py" not in paths

    def test_extension_filter(self, tmp_path):
        """scan_metadata applies extension filter (line 233)."""
        (tmp_path / "script.py").write_text("python")
        (tmp_path / "script.js").write_text("javascript")
        (tmp_path / "data.json").write_text("{}")

        discovery = FileDiscovery(extensions={".py"})
        results = list(discovery.scan_metadata(tmp_path))

        paths = [r[0] for r in results]
        assert "script.py" in paths
        assert "script.js" not in paths
        assert "data.json" not in paths

    def test_respects_gitignore(self, tmp_path):
        """scan_metadata respects .gitignore (line 237)."""
        (tmp_path / ".gitignore").write_text("*.log\n")
        (tmp_path / "main.py").write_text("main")
        (tmp_path / "debug.log").write_text("log")

        discovery = FileDiscovery(respect_gitignore=True)
        results = list(discovery.scan_metadata(tmp_path))

        paths = [r[0] for r in results]
        assert "main.py" in paths
        assert "debug.log" not in paths

    def test_exclude_patterns(self, tmp_path):
        """scan_metadata applies exclude patterns (lines 205, 241)."""
        (tmp_path / "main.py").write_text("main")
        (tmp_path / "test_main.py").write_text("test")

        discovery = FileDiscovery(exclude_patterns=["test_*"])
        results = list(discovery.scan_metadata(tmp_path))

        paths = [r[0] for r in results]
        assert "main.py" in paths
        assert "test_main.py" not in paths

    def test_handles_stat_oserror(self, tmp_path):
        """scan_metadata skips files with OSError during stat (lines 247-249)."""
        (tmp_path / "good.py").write_text("content")
        (tmp_path / "bad.py").write_text("content")

        discovery = FileDiscovery()

        # Track stat calls per file to only fail on second call (actual stat, not is_symlink)
        stat_calls = {}
        original_stat = Path.stat

        def mock_stat(self, *args, **kwargs):
            if self.name == "bad.py":
                stat_calls[self.name] = stat_calls.get(self.name, 0) + 1
                # First call is from is_symlink(), fail on second call (actual stat)
                if stat_calls[self.name] > 1:
                    raise OSError("Permission denied")
            return original_stat(self, *args, **kwargs)

        with patch.object(Path, "stat", mock_stat):
            results = list(discovery.scan_metadata(tmp_path))

        paths = [r[0] for r in results]
        assert "good.py" in paths
        # bad.py should be skipped

    def test_size_limit(self, tmp_path):
        """scan_metadata applies size limit (line 246)."""
        (tmp_path / "small.txt").write_text("small")
        (tmp_path / "large.txt").write_text("x" * 10000)

        discovery = FileDiscovery(max_size_kb=1)
        results = list(discovery.scan_metadata(tmp_path))

        paths = [r[0] for r in results]
        assert "small.txt" in paths
        assert "large.txt" not in paths


class TestNestedGitignore:
    """Tests for nested .gitignore support (rules accumulate down the tree)."""

    def test_subdir_gitignore_excludes_its_files(self, tmp_path):
        """A .gitignore in a subdirectory excludes matching files there."""
        (tmp_path / "keep.py").write_text("keep")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / ".gitignore").write_text("*.generated\n")
        (sub / "real.py").write_text("real")
        (sub / "thing.generated").write_text("gen")

        paths = {f.relative_path for f in FileDiscovery().discover(tmp_path)}

        assert "keep.py" in paths
        assert "sub/real.py" in paths
        assert "sub/thing.generated" not in paths

    def test_subdir_gitignore_scoped_to_its_subtree(self, tmp_path):
        """A subdir .gitignore must not affect siblings with the same filename."""
        a = tmp_path / "a"
        a.mkdir()
        b = tmp_path / "b"
        b.mkdir()
        (a / ".gitignore").write_text("secret\n")
        (a / "secret").write_text("x")
        (b / "secret").write_text("y")

        paths = {f.relative_path for f in FileDiscovery().discover(tmp_path)}

        assert "a/secret" not in paths
        assert "b/secret" in paths

    def test_nested_gitignore_prunes_directory(self, tmp_path):
        """A directory-only pattern in a nested .gitignore prunes that directory."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / ".gitignore").write_text("generated/\n")
        gen = sub / "generated"
        gen.mkdir()
        (gen / "out.py").write_text("x")
        (sub / "keep.py").write_text("y")

        paths = {f.relative_path for f in FileDiscovery().discover(tmp_path)}

        assert "sub/keep.py" in paths
        assert not any("generated" in p for p in paths)

    def test_respect_gitignore_false_disables_nested(self, tmp_path):
        """respect_gitignore=False ignores every ignore file, nested included."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / ".gitignore").write_text("*.gen\n")
        (sub / "a.gen").write_text("a")

        disc = FileDiscovery(respect_gitignore=False)
        paths = {f.relative_path for f in disc.discover(tmp_path)}

        assert "sub/a.gen" in paths


class TestGitInfoExclude:
    """Tests for repo-local .git/info/exclude support."""

    def test_git_info_exclude_respected(self, tmp_path):
        """Patterns in .git/info/exclude are honored (repo-local, uncommitted)."""
        info = tmp_path / ".git" / "info"
        info.mkdir(parents=True)
        (info / "exclude").write_text("ignored\n")
        (tmp_path / "ignored").write_text("x")
        (tmp_path / "kept.py").write_text("y")

        paths = {f.relative_path for f in FileDiscovery().discover(tmp_path)}

        assert "ignored" not in paths
        assert "kept.py" in paths


class TestIgnorePrecedence:
    """Tests for ignore precedence: negation and deeper-overrides-shallower."""

    def test_negation_reincludes(self, tmp_path):
        """A '!' pattern re-includes a file excluded by an earlier pattern."""
        (tmp_path / ".gitignore").write_text("gen_*\n!gen_keep\n")
        (tmp_path / "gen_a").write_text("x")
        (tmp_path / "gen_keep").write_text("y")

        paths = {f.relative_path for f in FileDiscovery().discover(tmp_path)}

        assert "gen_a" not in paths
        assert "gen_keep" in paths

    def test_deeper_negation_overrides_parent(self, tmp_path):
        """A child .gitignore '!' re-includes a file ignored by a parent pattern."""
        (tmp_path / ".gitignore").write_text("*.tmp\n")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / ".gitignore").write_text("!important.tmp\n")
        (sub / "important.tmp").write_text("keep")
        (sub / "other.tmp").write_text("drop")
        (tmp_path / "root.tmp").write_text("drop")

        paths = {f.relative_path for f in FileDiscovery().discover(tmp_path)}

        assert "root.tmp" not in paths
        assert "sub/other.tmp" not in paths
        assert "sub/important.tmp" in paths

    def test_exclude_patterns_stay_file_level(self, tmp_path):
        """exclude_patterns match at file level only (backwards compatible)."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "keep.py").write_text("k")
        (sub / "drop.py").write_text("d")

        disc = FileDiscovery(exclude_patterns=["sub/", "!sub/keep.py"])
        paths = {f.relative_path for f in disc.discover(tmp_path)}

        assert "sub/keep.py" in paths
        assert "sub/drop.py" not in paths


class TestConfigurableIgnoreFilenames:
    """Tests for the generic ignore_filenames parameter."""

    def test_extra_ignore_file_honored_when_configured(self, tmp_path):
        """A configured extra ignore file is honored; by default it is not."""
        (tmp_path / ".myignore").write_text("vendored.py\n")
        (tmp_path / "vendored.py").write_text("x")
        (tmp_path / "main.py").write_text("y")

        default_paths = {f.relative_path for f in FileDiscovery().discover(tmp_path)}
        extra = FileDiscovery(ignore_filenames=(".gitignore", ".myignore"))
        extra_paths = {f.relative_path for f in extra.discover(tmp_path)}

        assert "vendored.py" in default_paths
        assert "vendored.py" not in extra_paths
        assert "main.py" in extra_paths

    def test_extra_ignore_file_honored_when_nested(self, tmp_path):
        """Extra ignore files are honored at every directory level."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / ".myignore").write_text("*.big\n")
        (sub / "data.big").write_text("x")
        (sub / "code.py").write_text("y")

        disc = FileDiscovery(ignore_filenames=(".gitignore", ".myignore"))
        paths = {f.relative_path for f in disc.discover(tmp_path)}

        assert "sub/data.big" not in paths
        assert "sub/code.py" in paths


class TestNestedScanConsistency:
    """discover() and scan_metadata() must agree under nested ignore rules."""

    def test_scan_matches_discover_with_nested_ignores(self, tmp_path):
        """The full walk and the fast metadata scan select identical files."""
        (tmp_path / ".gitignore").write_text("*.log\n")
        (tmp_path / "main.py").write_text("m")
        (tmp_path / "skip.log").write_text("l")
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / ".gitignore").write_text("vendor_*\n")
        (pkg / "vendor_lib.py").write_text("v")
        (pkg / "app.py").write_text("a")

        disc = FileDiscovery()
        discovered = {f.relative_path for f in disc.discover(tmp_path)}
        scanned = {r[0] for r in disc.scan_metadata(tmp_path)}

        assert discovered == scanned
        assert discovered == {"main.py", "pkg/app.py"}
