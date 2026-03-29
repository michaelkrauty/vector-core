"""Tests for change detection."""

from unittest.mock import MagicMock, patch

from vector_core.indexing.change_detect import ChangeDetector, ChangeSet
from vector_core.indexing.discovery import DiscoveredFile, FileDiscovery


class TestChangeSet:
    """Tests for ChangeSet dataclass."""

    def test_empty_changeset(self):
        """Empty changeset has no changes."""
        cs = ChangeSet(added=[], modified=[], deleted=[])

        assert cs.has_changes is False
        assert cs.total_changes == 0

    def test_has_changes_added(self):
        """Detects added files."""
        mock_file = MagicMock(spec=DiscoveredFile)
        cs = ChangeSet(added=[mock_file], modified=[], deleted=[])

        assert cs.has_changes is True
        assert cs.total_changes == 1

    def test_has_changes_modified(self):
        """Detects modified files."""
        mock_file = MagicMock(spec=DiscoveredFile)
        cs = ChangeSet(added=[], modified=[mock_file], deleted=[])

        assert cs.has_changes is True
        assert cs.total_changes == 1

    def test_has_changes_deleted(self):
        """Detects deleted files."""
        cs = ChangeSet(added=[], modified=[], deleted=["deleted.py"])

        assert cs.has_changes is True
        assert cs.total_changes == 1

    def test_total_changes(self):
        """Counts all change types."""
        mock_file1 = MagicMock(spec=DiscoveredFile)
        mock_file2 = MagicMock(spec=DiscoveredFile)
        cs = ChangeSet(
            added=[mock_file1],
            modified=[mock_file2],
            deleted=["d1.py", "d2.py"],
        )

        assert cs.total_changes == 4


class TestChangeDetectorInit:
    """Tests for ChangeDetector initialization."""

    def test_default_discovery(self):
        """Creates default FileDiscovery."""
        detector = ChangeDetector()

        assert detector.discovery is not None
        assert isinstance(detector.discovery, FileDiscovery)

    def test_custom_discovery(self):
        """Accepts custom FileDiscovery."""
        custom_discovery = FileDiscovery(extensions={".py"})
        detector = ChangeDetector(discovery=custom_discovery)

        assert detector.discovery is custom_discovery


class TestDetectChanges:
    """Tests for detect_changes method."""

    def test_detects_new_files(self, tmp_path):
        """Detects newly added files."""
        (tmp_path / "new.py").write_text("new content")

        detector = ChangeDetector()
        changes = detector.detect_changes(tmp_path, indexed_files={})

        assert len(changes.added) == 1
        assert changes.added[0].relative_path == "new.py"
        assert len(changes.modified) == 0
        assert len(changes.deleted) == 0

    def test_detects_modified_files(self, tmp_path):
        """Detects modified files by hash."""
        file_path = tmp_path / "file.py"
        file_path.write_text("new content")

        detector = ChangeDetector()
        changes = detector.detect_changes(
            tmp_path,
            indexed_files={"file.py": "old_hash_different_from_new"}
        )

        assert len(changes.added) == 0
        assert len(changes.modified) == 1
        assert changes.modified[0].relative_path == "file.py"

    def test_detects_deleted_files(self, tmp_path):
        """Detects deleted files."""
        # Create one file, index shows two
        (tmp_path / "existing.py").write_text("content")

        detector = ChangeDetector()
        changes = detector.detect_changes(
            tmp_path,
            indexed_files={
                "existing.py": "some_hash",
                "deleted.py": "deleted_hash",
            }
        )

        assert "deleted.py" in changes.deleted

    def test_no_changes(self, tmp_path):
        """No changes when file hash matches."""
        file_path = tmp_path / "unchanged.py"
        file_path.write_text("content")

        # Get the actual hash
        from vector_core.indexing.discovery import hash_content
        actual_hash = hash_content("content")

        detector = ChangeDetector()
        changes = detector.detect_changes(
            tmp_path,
            indexed_files={"unchanged.py": actual_hash}
        )

        assert changes.has_changes is False


class TestDetectChangesFast:
    """Tests for fast change detection."""

    def test_fast_detection_new_file(self, tmp_path):
        """Fast detection finds new files."""
        (tmp_path / "new.py").write_text("content")

        detector = ChangeDetector()
        changes = detector.detect_changes_fast(tmp_path, indexed_metadata={})

        assert len(changes.added) == 1
        assert changes.added[0].relative_path == "new.py"

    def test_fast_detection_unchanged_file(self, tmp_path):
        """Fast detection skips unchanged files (same mtime+size)."""
        file_path = tmp_path / "unchanged.py"
        file_path.write_text("content")
        stat = file_path.stat()

        from vector_core.indexing.discovery import hash_content
        actual_hash = hash_content("content")

        detector = ChangeDetector()
        changes = detector.detect_changes_fast(
            tmp_path,
            indexed_metadata={
                "unchanged.py": {
                    "content_hash": actual_hash,
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                }
            }
        )

        assert changes.has_changes is False

    def test_fast_detection_mtime_changed(self, tmp_path):
        """Fast detection re-checks when mtime differs."""
        file_path = tmp_path / "touched.py"
        file_path.write_text("content")
        stat = file_path.stat()

        from vector_core.indexing.discovery import hash_content
        actual_hash = hash_content("content")

        detector = ChangeDetector()
        changes = detector.detect_changes_fast(
            tmp_path,
            indexed_metadata={
                "touched.py": {
                    "content_hash": actual_hash,
                    "mtime": stat.st_mtime - 100,  # Different mtime
                    "size": stat.st_size,
                }
            }
        )

        # File should NOT be in modified since hash is same
        assert len(changes.modified) == 0

    def test_fast_detection_size_changed(self, tmp_path):
        """Fast detection re-checks when size differs."""
        file_path = tmp_path / "resized.py"
        file_path.write_text("new content longer")
        stat = file_path.stat()

        detector = ChangeDetector()
        changes = detector.detect_changes_fast(
            tmp_path,
            indexed_metadata={
                "resized.py": {
                    "content_hash": "old_hash",
                    "mtime": stat.st_mtime,
                    "size": 10,  # Different size
                }
            }
        )

        assert len(changes.modified) == 1

    def test_fast_detection_deleted(self, tmp_path):
        """Fast detection finds deleted files."""
        (tmp_path / "existing.py").write_text("content")

        detector = ChangeDetector()
        changes = detector.detect_changes_fast(
            tmp_path,
            indexed_metadata={
                "existing.py": {"content_hash": "h1", "mtime": 0, "size": 0},
                "deleted.py": {"content_hash": "h2", "mtime": 0, "size": 0},
            }
        )

        assert "deleted.py" in changes.deleted

    def test_legacy_mtime_zero_forces_recheck(self, tmp_path):
        """Legacy data with mtime=0 forces hash check."""
        file_path = tmp_path / "legacy.py"
        file_path.write_text("content")
        stat = file_path.stat()

        from vector_core.indexing.discovery import hash_content
        actual_hash = hash_content("content")

        detector = ChangeDetector()
        changes = detector.detect_changes_fast(
            tmp_path,
            indexed_metadata={
                "legacy.py": {
                    "content_hash": actual_hash,
                    "mtime": 0.0,  # Legacy data
                    "size": stat.st_size,
                }
            }
        )

        # Should not be modified since hash matches
        assert len(changes.modified) == 0

    def test_fast_detection_content_same_but_touched(self, tmp_path):
        """File touched (mtime changed) but content same is not modified."""
        file_path = tmp_path / "touched.py"
        file_path.write_text("content")

        from vector_core.indexing.discovery import hash_content
        actual_hash = hash_content("content")

        detector = ChangeDetector()
        changes = detector.detect_changes_fast(
            tmp_path,
            indexed_metadata={
                "touched.py": {
                    "content_hash": actual_hash,
                    "mtime": 0,  # Force recheck
                    "size": 0,
                }
            }
        )

        # Content same, so not modified
        assert len(changes.modified) == 0


class TestChangeDetectorIntegration:
    """Integration tests for change detection."""

    def test_complex_scenario(self, tmp_path):
        """Test multiple change types together."""
        # Setup initial state
        (tmp_path / "unchanged.py").write_text("unchanged")
        (tmp_path / "modified.py").write_text("modified new")
        (tmp_path / "new.py").write_text("brand new")

        from vector_core.indexing.discovery import hash_content

        detector = ChangeDetector()
        changes = detector.detect_changes(
            tmp_path,
            indexed_files={
                "unchanged.py": hash_content("unchanged"),
                "modified.py": "old_different_hash",
                "deleted.py": "deleted_hash",
            }
        )

        assert len(changes.added) == 1
        assert changes.added[0].relative_path == "new.py"

        assert len(changes.modified) == 1
        assert changes.modified[0].relative_path == "modified.py"

        assert "deleted.py" in changes.deleted

    def test_subdirectory_changes(self, tmp_path):
        """Detects changes in subdirectories."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "new.py").write_text("nested content")

        detector = ChangeDetector()
        changes = detector.detect_changes(tmp_path, indexed_files={})

        assert len(changes.added) == 1
        assert "subdir/new.py" in changes.added[0].relative_path


class TestDetectChangesFastEdgeCases:
    """Edge case tests for fast change detection."""

    def test_legacy_mtime_zero_same_stat(self, tmp_path):
        """Line 117: File with mtime=0 where stat also returns 0.

        This tests the rare case where:
        - indexed_mtime == 0 (legacy data)
        - current file mtime == 0 (matches indexed)
        - sizes match
        This triggers line 117's elif branch for legacy data.
        """
        from unittest.mock import MagicMock

        from vector_core.indexing.discovery import DiscoveredFile

        content = "legacy content"
        file_size = len(content)

        indexed_metadata = {
            "legacy.py": {
                "content_hash": "different_hash",  # Different from actual
                "mtime": 0.0,  # Legacy data
                "size": file_size,  # Same size
            }
        }

        from vector_core.indexing.change_detect import ChangeDetector
        detector = ChangeDetector()

        # Mock scan_metadata to return mtime=0, matching the indexed mtime
        def mock_scan_metadata(root):
            yield ("legacy.py", 0.0, file_size)  # mtime=0, size matches

        # Mock discover to return file with different hash
        mock_discovered = MagicMock(spec=DiscoveredFile)
        mock_discovered.relative_path = "legacy.py"
        mock_discovered.content_hash = "actual_hash_different"

        def mock_discover(root, include_content=False):
            yield mock_discovered

        with patch.object(detector.discovery, 'scan_metadata', mock_scan_metadata):
            with patch.object(detector.discovery, 'discover', mock_discover):
                changes = detector.detect_changes_fast(tmp_path, indexed_metadata=indexed_metadata)

        # Should detect as modified due to hash difference
        assert len(changes.modified) == 1
        assert changes.modified[0].relative_path == "legacy.py"
