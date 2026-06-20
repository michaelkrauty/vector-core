"""Tests for storage/hash_registry module."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from vector_core.storage.hash_registry import HashRegistry, RegistryEntry


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_registry.db"


@pytest.fixture
def registry(temp_db):
    """Create a HashRegistry instance with temporary database."""
    reg = HashRegistry(db_path=temp_db)
    yield reg
    reg.close()


class TestHashRegistry:
    """Tests for HashRegistry."""

    def test_register_creates_entry(self, registry):
        """Should create entry on register."""
        content_hash = "abc123" * 10 + "abcd"  # 64 chars
        test_uuid = uuid4()
        registry.register(content_hash, test_uuid, "/path/to/file.pdf", "pdf")

        entry = registry.lookup_by_hash(content_hash)
        assert entry is not None
        assert entry.uuid == test_uuid
        assert entry.path == "/path/to/file.pdf"
        assert entry.doc_type == "pdf"
        assert entry.status == "active"

    def test_register_replaces_existing(self, registry):
        """Should replace entry on duplicate hash."""
        content_hash = "abc123" * 10 + "abcd"
        uuid1 = uuid4()
        uuid2 = uuid4()

        registry.register(content_hash, uuid1, "/path1", "pdf")
        registry.register(content_hash, uuid2, "/path2", "docx")

        entry = registry.lookup_by_hash(content_hash)
        assert entry.uuid == uuid2
        assert entry.path == "/path2"
        assert entry.doc_type == "docx"

    def test_lookup_by_hash_not_found(self, registry):
        """Should return None for unknown hash."""
        result = registry.lookup_by_hash("nonexistent" * 6 + "xx")
        assert result is None

    def test_lookup_by_uuid(self, registry):
        """Should find entry by UUID."""
        content_hash = "def456" * 10 + "defg"
        test_uuid = uuid4()
        registry.register(content_hash, test_uuid, "/path/to/file.pdf", "pdf")

        entry = registry.lookup_by_uuid(test_uuid)
        assert entry is not None
        assert entry.content_hash == content_hash
        assert entry.uuid == test_uuid

    def test_lookup_by_uuid_not_found(self, registry):
        """Should return None for unknown UUID."""
        result = registry.lookup_by_uuid(uuid4())
        assert result is None

    def test_update_path(self, registry):
        """Should update path for existing entry."""
        content_hash = "path123" * 10 + "path"
        test_uuid = uuid4()
        registry.register(content_hash, test_uuid, "/old/path.pdf", "pdf")

        result = registry.update_path(content_hash, "/new/path.pdf")
        assert result is True

        entry = registry.lookup_by_hash(content_hash)
        assert entry.path == "/new/path.pdf"

    def test_update_path_not_found(self, registry):
        """Should return False for unknown hash."""
        result = registry.update_path("nonexistent" * 6 + "xx", "/new/path")
        assert result is False

    def test_update_status(self, registry):
        """Should update status for existing entry."""
        content_hash = "status12" * 8
        test_uuid = uuid4()
        registry.register(content_hash, test_uuid, "/path.pdf", "pdf")

        result = registry.update_status(content_hash, "deleted")
        assert result is True

        entry = registry.lookup_by_hash(content_hash)
        assert entry.status == "deleted"

    def test_update_status_not_found(self, registry):
        """Should return False for unknown hash."""
        result = registry.update_status("nonexistent" * 6 + "xx", "deleted")
        assert result is False

    def test_mark_verified(self, registry):
        """Should update last_verified timestamp."""
        content_hash = "verify12" * 8
        test_uuid = uuid4()
        registry.register(content_hash, test_uuid, "/path.pdf", "pdf")

        entry_before = registry.lookup_by_hash(content_hash)
        verified_before = entry_before.last_verified

        result = registry.mark_verified(content_hash)
        assert result is True

        entry_after = registry.lookup_by_hash(content_hash)
        assert entry_after.last_verified is not None
        # Verified timestamp should be updated
        if verified_before:
            assert entry_after.last_verified >= verified_before

    def test_mark_verified_not_found(self, registry):
        """Should return False for unknown hash."""
        result = registry.mark_verified("nonexistent" * 6 + "xx")
        assert result is False

    def test_delete(self, registry):
        """Should delete entry."""
        content_hash = "delete12" * 8
        test_uuid = uuid4()
        registry.register(content_hash, test_uuid, "/path.pdf", "pdf")

        result = registry.delete(content_hash)
        assert result is True

        entry = registry.lookup_by_hash(content_hash)
        assert entry is None

    def test_delete_not_found(self, registry):
        """Should return False for unknown hash."""
        result = registry.delete("nonexistent" * 6 + "xx")
        assert result is False

    def test_list_by_status(self, registry):
        """Should list entries by status."""
        for i in range(5):
            registry.register(f"active{i:02d}" * 6 + "xx", uuid4(), f"/path{i}", "pdf")

        for i in range(3):
            content_hash = f"delete{i:02d}" * 6 + "xx"
            registry.register(content_hash, uuid4(), f"/deleted{i}", "pdf")
            registry.update_status(content_hash, "deleted")

        active_entries = registry.list_by_status("active")
        assert len(active_entries) == 5

        deleted_entries = registry.list_by_status("deleted")
        assert len(deleted_entries) == 3

    def test_list_by_status_with_limit(self, registry):
        """Should respect limit parameter."""
        for i in range(10):
            registry.register(f"limit{i:03d}" * 6 + "xx", uuid4(), f"/path{i}", "pdf")

        entries = registry.list_by_status("active", limit=5)
        assert len(entries) == 5

    def test_list_by_status_limit_zero_returns_no_entries(self, registry):
        """limit=0 means zero rows (SQLite LIMIT 0 semantics), not unlimited."""
        for i in range(3):
            registry.register(f"zero{i:03d}" * 6 + "xxx", uuid4(), f"/path{i}", "pdf")

        assert registry.list_by_status("active", limit=0) == []
        # None still means no limit.
        assert len(registry.list_by_status("active", limit=None)) == 3

    def test_count(self, registry):
        """Should count entries."""
        assert registry.count() == 0

        for i in range(5):
            registry.register(f"count{i:03d}" * 6 + "xx", uuid4(), f"/path{i}", "pdf")

        assert registry.count() == 5

    def test_count_with_status_filter(self, registry):
        """Should count entries by status."""
        for i in range(3):
            registry.register(f"cstat{i:03d}" * 6 + "xx", uuid4(), f"/path{i}", "pdf")

        for i in range(2):
            content_hash = f"cdel{i:04d}" * 6 + "xx"
            registry.register(content_hash, uuid4(), f"/deleted{i}", "pdf")
            registry.update_status(content_hash, "deleted")

        assert registry.count() == 5
        assert registry.count("active") == 3
        assert registry.count("deleted") == 2
        assert registry.count("modified") == 0

    def test_context_manager(self, temp_db):
        """Should support context manager protocol."""
        content_hash = "context1" * 8
        test_uuid = uuid4()

        with HashRegistry(db_path=temp_db) as reg:
            reg.register(content_hash, test_uuid, "/path", "pdf")

        # Verify data persisted
        reg2 = HashRegistry(db_path=temp_db)
        entry = reg2.lookup_by_hash(content_hash)
        assert entry is not None
        reg2.close()

    def test_registry_entry_fields(self, registry):
        """Should return correct RegistryEntry fields."""
        content_hash = "fields12" * 8
        test_uuid = uuid4()
        registry.register(content_hash, test_uuid, "/path/to/file.pdf", "pdf")

        entry = registry.lookup_by_hash(content_hash)

        assert isinstance(entry, RegistryEntry)
        assert entry.content_hash == content_hash
        assert entry.uuid == test_uuid
        assert entry.path == "/path/to/file.pdf"
        assert entry.doc_type == "pdf"
        assert entry.status == "active"
        assert isinstance(entry.registered_at, datetime)
        # last_verified is set on register
        assert isinstance(entry.last_verified, datetime)


class TestHashRegistryThreadSafety:
    """Tests for thread safety of HashRegistry."""

    def test_concurrent_reads(self, registry):
        """Should handle concurrent reads."""
        import concurrent.futures

        # Register some entries
        hashes = []
        for i in range(10):
            h = f"concur{i:03d}" * 6 + "xx"
            registry.register(h, uuid4(), f"/path{i}", "pdf")
            hashes.append(h)

        def read_entry(content_hash):
            return registry.lookup_by_hash(content_hash)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(read_entry, h) for h in hashes * 10]
            results = [f.result() for f in futures]

        assert all(r is not None for r in results)

    def test_concurrent_writes(self, registry):
        """Should handle concurrent writes."""
        import concurrent.futures

        def write_entry(i):
            h = f"write{i:04d}" * 6 + "xx"
            registry.register(h, uuid4(), f"/path{i}", "pdf")
            return h

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(write_entry, i) for i in range(20)]
            hashes = [f.result() for f in futures]

        # Verify all entries exist
        for h in hashes:
            entry = registry.lookup_by_hash(h)
            assert entry is not None
