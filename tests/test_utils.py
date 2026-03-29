"""Tests for vector-core utility functions."""

from vector_core.storage.qdrant import generate_collection_name, generate_point_id


class TestGenerateCollectionName:
    """Tests for generate_collection_name function."""

    def test_deterministic(self):
        """Same input produces same output."""
        name1 = generate_collection_name("/path/to/project")
        name2 = generate_collection_name("/path/to/project")
        assert name1 == name2

    def test_different_paths_different_names(self):
        """Different paths produce different names."""
        name1 = generate_collection_name("/path/to/project1")
        name2 = generate_collection_name("/path/to/project2")
        assert name1 != name2

    def test_default_prefix(self):
        """Default prefix is 'vc'."""
        name = generate_collection_name("/some/path")
        assert name.startswith("vc_")

    def test_custom_prefix(self):
        """Custom prefix is applied."""
        name = generate_collection_name("/some/path", prefix="custom")
        assert name.startswith("custom_")

    def test_trailing_slash_normalized(self):
        """Trailing slashes are normalized."""
        name1 = generate_collection_name("/path/to/project")
        name2 = generate_collection_name("/path/to/project/")
        name3 = generate_collection_name("/path/to/project//")
        assert name1 == name2 == name3

    def test_hash_length(self):
        """Hash portion is 12 characters."""
        name = generate_collection_name("/some/path", prefix="vc")
        # Format: prefix_hash12
        hash_part = name.split("_")[1]
        assert len(hash_part) == 12

    def test_valid_collection_name_format(self):
        """Generated name is valid for Qdrant."""
        name = generate_collection_name("/path/with spaces/and-dashes")
        # Qdrant collection names should be alphanumeric with underscores
        assert name.replace("_", "").isalnum()


class TestGeneratePointId:
    """Tests for generate_point_id function."""

    def test_deterministic(self):
        """Same input produces same output."""
        id1 = generate_point_id("file:/path/to/file.py")
        id2 = generate_point_id("file:/path/to/file.py")
        assert id1 == id2

    def test_different_keys_different_ids(self):
        """Different keys produce different IDs."""
        id1 = generate_point_id("file:/path/to/file1.py")
        id2 = generate_point_id("file:/path/to/file2.py")
        assert id1 != id2

    def test_returns_uuid_string(self):
        """Returns a 36-character UUID string."""
        point_id = generate_point_id("some_key")
        assert isinstance(point_id, str)
        assert len(point_id) == 36  # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        # Must be valid UUID format
        from uuid import UUID
        UUID(point_id)  # Raises if not valid UUID

    def test_consistent_across_key_formats(self):
        """Same logical key always produces same ID."""
        # Common key formats used in codesearch
        id1 = generate_point_id("file:/path/to/code.py:0")
        id2 = generate_point_id("file:/path/to/code.py:0")
        assert id1 == id2

    def test_chunk_keys_unique(self):
        """Different chunk indices produce different IDs."""
        id1 = generate_point_id("chunk:/path/file.py:0")
        id2 = generate_point_id("chunk:/path/file.py:1")
        id3 = generate_point_id("chunk:/path/file.py:2")
        assert len({id1, id2, id3}) == 3  # All unique

    def test_uuid_format_valid(self):
        """Returns valid UUID4-like format."""
        point_id = generate_point_id("test_key")
        # UUID format with dashes
        assert len(point_id) == 36
        parts = point_id.split("-")
        assert len(parts) == 5
        assert [len(p) for p in parts] == [8, 4, 4, 4, 12]
        # Verify it's lowercase hex
        hex_only = point_id.replace("-", "")
        assert hex_only == hex_only.lower()
        assert all(c in "0123456789abcdef" for c in hex_only)
