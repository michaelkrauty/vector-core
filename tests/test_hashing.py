"""Tests for utils/hashing module."""

import tempfile
from pathlib import Path

import pytest

from vector_core.utils.hashing import compute_file_hash, hash_content


class TestHashContent:
    """Tests for hash_content function."""

    def test_returns_hex_string(self):
        """Should return a hex string."""
        result = hash_content("test")
        assert isinstance(result, str)
        assert all(c in "0123456789abcdef" for c in result)

    def test_returns_64_chars(self):
        """Should return 64-character SHA256 hex digest."""
        result = hash_content("test")
        assert len(result) == 64

    def test_deterministic(self):
        """Same input should produce same output."""
        content = "Hello, World!"
        assert hash_content(content) == hash_content(content)

    def test_different_inputs_different_hashes(self):
        """Different inputs should produce different hashes."""
        assert hash_content("hello") != hash_content("world")

    def test_empty_string(self):
        """Should handle empty string."""
        result = hash_content("")
        assert len(result) == 64

    def test_unicode_content(self):
        """Should handle unicode content."""
        result = hash_content("日本語テスト")
        assert len(result) == 64

    def test_known_hash(self):
        """Should produce correct SHA256 for known input."""
        # "test" -> SHA256
        expected = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        assert hash_content("test") == expected


class TestComputeFileHash:
    """Tests for compute_file_hash function."""

    def test_returns_hex_string(self):
        """Should return a hex string."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            f.flush()
            result = compute_file_hash(Path(f.name))

        assert isinstance(result, str)
        assert all(c in "0123456789abcdef" for c in result)
        Path(f.name).unlink()

    def test_returns_64_chars_sha256(self):
        """Should return 64-character SHA256 hex digest by default."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            f.flush()
            result = compute_file_hash(Path(f.name))

        assert len(result) == 64
        Path(f.name).unlink()

    def test_deterministic(self):
        """Same file content should produce same hash."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"Hello, World!")
            f.flush()
            path = Path(f.name)
            result1 = compute_file_hash(path)
            result2 = compute_file_hash(path)

        assert result1 == result2
        path.unlink()

    def test_different_content_different_hash(self):
        """Different content should produce different hashes."""
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(b"content1")
            f1.flush()
            path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(b"content2")
            f2.flush()
            path2 = Path(f2.name)

        assert compute_file_hash(path1) != compute_file_hash(path2)
        path1.unlink()
        path2.unlink()

    def test_binary_content(self):
        """Should handle binary content."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(bytes(range(256)))
            f.flush()
            result = compute_file_hash(Path(f.name))

        assert len(result) == 64
        Path(f.name).unlink()

    def test_large_file(self):
        """Should handle large files."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Write 1MB of data
            f.write(b"x" * (1024 * 1024))
            f.flush()
            result = compute_file_hash(Path(f.name))

        assert len(result) == 64
        Path(f.name).unlink()

    def test_empty_file(self):
        """Should handle empty files."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.flush()
            result = compute_file_hash(Path(f.name))

        # SHA256 of empty content
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert result == expected
        Path(f.name).unlink()

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            compute_file_hash(Path("/nonexistent/path/file.txt"))

    def test_custom_algorithm(self):
        """Should support different hash algorithms."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test")
            f.flush()
            path = Path(f.name)
            sha256_result = compute_file_hash(path, algorithm="sha256")
            md5_result = compute_file_hash(path, algorithm="md5")

        assert len(sha256_result) == 64  # SHA256
        assert len(md5_result) == 32  # MD5
        assert sha256_result != md5_result
        path.unlink()

    def test_custom_chunk_size(self):
        """Should work with different chunk sizes."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content " * 1000)
            f.flush()
            path = Path(f.name)
            result1 = compute_file_hash(path, chunk_size=1024)
            result2 = compute_file_hash(path, chunk_size=8192)
            result3 = compute_file_hash(path, chunk_size=128)

        # Same content, same hash regardless of chunk size
        assert result1 == result2 == result3
        path.unlink()
