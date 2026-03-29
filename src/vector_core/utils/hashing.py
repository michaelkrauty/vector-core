"""Hashing utilities for content and files."""

import hashlib
from pathlib import Path


def hash_content(content: str) -> str:
    """
    Generate SHA256 hash of text content.

    Args:
        content: Text content to hash

    Returns:
        Full 64-character hex digest
    """
    return hashlib.sha256(content.encode()).hexdigest()


def compute_file_hash(
    path: Path,
    algorithm: str = "sha256",
    chunk_size: int = 8192,
) -> str:
    """
    Stream-hash binary files.

    Streams file in chunks to handle large files (100MB+) without loading
    into memory.

    Args:
        path: Path to file to hash
        algorithm: Hash algorithm to use (default: sha256)
        chunk_size: Chunk size for streaming reads (default: 8KB)

    Returns:
        Full hex digest of file content

    Raises:
        FileNotFoundError: If file does not exist
        IsADirectoryError: If path is a directory
        PermissionError: If file cannot be read
    """
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
