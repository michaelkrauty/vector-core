"""File discovery, change detection, and indexing utilities."""

from vector_core.indexing.change_detect import ChangeDetector, ChangeSet
from vector_core.indexing.discovery import (
    DiscoveredFile,
    FileDiscovery,
    FileMetadata,
    get_file_hash,
    hash_content,
    read_file_content,
)
from vector_core.indexing.points import (
    create_hybrid_point,
    create_hybrid_point_with_key,
    sparse_to_qdrant,
)

__all__ = [
    "FileDiscovery",
    "DiscoveredFile",
    "FileMetadata",
    "ChangeDetector",
    "ChangeSet",
    "hash_content",
    "get_file_hash",
    "read_file_content",
    # Point creation utilities
    "create_hybrid_point",
    "create_hybrid_point_with_key",
    "sparse_to_qdrant",
]
