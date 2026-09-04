"""Centralized settings for vector-core using pydantic-settings."""

from pathlib import Path
from typing import Self

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorCoreSettings(BaseSettings):
    """Base settings for vector-core. Extend for domain-specific settings."""

    model_config = SettingsConfigDict(env_prefix="VECTOR_")

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    # Collection name override (for unified servers like notes/docs)
    # If set, use this exact name; otherwise generate from path
    # NOTE: This is intended for mcp-notes/mcp-docs sharing a collection.
    # mcp-codesearch uses its own collection_name() function and is unaffected.
    collection_name: str | None = None

    # Embeddings (OpenAI-compatible API)
    embedding_url: str = "http://localhost:8080"
    embedding_model: str = ""
    embedding_dim: int = 0
    embedding_batch_size: int = 8
    embedding_concurrency: int = 2
    embedding_timeout: int = 120
    # Max chars for text before truncation (~32k context -> ~8000 safe at 4 chars/token avg)
    embedding_max_text_chars: int = 8000
    # Persistent reuse is opt-in: unset namespace means no cache is opened.
    embedding_cache_namespace: str | None = None
    # Cross-process backend request capacity. 0 disables global coordination.
    embedding_global_concurrency: int = 0

    # Cache (reconstructible data)
    cache_dir: Path = Path.home() / ".cache" / "vector-core"
    cache_max_size_gb: float = 10.0
    cache_max_entries: int = 100000

    # Shared data directory (persistent data shared by multiple servers)
    # Used for: glossary.db, facts.db
    shared_data_dir: Path = Path.home() / ".local" / "share" / "vector-core"

    # Indexing
    max_file_size_kb: int = 500
    max_payload_content_chars: int = 30000  # Chunk content stored in Qdrant payloads

    # Search - Hybrid RRF weights
    dense_weight: float = 1.0
    sparse_weight: float = 0.8
    rrf_k: int = 60
    rrf_prefetch_limit: int = 50

    # Timeouts (seconds)
    # search_timeout and qdrant_operation_timeout nest: the latter is the
    # transport bound every Qdrant request is subject to, while the former is
    # an asyncio budget for a whole search and must stay under it to be the
    # binding limit.
    search_timeout: int = 30  # Hybrid search operations
    qdrant_operation_timeout: int = 60  # Transport timeout for any Qdrant request
    file_lock_timeout: float = 10.0  # File locking timeout

    # Limits
    scroll_max_results: int = 100000  # Max points returned by scroll_points

    # GlobalVocabulary settings
    # Cache TTL in seconds - lower values improve multi-server consistency
    # Default 5s is good for multi-server scenarios, increase if running single server
    global_vocab_cache_ttl: float = 5.0

    # Content hash display length (for cache keys, logs, UI)
    # Truncation from SHA256 (64 chars) to this length for display
    content_hash_display_length: int = 16

    # Circuit breaker settings for embedding client
    # Opens after this many consecutive failures
    circuit_breaker_threshold: int = 5
    # Keeps circuit open for this many seconds before allowing retry
    circuit_breaker_reset_seconds: float = 60.0

    # --- Validators ---

    @field_validator(
        "embedding_batch_size",
        "embedding_concurrency",
        "embedding_timeout",
        "embedding_max_text_chars",
        "max_file_size_kb",
        "max_payload_content_chars",
        "cache_max_entries",
        "rrf_k",
        "rrf_prefetch_limit",
        "search_timeout",
        "qdrant_operation_timeout",
        "scroll_max_results",
        "content_hash_display_length",
        "circuit_breaker_threshold",
        mode="after",
    )
    @classmethod
    def validate_positive_int(cls, v: int, info) -> int:
        """Validate that integer settings are positive."""
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive, got {v}")
        return v

    @field_validator("embedding_global_concurrency", mode="after")
    @classmethod
    def validate_non_negative_int(cls, v: int) -> int:
        """Validate integer settings where zero explicitly disables a feature."""
        if v < 0:
            raise ValueError(f"embedding_global_concurrency must be non-negative, got {v}")
        return v

    @field_validator(
        "dense_weight",
        "sparse_weight",
        "cache_max_size_gb",
        "file_lock_timeout",
        "global_vocab_cache_ttl",
        "circuit_breaker_reset_seconds",
        mode="after",
    )
    @classmethod
    def validate_non_negative_float(cls, v: float, info) -> float:
        """Validate that float settings are non-negative."""
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative, got {v}")
        return v

    @field_validator("embedding_dim", mode="after")
    @classmethod
    def validate_embedding_dim_reasonable(cls, v: int) -> int:
        """Validate embedding dimension. 0 means auto-detect at runtime."""
        if v == 0:
            return v
        if v < 0:
            raise ValueError(f"embedding_dim must be non-negative, got {v}")
        # Common embedding dimensions from popular models
        common_dims = {64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192}
        if v not in common_dims:
            import warnings

            warnings.warn(
                f"embedding_dim={v} is not a common value. "
                f"Common values: {sorted(common_dims)}. "
                f"Ensure this matches your embedding model's output dimension.",
                UserWarning,
                stacklevel=2,
            )
        return v

    @model_validator(mode="after")
    def validate_weights_not_both_zero(self) -> Self:
        """Ensure at least one search weight is positive."""
        if self.dense_weight == 0 and self.sparse_weight == 0:
            raise ValueError("At least one of dense_weight or sparse_weight must be > 0")
        return self


# Default settings instance (can be overridden)
settings = VectorCoreSettings()


class VectorCoreSettingsMixin:
    """
    Mixin providing automatic delegation to VectorCoreSettings.

    Use this mixin in server-specific settings classes to inherit common
    vector-core properties without duplicating @property methods.

    Example:
        class MyServerSettings(VectorCoreSettingsMixin, BaseSettings):
            model_config = SettingsConfigDict(env_prefix="MYSERVER_")

            # Server-specific settings only
            my_setting: str = "default"

            # Access vector-core settings via attribute access:
            # settings.embedding_url -> delegates to vector_core.settings.embedding_url

    The mixin uses __getattr__ for dynamic delegation, which means:
    - Any attribute not found on the class is looked up in vector-core settings
    - Server-specific settings take precedence (defined on the class itself)
    - IDE autocomplete may be limited; use type stubs if needed
    """

    # Properties to delegate to vector-core settings
    # Subclasses can extend this set if needed
    _delegated_properties: frozenset[str] = frozenset(
        {
            # Qdrant
            "qdrant_url",
            "qdrant_api_key",
            "collection_name",
            # Embeddings
            "embedding_url",
            "embedding_model",
            "embedding_dim",
            "embedding_batch_size",
            "embedding_concurrency",
            "embedding_timeout",
            "embedding_max_text_chars",
            "embedding_cache_namespace",
            "embedding_global_concurrency",
            # Cache
            "cache_dir",
            "cache_max_size_gb",
            "cache_max_entries",
            # Shared data
            "shared_data_dir",
            # Indexing
            "max_file_size_kb",
            "max_payload_content_chars",
            # Search
            "dense_weight",
            "sparse_weight",
            "rrf_k",
            "rrf_prefetch_limit",
            # Timeouts
            "search_timeout",
            "qdrant_operation_timeout",
            "file_lock_timeout",
            # Limits
            "scroll_max_results",
            # GlobalVocabulary
            "global_vocab_cache_ttl",
            # Display
            "content_hash_display_length",
            # Circuit breaker
            "circuit_breaker_threshold",
            "circuit_breaker_reset_seconds",
        }
    )

    def __getattr__(self, name: str):
        """Delegate attribute access to vector-core settings for known properties."""
        if name in self._delegated_properties:
            return getattr(settings, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
