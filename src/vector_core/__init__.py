"""
Vector-core: Shared vector search infrastructure for MCP servers.

Provides:
- Dense embeddings via OpenAI-compatible APIs
- Sparse embeddings via TF-IDF
- Hybrid search with RRF fusion
- Qdrant vector storage
- Persistent embedding cache
- File discovery and change detection
- Shared glossary system
- Hash registry for document tracking
"""

__version__ = "1.4.3"

# Main exports for convenience
from vector_core.embeddings.cache import EmbeddingCache
from vector_core.embeddings.client import (
    CircuitBreakerOpenError,
    EmbeddingClient,
    EmbeddingServiceError,
)
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.embeddings.sparse import SparseVector, SparseVectorizer
from vector_core.facts import (
    FACTS_CODEBASE_ID,
    DuplicateFactError,
    Fact,
    FactError,
    FactIndexer,
    FactNotFoundError,
    FactSource,
    FactStore,
    FactSummary,
    IntegrityCheckResult,
    SourceIntegrityManager,
    SourceStatus,
    SourceType,
    compute_spo_hash,
    generate_fact_text,
)
from vector_core.glossary import (
    GlossaryEntry,
    GlossaryEntrySummary,
    GlossaryError,
    GlossaryIndexer,
    GlossaryNotFoundError,
    GlossaryStore,
    GlossaryToolHelper,
    TermExistsError,
)
from vector_core.indexing.change_detect import ChangeDetector, ChangeSet
from vector_core.indexing.discovery import DiscoveredFile, FileDiscovery, FileMetadata
from vector_core.indexing.points import (
    create_hybrid_point,
    create_hybrid_point_with_key,
    sparse_to_qdrant,
)
from vector_core.search.preprocess import (
    ProcessedQuery,
    QueryPreprocessor,
    create_default_preprocessor,
)
from vector_core.settings import VectorCoreSettings, VectorCoreSettingsMixin, settings
from vector_core.storage import (
    HashRegistry,
    HybridSearcher,
    PayloadSchemaType,
    PointId,
    QdrantConnectionError,
    QdrantStorage,
    RegistryEntry,
    SearchResult,
    generate_collection_name,
    generate_point_id,
)
from vector_core.utils.hashing import compute_file_hash, hash_content
from vector_core.utils.async_helpers import (
    AsyncSingleton,
    SingletonInitError,
    clear_init_locks,
    get_async_init_lock,
    sync_cleanup_wrapper,
)
from vector_core.utils.sync_singleton import SyncSingleton
from vector_core.utils.cache import CacheConfig, CacheStats, TTLCache
from vector_core.utils.locking import (
    LockManager,
    async_file_lock,
    cleanup_stale_locks,
    file_lock,
)
from vector_core.utils.datetime import (
    DEFAULT_DATETIME,
    DEFAULT_TIMESTAMP,
    now_utc,
    parse_iso_datetime,
    parse_payload_timestamps,
)
from vector_core.utils.validation import (
    DEFAULT_MAX_LIMIT,
    DEFAULT_MIN_LIMIT,
    parse_uuid_or_none,
    validate_directory_path,
    validate_file_path,
    validate_limit,
    validate_uuid_string,
)
from vector_core.utils.sentinel import (
    UNSET,
    UnsetType,
    is_set,
)
from vector_core.utils.retry import (
    RetryExhaustedError,
    async_retry,
    retry_operation,
)
from vector_core.errors import (
    CollectionError,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseIntegrityError,
    DatabaseLockError,
    ErrorCategory,
    ErrorCode,
    ErrorCollector,
    ErrorSeverity,
    IndexingError,
    PointOperationError,
    StorageError,
    error_response,
    format_error,
    is_error_response,
)
from vector_core.logging import (
    CorrelationLogAdapter,
    OperationMetrics,
    clear_correlation_id,
    get_correlation_id,
    get_logger,
    log_mcp_request,
    log_mcp_response,
    mcp_logged,
    operation_timer,
    set_correlation_id,
)
from vector_core.paths import (
    clear_path_cache,
    get_cache_dir,
    get_lock_dir,
)
from vector_core.models import (
    BasePayload,
    ChunkPayloadMixin,
    CodeChunkPayload,
    CodeFilePayload,
    NoteChunkPayload,
    NotePayload,
    payload_to_dict,
)
from vector_core.mcp import (
    ToolRegistrationError,
    log_registered_tools,
    verify_tools_registered,
)

__all__ = [
    # Version
    "__version__",
    # Embeddings
    "EmbeddingClient",
    "EmbeddingServiceError",
    "CircuitBreakerOpenError",
    "EmbeddingCache",
    "GlobalVocabulary",
    "SparseVectorizer",
    "SparseVector",
    # Facts
    "FactStore",
    "FactIndexer",
    "FACTS_CODEBASE_ID",
    "generate_fact_text",
    "SourceIntegrityManager",
    "IntegrityCheckResult",
    "Fact",
    "FactSource",
    "FactSummary",
    "SourceType",
    "SourceStatus",
    "compute_spo_hash",
    "FactError",
    "FactNotFoundError",
    "DuplicateFactError",
    # Storage
    "QdrantStorage",
    "QdrantConnectionError",
    "HybridSearcher",
    "SearchResult",
    "generate_collection_name",
    "generate_point_id",
    "PointId",
    "HashRegistry",
    "RegistryEntry",
    "PayloadSchemaType",
    # Glossary
    "GlossaryStore",
    "GlossaryIndexer",
    "GlossaryToolHelper",
    "GlossaryEntry",
    "GlossaryEntrySummary",
    "GlossaryError",
    "GlossaryNotFoundError",
    "TermExistsError",
    # Indexing
    "FileDiscovery",
    "DiscoveredFile",
    "FileMetadata",
    "ChangeDetector",
    "ChangeSet",
    "create_hybrid_point",
    "create_hybrid_point_with_key",
    "sparse_to_qdrant",
    # Utilities
    "hash_content",
    "compute_file_hash",
    "get_async_init_lock",
    "clear_init_locks",
    "sync_cleanup_wrapper",
    "AsyncSingleton",
    "SingletonInitError",
    "SyncSingleton",
    # Cache
    "TTLCache",
    "CacheConfig",
    "CacheStats",
    # Locking
    "file_lock",
    "async_file_lock",
    "cleanup_stale_locks",
    "LockManager",
    "validate_limit",
    "validate_uuid_string",
    "validate_directory_path",
    "validate_file_path",
    "parse_uuid_or_none",
    "DEFAULT_MIN_LIMIT",
    "DEFAULT_MAX_LIMIT",
    # Sentinel
    "UNSET",
    "UnsetType",
    "is_set",
    # Retry
    "async_retry",
    "retry_operation",
    "RetryExhaustedError",
    # Datetime
    "parse_iso_datetime",
    "parse_payload_timestamps",
    "now_utc",
    "DEFAULT_DATETIME",
    "DEFAULT_TIMESTAMP",
    # Search
    "QueryPreprocessor",
    "ProcessedQuery",
    "create_default_preprocessor",
    # Settings
    "VectorCoreSettings",
    "VectorCoreSettingsMixin",
    "settings",
    # Errors
    "ErrorCode",
    "ErrorCollector",
    "ErrorSeverity",
    "ErrorCategory",
    "IndexingError",
    "error_response",
    "format_error",
    "is_error_response",
    # Database/Storage Exceptions
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseIntegrityError",
    "DatabaseLockError",
    "StorageError",
    "CollectionError",
    "PointOperationError",
    # Logging
    "get_logger",
    "get_correlation_id",
    "set_correlation_id",
    "clear_correlation_id",
    "CorrelationLogAdapter",
    "operation_timer",
    "OperationMetrics",
    "mcp_logged",
    "log_mcp_request",
    "log_mcp_response",
    # Paths
    "get_cache_dir",
    "get_lock_dir",
    "clear_path_cache",
    # Models
    "BasePayload",
    "ChunkPayloadMixin",
    "NotePayload",
    "NoteChunkPayload",
    "CodeFilePayload",
    "CodeChunkPayload",
    "payload_to_dict",
    # MCP utilities
    "verify_tools_registered",
    "log_registered_tools",
    "ToolRegistrationError",
]
