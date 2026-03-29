# Architectural Patterns in vector-core

This document describes the key architectural patterns used in vector-core and the MCP server ecosystem.

## Singleton Patterns

### AsyncSingleton (utils/async_helpers.py)

Thread-safe, lazy-initialized singleton for async resources.

```python
from vector_core.utils.async_helpers import AsyncSingleton

_storage = AsyncSingleton("storage")

async def get_storage() -> QdrantStorage:
    return await _storage.get(lambda: QdrantStorage(...))
```

**Features:**
- Double-checked locking for thread safety
- Recovery after transient failures (30s cooldown)
- Explicit close() for cleanup

**Recovery Behavior:**
After initialization failure, the singleton will retry after `recovery_delay` seconds (default 30s). This handles transient issues like network blips during startup.

### SyncSingleton (utils/sync_singleton.py)

Similar pattern for synchronous resources like SQLite stores.

```python
from vector_core.utils.sync_singleton import SyncSingleton

_store = SyncSingleton("glossary_store")

def get_store() -> GlossaryStore:
    return _store.get(lambda: GlossaryStore(...))
```

## Error Handling

### error_response() Pattern

All MCP servers use `error_response()` for consistent error returns:

```python
from vector_core.errors import error_response, ErrorCode, is_error_response

# In tool handlers
if not path.exists():
    return error_response(ErrorCode.FILE_NOT_FOUND, f"Path does not exist: {path}")

# Checking for errors
if is_error_response(result):
    log_error(result)
```

**Standard Error Codes:**
- `ErrorCode.VALIDATION_FAILED` - Invalid input
- `ErrorCode.FILE_NOT_FOUND` - Path doesn't exist
- `ErrorCode.TIMEOUT` - Operation timed out
- `ErrorCode.INTERNAL_ERROR` - Unexpected error

### ErrorCollector (errors.py)

For batch operations that may partially fail:

```python
from vector_core.errors import ErrorCollector, ErrorCategory

collector = ErrorCollector(max_errors=100)

for file in files:
    try:
        process(file)
    except ParseError as e:
        collector.add_parse_error(file, e)
    except IOError as e:
        collector.add_file_access_error(file, e)

if collector.has_errors:
    summary = collector.format_summary()
```

## Cleanup Patterns

### sync_cleanup_wrapper (utils/async_helpers.py)

Convert async cleanup to synchronous for signal handlers:

```python
from vector_core.utils.async_helpers import sync_cleanup_wrapper, AsyncSingleton

_storage = AsyncSingleton("storage")
_cache = AsyncSingleton("cache")

async def cleanup():
    await _storage.close()
    await _cache.close()

def main():
    singletons = [_storage, _cache]
    signal.signal(signal.SIGTERM, lambda *_: sync_cleanup_wrapper(cleanup, singletons))
    # or
    atexit.register(lambda: sync_cleanup_wrapper(cleanup, singletons))
```

### Context Manager Protocol

All resources support context managers:

```python
with GlossaryStore(db_path) as store:
    store.add_entry(...)
# Automatically closed

async with QdrantStorage(...) as storage:
    await storage.upsert(...)
# Automatically closed
```

## Resilience Patterns

### Circuit Breaker (embeddings/client.py)

Prevents cascade failures when external services are down:

```python
class EmbeddingClient:
    # After N consecutive failures, circuit opens for M seconds
    # Defaults: threshold=5, reset=60s (configurable via settings)
    self._circuit_threshold = settings.circuit_breaker_threshold
    self._circuit_reset_time = settings.circuit_breaker_reset_seconds
```

**States:**
- **Closed**: Normal operation
- **Open**: All requests fail fast for `_circuit_reset_time` seconds
- **Half-Open**: After reset time, allow one request to test recovery

### Health Checks

#### SQLite (utils/sqlite.py)
```python
# Connections validated with SELECT 1 before use
conn = self._get_conn()  # Auto-reconnects if unhealthy
```

#### Qdrant (storage/qdrant.py)
```python
# Health check embedded in client accessor
async def _get_client(self):
    # Validates connection, auto-reconnects if unhealthy
    ...
```

### Stale Lock Detection (utils/locking.py)

Lock files older than 1 hour are automatically broken:

```python
from vector_core.utils.locking import file_lock, async_file_lock

# Stale locks auto-detected and removed
with file_lock(path, timeout=10.0):
    process(path)

# Or async
async with async_file_lock("operation_name"):
    await process()
```

## Caching Patterns

### TTLCache (utils/cache.py)

In-memory LRU cache with TTL expiration:

```python
from vector_core.utils.cache import TTLCache, CacheConfig

config = CacheConfig(max_size=1000, ttl_seconds=300)
cache: TTLCache[SearchResult] = TTLCache(config)

# Uses OrderedDict for O(1) LRU eviction
cache.set(key, value)
result = cache.get(key)  # Updates LRU position

# Prefix invalidation after updates
cache.invalidate_prefix(f"{codebase_id}|")
```

### GlobalVocabulary Cache (embeddings/global_vocab.py)

Multi-server safe caching with configurable TTL (default 5s):

```python
# In-memory cache refreshes every 5s for multi-server consistency
vocab = GlobalVocabulary.get_instance()
# Automatically refreshes stale cache from SQLite
```

## SQLite Patterns

### ThreadSafeSQLiteStore (utils/sqlite.py)

Base class for thread-safe SQLite stores:

```python
from vector_core.utils.sqlite import ThreadSafeSQLiteStore, SQLiteConfig

class MyStore(ThreadSafeSQLiteStore):
    def __init__(self, db_path: Path):
        config = SQLiteConfig(busy_timeout_ms=10000)
        super().__init__(db_path, config)
        self._init_db()

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("CREATE TABLE IF NOT EXISTS ...")
        conn.commit()
```

**Features:**
- Per-thread connection management
- Automatic health checks (SELECT 1)
- Auto-reconnect on stale connections
- WAL mode for concurrent access

## Data Models

### Pydantic Models

All data transfer objects use Pydantic for validation:

```python
from pydantic import BaseModel, computed_field

class SearchResult(BaseModel):
    path: str
    score: float
    language: str

    @computed_field
    @property
    def is_high_score(self) -> bool:
        return self.score > 0.8

# Serialization
result.model_dump(mode="json")
```

## Retry Patterns

### async_retry (utils/retry.py)

Exponential backoff with configurable retries:

```python
from vector_core.utils.retry import async_retry

@async_retry(max_attempts=3, base_delay=1.0, max_delay=30.0)
async def embed_text(text: str) -> list[float]:
    return await client.embed(text)
```

## Cross-Process Locking Strategy

Different MCP servers use different locking strategies based on their needs:

### SQLite WAL Mode (mcp-notes, mcp-docs)

For notes and documents, SQLite WAL mode provides sufficient cross-process safety:

```python
# Configured via ThreadSafeSQLiteStore
config = SQLiteConfig(
    journal_mode="WAL",      # Concurrent readers, single writer
    busy_timeout_ms=5000,    # Wait 5s for locks
    synchronous="NORMAL",    # Good balance of safety/performance
)
```

**Why WAL is sufficient:**
- Notes/docs operations are typically fast (<100ms)
- Low contention - single user, single process most of the time
- SQLite handles race conditions automatically
- No need for explicit file locks

### fcntl File Locks (mcp-codesearch)

Code search uses explicit POSIX file locks for expensive operations:

```python
from vector_core.utils.locking import file_lock

# Used for full codebase reindexing (can take minutes)
with file_lock(index_lock_path, timeout=0):  # Non-blocking
    reindex_full_codebase()
```

**Why fcntl locks:**
- Full reindex operations are expensive (minutes, not milliseconds)
- Prevents duplicate work if user triggers multiple reindexes
- Non-blocking check: if locked, skip rather than wait

### Thread Locks vs Process Locks

| Lock Type | Scope | Use Case |
|-----------|-------|----------|
| `threading.Lock` | Single process | In-memory data structures |
| `threading.RLock` | Single process | Recursive locking (e.g., GitManager) |
| `fcntl.flock` | Cross-process | File-based operations |
| SQLite WAL | Cross-process | Database operations |

**Decision**: Don't standardize on one approach. Each mechanism is optimized for its use case.

## Query Preprocessing

### Domain-Specific Parsers

Query preprocessing is intentionally NOT consolidated because each domain has unique needs:

| Server | Parser | Key Features |
|--------|--------|--------------|
| mcp-codesearch | `search/preprocess.py` | function:, class:, scope:, language inference |
| mcp-notes | `search/filters.py` | tag:, category:, after:, before:, links_to: |
| vector-core | `search/preprocess.py` | Generic QueryPreprocessor base class |

### Shared Synonyms (vector-core)

Common synonyms are centralized in vector-core for reuse:

```python
from vector_core.search.preprocess import (
    DEFAULT_SYNONYMS,      # Basic abbreviations (fn, cls, err, etc.)
    CODE_SEARCH_SYNONYMS,  # Extended code terms (async, mutex, etc.)
    get_all_code_synonyms, # Combined DEFAULT + CODE_SEARCH
)

# For code search applications
preprocessor = create_default_preprocessor(include_code_synonyms=True)
```

### When to Use What

- **Use vector-core's QueryPreprocessor**: For new search features needing basic expansion
- **Use domain-specific parsers**: When filter syntax is unique to the domain
- **Import CODE_SEARCH_SYNONYMS**: For code-related search expansion

## Best Practices

1. **Always use singletons** for shared resources (storage, vocab, cache)
2. **Always check for errors** with `is_error_response()` after tool calls
3. **Always use context managers** or explicit `close()` for resources
4. **Prefer Pydantic** over dataclasses for API boundary objects
5. **Use circuit breakers** for external service calls
6. **Add health checks** for long-lived connections
7. **Set appropriate TTLs** for caches in multi-server scenarios
8. **Choose locking strategy** based on operation cost and frequency
9. **Keep domain parsers separate** - consolidation adds complexity without benefit
