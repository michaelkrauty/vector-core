# Changelog

## [1.0.1] - 2026-04-12

### Fixed

- `QdrantStorage.upsert_batch` and all four `HybridSearcher.search` code
  paths wrapped their `asyncio.timeout()` blocks with no `except TimeoutError`
  handler, so bare `TimeoutError()` (empty `__str__`) propagated out of the
  library. Downstream MCP servers using FastMCP surfaced this as
  `"Error executing tool X: "` at the client with no type, message, or
  traceback. Timeouts now raise with an operation-specific message
  (operation name, timeout value, collection, and caller scope) while
  preserving the original chain via `raise ... from e`. (#1, #2)

## [1.0.0] - 2026-03-20

Initial public release.

### Features

- **Embedding client** with circuit breaker, retry logic, and connection health monitoring
- **Embedding cache** (SQLite-backed) to avoid redundant API calls
- **Sparse vectorizer** (TF-IDF) for keyword-based search with configurable vocabulary training
- **Qdrant storage** abstraction with hybrid search (dense + sparse vectors, RRF fusion)
- **File discovery** with `.gitignore`-aware filtering and change detection
- **Query preprocessor** with synonym expansion and scope-aware parsing
- **Glossary store** (SQLite) for term definitions with vector-indexed search
- **Fact store** (SQLite) for structured facts with source tracking and vector search
- **Settings mixin** for consistent configuration across downstream MCP servers
- **Logging** with correlation IDs for request tracing
- **POSIX file locking** for safe multi-process access (Linux/macOS)
