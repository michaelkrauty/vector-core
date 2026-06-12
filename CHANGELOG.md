# Changelog

## [1.2.4] - 2026-06-12

### Fixed

- `FactStore.create()` now rejects blank or whitespace-only `subject`, `predicate`, `object_value`, `subject_type`, and `object_type` with a `ValueError` raised before any database access, instead of silently storing them. All five fields feed `spo_hash`-based duplicate detection and the entity adjacency graph, so blank values corrupted both. The store validates but does not normalize — accepted values are stored exactly as given.
- `FactStore.create()` and `update()` now reject `confidence` outside the documented 0.0–1.0 range (including NaN) with a `ValueError`; both bounds remain inclusive.

## [1.2.3] - 2026-06-12

### Fixed

- `GlossaryStore.update()` no longer rejects term renames that only collide with the entry's own rows. The uniqueness check for a new term excluded nothing, so a case-only rename (`"USAF"` → `"Usaf"`) matched the entry's own `term_normalized` and raised `TermExistsError`, as did renaming a term to one of the entry's own current aliases. The check now excludes the entry being updated (mirroring how replacement aliases are validated); collisions with *other* entries' terms and aliases are still rejected.

## [1.2.2] - 2026-06-12

### Fixed

- `GlossaryStore.update()` and `create()` are now atomic: a `TermExistsError` is raised before any row is written, so a failed mutation leaves the store fully unchanged. Previously, `update()` deleted the entry's existing aliases *before* validating the replacements against other entries, and `create()` inserted the entry row before alias insertion could fail — both on a long-lived connection whose pending partial state the next successful operation would silently commit. As a backstop, any unexpected error during the write phase now rolls back the transaction instead of leaving it pending.
- Case-normalized duplicate aliases within a single `create()`/`update()` call (e.g. `["kit", "KIT"]`) now raise `TermExistsError` instead of escaping as a raw `sqlite3.IntegrityError` mid-mutation.

## [1.2.1] - 2026-06-11

### Fixed

- `FactStore` batch reads (`query()`, `get_entity_facts()`, `get_facts_by_source()`, `get_facts_by_source_status()`) now return facts in the order their IDs were selected. The internal batch read used SQL `IN (...)`, which returns rows in arbitrary order, so the `ORDER BY modified DESC` those callers request was silently lost.
- `GlossaryStore.update()` now computes `entry_hash` after alias changes are applied. Previously the hash was computed first, so an alias-only update left a stale hash stored even though aliases are part of the hashed content — defeating hash-based change detection for alias edits.
- `EmbeddingClient.embed_all()` invokes `progress_cb` after each batch completes, with a running count of embedded texts, as the parameter's documentation always implied. Previously the callback fired exactly once, at the very end, making it useless for progress reporting. The final invocation is still `(total, total)`.
- `GlossaryToolHelper.add_entry()` and `update_entry()` reject blank or whitespace-only `term`, `expansion`, `definition`, `domain`, and alias entries with an `INVALID_INPUT` error response instead of storing them, and strip surrounding whitespace from accepted values. Passing `domain=None` to `update_entry()` still clears the domain.

## [1.2.0] - 2026-05-30

### Added

- `QdrantStorage.delete_points(collection, point_ids)` deletes points by their IDs, completing the documented point-operations API (which previously offered only `delete_by_filter` and `delete_collection`). Passing an empty sequence is a no-op and sends no request to Qdrant. This is the operation a consumer needs to remove a known set of points — e.g. pruning the now-orphaned chunks of a re-indexed document whose chunk count shrank.

## [1.1.0] - 2026-05-30

### Added

- `FileDiscovery` now honors **nested ignore files** at every directory level with git's "deeper overrides shallower" precedence (including `!` re-include negations), plus `.git/info/exclude`, instead of only the repository-root `.gitignore`. The user's global `core.excludesFile` is intentionally not consulted, keeping discovery reproducible across machines.
- `FileDiscovery(ignore_filenames=...)` lets callers honor additional gitignore-syntax ignore files (e.g. a project-specific `.myignore`) at every directory level. Defaults to `(".gitignore",)`.

### Changed

- `FileDiscovery.discover()` and `scan_metadata()` now share a single internal walk so they cannot drift apart. Ignore rules (nested ignore files and `exclude_patterns` alike) are applied at file level, preserving the prior behavior where a `!dir/keep` pattern can still re-include a file under an otherwise-ignored directory.

## [1.0.5] - 2026-05-27

### Fixed

- Aligned the runtime `vector_core.__version__` constant with package metadata.

## [1.0.4] - 2026-05-25

### Fixed

- Corrected the README prerequisite from Python 3.12+ to Python 3.11+ so the
  published long description matches package metadata and tested compatibility.

## [1.0.3] - 2026-05-21

### Added

- Added `SyncEmbeddingClient`, a synchronous facade over `EmbeddingClient` that
  keeps a persistent background event loop and closes the async HTTP client on
  that same loop. This gives synchronous consumers a safe alternative to
  calling `asyncio.run()` around every embedding request.

## [1.0.2] - 2026-05-21

### Fixed

- Restored Python 3.11 compatibility by replacing Python 3.12-only generic
  function syntax in `vector_core.utils.sentinel` with a standard `TypeVar`.
- Lowered package metadata, Ruff target, and mypy target from Python 3.12 to
  Python 3.11 after validating the test suite on both Python versions.

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
