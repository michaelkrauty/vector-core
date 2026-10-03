# Changelog

## [1.8.0] - 2026-10-02

### Added

- Opt-in token-ID embedding transport encodes complete role-formatted inputs with the configured local tokenizer and sends OpenAI-compatible arrays of ID arrays. It preserves source strings, lossless split offsets, input validation, response ordering, and retry behavior without adding or removing special tokens beyond the configured tokenizer policy. The server must accept token IDs with the exact same vocabulary mapping.
- Token-ID identities and cache keys include the transport mode and tokenizer implementation, version, and content hash. Text remains the default and retains its existing identity serialization and fingerprints. Serialized request budgets measure actual ID payloads independently of semantic identity.

## [1.7.0] - 2026-10-02

### Changed

- Embedding methods now preserve complete inputs or reject them explicitly. Character, byte, and token limits are deployment settings with no model-family capacity defaults. Oversized inputs raise a typed validation error rather than silently truncating queries or documents. Local token counting has explicit special-token and backend-overhead policies.
- Embedding identity and cache preprocessing are fenced from older truncated vectors. Raw input retention and historical collection generations remain intact.

### Added

- A lossless source-span splitter budgets role and optional contextual prefixes with bounded tokenizer work. Consumers can index every span independently without changing the one-vector-per-input embedding API.
- Serialized HTTP request budgeting splits batches without shortening individual inputs. Transport limits are separate from model context and embedding identity.
- Source-preserving fragment migration and writer helpers retain canonical IDs, full raw payloads, and original sparse vectors, while derived points independently represent later spans. Explicit lineage and coverage metadata support validated regeneration and stale-fragment retirement.
- Optional grouped hybrid search retrieves distinct groups from each enabled branch before rank fusion, preserving entity-level coverage when one source has many matching fragments.
- Glossary fragment results retain the canonical expansion and display definition through winner-only hydration, while matched snippets remain separate and stored children stay small.
- Fragment group replacement durably journals the prior group and compensates partial writes, stale-child cleanup failures, and cancellation. Failed rollback retains exact recovery inputs and raises an explicit recovery-required error. This does not imply cross-request atomicity for concurrent readers or process crashes.
- Recovery journal serialization and filesystem synchronization run off the event loop, with cancellation settling workers and cleaning unused journals before releasing writer locks.
- Byte-only embedding limits bound source slicing before UTF-8 encoding, keeping splitter work linear for large retained inputs.

### Upgrade notes

Configure the actual embedding deployment's capacity and matching tokenizer explicitly. Stop older writers before migrating to the fragment-aware representation, and upgrade every writer sharing a logical collection. Migrations retain old generations and publish only complete candidates. Query inputs exceeding configured limits now fail explicitly; applications must choose any multi-query policy themselves. Source reconstruction must exclude derived fragment records, while search should include them.

## [1.6.1] - 2026-10-01

### Fixed

- Dense collection migration no longer duplicates retained document content into an additional embedding-input payload field. When the exact input already exists in `content`, a validated `embedding_text_field="content"` reference preserves source-independent reconstruction without doubling large payloads. Existing explicit embedding inputs retain precedence and are preserved unchanged.
- Migration upserts are bounded by their serialized UTF-8 REST body size instead of the scroll page's point count. Requests use a conservative 30 MiB budget and retain server-confirmed strong ordering. An oversized individual point fails with its ID and measured size, preserving the original collection and leaving the candidate unselected.
- Migration errors now retain exception types and wrapped causes when an upstream transport exception has an empty message.

## [1.6.0] - 2026-10-01

### Added

- Embedding identities cover the configured model, deployment namespace, endpoint, resolved vector dimension, and preprocessing. Configuration changes can automatically rebuild dense indices into retained physical collection generations. An atomic discovery pointer selects a replacement only after its complete build; vector operations always use an immutable physical target. Unknown legacy identity requires rebuilding rather than backfilling an unverifiable model stamp.
- Migration preserves original collections, point IDs, payloads, and sparse vectors. Interrupted candidates remain unselected and retries start clean. Returning to an earlier model rebuilds from the current generation rather than reusing its stale historical index. Coordinated operation locks prevent upgraded local writers from racing migration, and superseded clients cannot write to a different embedding space.
- Glossary and fact indexers participate in automatic generation resolution and retain their full embedding input. Glossary mutations prepare the generation before changing SQLite source records. Legacy document, fact, and note-chunk vectors rebuild from retained text; legacy note summaries use a canonical metadata summary with explicit provenance. Legacy glossary definitions recover from verified SQLite records when retained payloads were truncated.

### Changed

- Embedding methods distinguish document and query roles, with document as the backward-compatible default. Formatting profiles and optional local-tokenizer limits are included in embedding identity and cache separation. Install the `tokenizer` extra to use a local tokenizer JSON without model downloads.
- Collection metadata updates preserve generation identity and readiness markers. Dense-only migrations do not register duplicate sparse-vocabulary contributions.

### Upgrade notes

Stop legacy writers before an initial migration: older binaries do not participate in the new locking protocol. Cooperating processes must share the migration lock directory. Keep deployment namespaces immutable for a given model artifact and serving behavior; the embeddings protocol cannot independently attest model weights. Retained generations require additional storage and are not automatically deleted. Their payloads and vectors remain available for source-independent recovery; returning the configuration to an older model normally rebuilds current content instead of restoring an outdated corpus.

## [1.5.0] - 2026-09-04

### Added

- **`EmbeddingClient.embed_all()` can now reuse embeddings through an opt-in persistent cache.** Set `VECTOR_EMBEDDING_CACHE_NAMESPACE` to an immutable model-artifact/deployment fingerprint to enable it, and change it whenever weights or serving behavior change. Cache keys include that fingerprint, API endpoint, requested model, output dimension, cache and preprocessing versions, and the hash of the effective post-truncation text. Reads and writes are bulk transactions, vectors use compact binary float32 storage, duplicate inputs are embedded once and scattered back in exact input order, and corrupt or incompatible values are discarded. Cache initialization and I/O fail open to ordinary embedding.
- **Embedding HTTP attempts can be limited across processes sharing one backend.** `VECTOR_EMBEDDING_GLOBAL_CONCURRENCY` configures the endpoint/model-scoped capacity, with `0` retaining the previous per-process-only behavior. POSIX `flock` slot files retain stable inodes and are released on cancellation, process exit, or fork inheritance. A persistent capacity manifest rejects mixed nonzero configurations for the same scope. Slots cover each actual HTTP attempt but not cache hits or retry sleeps.
- **Sparse vectors can be mapped back to their current vocabulary tokens.** `GlobalVocabulary.get_tokens_by_indices()` returns every requested append-only index and raises on an unknown index rather than returning a partial mapping. `register_codebase_frequencies()` atomically replaces a large contribution from pre-aggregated frequencies, `get_codebase_ids()` includes contribution-only registrations, and `rebuild_aggregate_doc_frequencies()` restores each global frequency from the sum of recorded contributions without changing append-only token IDs. The vocabulary constructor accepts an optional SQLite synchronous mode for consumers with an external durability protocol. `file_lock()` and `async_file_lock()` now support optional POSIX shared/read locking through `shared=True`; both lock modes retain parent ownership when a forked child unwinds an inherited context.

### Changed

- Embedding responses are validated for exact result count, unique and complete indices, configured dimension, and finite numeric values before they can be returned or cached. A configured dimension of `0` is inferred from the first valid response.
- Collection metadata upserts request server-confirmed completion with strong ordering, preventing an older delayed metadata write from overtaking a newer model/deployment marker.

## [1.4.3] - 2026-09-01

### Fixed

- **File locks now retain a stable inode for the lifetime of their lock namespace.** The synchronous and asynchronous context managers unlinked the lock pathname after releasing it, so a waiter that had already opened the old inode could later acquire it while a new arrival acquired a replacement inode. Both then entered the protected section concurrently. Normal release, age-based stale cleanup, and forced cleanup now leave the empty lock file in place; `flock` itself is released automatically when descriptors close, including after process death. The cleanup APIs remain callable for compatibility and return zero.

## [1.4.2] - 2026-08-20

### Fixed

- **MCP tool registration verification now supports the stable MCP Python SDK v2 API.** The development dependency allowed any SDK version but the tests and type annotations imported `mcp.server.fastmcp.FastMCP`, which SDK v2 removed in favor of `mcp.server.MCPServer`. A clean `uv sync` therefore resolved SDK v2 and failed during test collection before exercising the library. The MCP dependency groups now require the v2 line, and the helper and its real-server tests use `MCPServer` while retaining the defensive registry fallbacks for layout changes.

## [1.4.1] - 2026-07-25

### Changed

- **Development dependencies moved from an extra to a PEP 735 dependency group.** `uv sync` installs a dependency group by default but skips an extra unless it is named, so the project environment was left without pytest and `uv run pytest` silently fell through to whatever pytest was on `PATH`. That interpreter brought its own installed `vector-core`, meaning the suite could report on a copy of the library other than the checkout under test. Running `uv sync` followed by `uv run pytest` now tests this working tree.

  Contributors installing with pip need `pip install -e . --group dev` (pip 25.1+) instead of `pip install -e ".[dev]"`; the README covers both. Deployments that want a lean environment can pass `uv sync --no-default-groups`.

## [1.4.0] - 2026-07-25

### Fixed

- **Every Qdrant request now carries the configured `qdrant_operation_timeout` instead of silently falling back to qdrant-client's five second default.** `QdrantStorage` constructed its client without a `timeout`, so the transport abandoned any request that took longer than five seconds regardless of how the setting was configured. The cost fell on exactly the operations the setting was introduced to cover, the ones whose duration scales with the size of the working set: a delete filtered on many payload values, or a bulk upsert of a large batch, exceeded five seconds and raised `ResponseHandlingException(ReadTimeout(''))`. That exception's message is empty, so a caller reported a failure that named neither a timeout nor a duration, and the setting appeared to be in force the whole time because it was declared, documented as covering bulk operations, and validated — only never read on the path that builds the client.
- **The `asyncio.timeout(qdrant_operation_timeout)` guard around `upsert_batch` is reachable again.** The guard budgeted sixty seconds for a batch while every attempt inside it was capped at the transport's five, so a Qdrant that was healthy but merely slow failed each attempt, exhausted its retry backoff, and surfaced the transport error without the budget ever binding. Both now derive from one value, so the guard bounds the operation as its message has always claimed.

### Changed

- `QdrantStorage` accepts an explicit `timeout` argument, defaulting to `qdrant_operation_timeout`. The initial-connect and reconnect paths now build the client through a single helper, so a timeout can no longer be applied to one path and forgotten on the other — the asymmetry that produced this defect.

### Upgrade notes

A request that previously failed after five seconds now has until `qdrant_operation_timeout` (default 60) to finish, so a saturated or slow Qdrant presents as a slower call rather than an immediate error with an empty message. The tighter bounds above it are unchanged and still bind first: `check_health` keeps its own five second `asyncio` limit, and hybrid search stays bounded by `search_timeout` (default 30). A deployment that relied on the old five second transport cap as a de facto liveness signal should set `VECTOR_QDRANT_OPERATION_TIMEOUT` explicitly.

## [1.3.1] - 2026-07-24

### Fixed

- **`GlobalVocabulary.update_codebase_incremental()` now establishes a codebase's document count instead of silently skipping it.** The count was moved with a bare `UPDATE`, which matches nothing for a codebase that has never called `register_codebase`. Such a codebase's tokens entered the vocabulary and its per-token contributions were recorded, but its document count stayed absent and read as zero forever. That is not a local inaccuracy: `total_docs` is the sum of every codebase's count and feeds the IDF weights used to rank results for all of them, so a corpus contributing document frequencies while reporting no documents inflates every term's apparent rarity across the whole database. Any consumer that maintains its contribution purely incrementally, rather than seeding it with a full registration first, was affected.
- **An incremental removal can no longer take back more than the codebase contributed.** `update_codebase_incremental` applied the caller's figures to the global `doc_freq` and to the per-codebase contribution independently, so removing a shared token more often than it was added consumed another codebase's share of it, leaving the aggregate below the sum of the contributions. A token now settles at `max(existing + added - removed, 0)` and one delta drives both the contribution and the global counter, so the two stay in step and a call that both adds and removes the same token settles on the same answer. Only tokens being removed have their current contribution read, so an addition-only call reads nothing, which matters because this runs inside the writer transaction.
- **Document counts and document frequencies can no longer go negative.** Query weighting is `log((total_docs + 1) / (doc_freq + 1)) + 1`. A value of -1 divides by zero and anything below it takes the logarithm of a negative number, so a single consumer's accounting drift would raise out of `vectorize_query` and take searching down for every codebase sharing the database, not only the one that drifted. Both write paths, `update_codebase_incremental` and the removal half of `register_codebase` and `unregister_codebase`, now floor at zero as a backstop behind the cap above. Neither quantity counts anything that can be negative, so the floor asserts a domain invariant rather than hiding a value that was meaningful.
- **Unregistering a codebase removes its document count even when it holds no token contributions.** `_remove_codebase_contribution` returned early when there was nothing to subtract, so a codebase whose count row existed without contributions stayed visible through `get_codebase_ids()` and was counted by `stats()`.

### Upgrade notes

An existing database cannot recover a document count the old bare `UPDATE` already discarded, because it was never written. The first incremental call after upgrading establishes the row from that call's delta alone, so an affected codebase starts counting from there rather than from its true corpus size. Consumers wanting an accurate count should re-register their corpus once with `register_codebase`, which every full or forced re-index already does.

Where a count or frequency had already been driven negative, the floor stops it getting worse but does not reconstruct the lost quantity, and the aggregate `vocabulary.doc_freq` can sit above the sum of the per-codebase contributions until the codebases involved re-register. The result is a bounded ranking inaccuracy in place of an exception on every search.

## [1.3.0] - 2026-07-10

### Fixed

- **MCP tool registration verification now discovers tools through the public `ToolManager.list_tools()` API.** `verify_tools_registered()` and `log_registered_tools()` previously probed obsolete private attributes, so current FastMCP releases always skipped verification and reported no registered tools. Both functions now share one discovery path with tolerant fallbacks for older SDK layouts, while preserving warnings when the registry cannot be inspected.

## [1.2.12] - 2026-07-10

### Fixed

- **`FactStore.find_connections()` no longer returns duplicate paths for a self-referential fact.** A fact whose subject and object are the same entity creates separate subject and object adjacency rows. The breadth-first traversal previously processed both rows and returned the same fact path twice when that entity was also the target. Each fact ID is now traversed once per entity node, while distinct facts between the same entities still produce distinct paths. (#30)

## [1.2.11] - 2026-06-20

### Fixed

- **`FactIndexer.index_fact` no longer deletes the existing point before re-indexing, so a failed re-index can no longer drop a fact from search.** Each fact maps to a single point with a stable id, so the upsert overwrites any existing point in place; the preceding delete was redundant and meant that if the embedding or upsert failed after it, the fact disappeared from search until a full reindex. Re-indexing a fact (for example on an update) is now atomic: a transient failure leaves the previous point intact.

## [1.2.10] - 2026-06-20

### Fixed

- **`find_connections(target_entity=None)` no longer duplicates an entity reachable through several facts at the maximum depth.** In all-reachable mode the BFS only marked an entity visited when it was below the maximum depth (so it could be queued for further traversal). An entity reached at exactly the maximum depth was therefore never marked, so when two or more distinct facts connected to it, it was emitted once per connecting fact, duplicating it and crowding out other distinct reachable entities once the result limit was hit. Reachable entities are now deduplicated through a separate set so each is returned once regardless of depth, without suppressing the below-max-depth traversal that the visited set still governs.

## [1.2.9] - 2026-06-20

### Fixed

- **Fuzzy query-token matching no longer drops the genuinely closest vocabulary term on a large vocabulary.** `_find_fuzzy_match` (used by `vectorize_query`, which has fuzzy matching on by default, so this is on the live query path) filtered the vocabulary to tokens within two characters of the query token's length, then kept the first `max_candidates` of them in arbitrary dictionary order before scoring. On a large vocabulary the closest match could sit past that cut and never be scored, so an unknown query token (a typo, or a rare or new identifier) could match a worse term or none at all. The candidates are now ordered by how close their length is to the target before the cap, since the length difference is a lower bound on edit distance, so the best Levenshtein candidates are the ones scored.
- **`FactStore.update_source_status` (and `SourceIntegrityManager.revalidate_sources`) no longer rewrites the status of every source when called with no selector.** With `source_type`, `source_id`, and `content_hash` all `None`, the method issued a `WHERE`-less `UPDATE fact_sources SET status = ...`, resetting every source in the table. It now returns 0 and makes no change for an unscoped call; a caller must pass at least one selector. The exposed mcp-notes tool already guards this, so this is library-level defense in depth.
- **The query field-prefix parser now handles a negated prefix such as `-path:` instead of mis-parsing it.** `extract_fields` anchored each prefix with `\b`, which can never match before a leading `-`, and tried prefixes in declaration order, so `-path:tests` was captured as the positive field `path` with a stray `-` left in the query. Prefixes are now matched longest-first and anchored on a start-of-string or whitespace boundary, so `-path:` is extracted correctly and is not shadowed by `path:`. This parser is not currently read by the bundled consumers, so the fix is latent there, but the parser is now correct.

## [1.2.8] - 2026-06-19

### Fixed

- **`SparseVectorizer.extend_vocab` now recomputes IDF for the whole vocabulary, not just the tokens in the new batch.** Each `extend_vocab` call grows the corpus size, which changes the IDF of every term, but the recompute loop only touched terms that appeared in the new documents. Existing terms absent from the batch kept an IDF computed against the smaller corpus, so they were progressively under-weighted relative to the freshly added terms, and the skew compounded with each incremental call. The recompute now covers the full vocabulary, matching `fit()`, so the method behaves as its docstring states.
- **`GlossaryStore.list_all` and `HashRegistry.list_by_status` now treat `limit=0` as "return nothing" rather than "return everything".** Both gated the SQL `LIMIT` clause on a truthiness check (`if limit:`), so `limit=0` dropped the clause and returned the full table, the opposite of SQLite's `LIMIT 0` and inconsistent with `FactStore`. The guard is now `if limit is not None:`, so `0` returns no rows and `None` still means unlimited. The shipped MCP tools route limits through `validate_limit`, which maps `0` to a default, so this is a library-contract fix with no tool-level behavior change.

### Documentation

- Corrected the `generate_point_id` docstring, which described the output as a "version 5" UUID and showed an example the function cannot produce. It emits a deterministic version-4 UUID; the docstring and example now reflect that.
- Corrected the `QueryPreprocessor.expand_camelcase` docstring example, which showed mixed-case expanded terms while the implementation lowercases them.

## [1.2.7] - 2026-06-14

### Fixed

- `FactStore.query()` and `list_summaries()` now match the `subject_type`/`object_type` filters case-insensitively, consistent with their own subject/predicate/object filters (already `LOWER()`-compared) and with the graph methods (`get_entity_facts`/`find_connections`, normalized via the lowercased adjacency table). The facts table stores types case-preserving, so a fact created with `subject_type="Person"` was findable via `get_entity_facts(entity_type="person")` but not `query(subject_type="person")` — user-reachable since the consumer tools pass types through unmodified. This completes the type-filter case-insensitivity begun in v1.2.5 (#18, which fixed `find_connections`).
- `FactStore.create()` and `update()` now reject an inverted validity interval (`valid_from` after `valid_to`) with a `ValueError`, raised before any row is written. An inverted range is never meaningful and silently made the fact unmatchable by any `valid_at` query (no date satisfies both bounds). `update()` validates the effective range — the new value where provided, otherwise the fact's existing one — so partially updating one bound into an inverted state is also rejected.

## [1.2.6] - 2026-06-13

### Fixed

- `FactIndexer.index_all()` and `_train_vocabulary()` now index the complete fact corpus instead of only the 50 most-recently-modified facts. Both called `FactStore.list_summaries()`, whose `limit` defaults to 50, so on a store with more than 50 facts "index all facts" silently left every older fact out of Qdrant — invisible to semantic fact search — and trained the sparse vocabulary on only those 50. They now iterate `FactStore.iter_all()`.
- Incremental `FactIndexer.index_all(force=False)` no longer corrupts the facts sparse-search vocabulary. It registered the `GlobalVocabulary` contribution from only the newly-indexed facts, but `register_codebase()` replaces the codebase's entire contribution, so each incremental run dropped the facts document count to the size of that run and skewed every IDF weight. Vocabulary tokens are now collected from all facts (only the new ones are still upserted), matching `NoteIndexer.index_all`. A reindex (`force=True`) fully heals a vocabulary already corrupted by prior incremental runs.

## [1.2.5] - 2026-06-12

### Fixed

- `FactStore.find_connections()` type filters (`source_type`, `target_type`) are now case-insensitive, matching how the entity adjacency table stores types (lowercased at write time). Previously, passing a type exactly as facts display it (e.g. `"Person"`) silently returned no paths; `get_entity_facts()` and `get_neighbors()` already normalized their type filters. (#18)
- `QdrantStorage.get_metadata()` no longer JSON-deserializes string values that parse as non-dict JSON. `store_metadata()` only serializes dict values, so a stored string like `"123"` or `"true"` round-tripped asymmetrically as `int`/`bool`. Only dict-shaped strings are deserialized now; all other strings come back unchanged.

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
