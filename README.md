# vector-core

Shared vector search infrastructure for MCP servers. Provides dense and sparse embeddings, hybrid search with Reciprocal Rank Fusion, Qdrant vector storage, and supporting utilities (caching, file discovery, change detection, glossary, facts) as a reusable Python library.

## Features

- **Dense embeddings** via any OpenAI-compatible API (llama.cpp, vLLM, Ollama, OpenAI, etc.)
- **Sparse embeddings** via TF-IDF with a shared global vocabulary
- **Hybrid search** combining dense + sparse results with RRF (Reciprocal Rank Fusion)
- **Qdrant vector storage** with health checks and automatic reconnection
- **Persistent SQLite-backed embedding cache** to avoid redundant API calls
- **Cross-process embedding request limits** to protect a shared backend
- **Recoverable sparse indices** through the append-only global vocabulary
- **File discovery and change detection** with nested `.gitignore` / `.git/info/exclude`-aware path filtering (plus configurable extra ignore files)
- **Glossary subsystem** -- shared term definitions stored in SQLite and indexed in Qdrant
- **Facts subsystem** -- knowledge graph storage with subject-predicate-object triples and source integrity tracking
- **Circuit breaker** on the embedding client to fail fast when the upstream API is down
- **Query preprocessing** with synonym expansion (generic and code-specific)
- **Structured error handling** with error codes, collectors, and consistent response formatting
- **Pydantic-based configuration** via environment variables with validation

## Prerequisites

- **Python 3.11+**
- **Linux or macOS** (uses POSIX `fcntl` for file locking; not compatible with Windows)
- **Qdrant** running on `localhost:6333` (or configured via `VECTOR_QDRANT_URL`)
- **An OpenAI-compatible embedding API** (e.g., llama.cpp `/v1/embeddings`, vLLM, Ollama, OpenAI)

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/michaelkrauty/vector-core.git
```

Or clone and install in editable mode for development:

```bash
git clone https://github.com/michaelkrauty/vector-core.git
cd vector-core
uv sync
```

`uv sync` installs the `dev` dependency group, so `uv run pytest` runs the
suite against this checkout. With pip, development tooling lives in a
[PEP 735](https://peps.python.org/pep-0735/) dependency group rather than an
extra:

```bash
pip install -e . --group dev   # pip 25.1+
```

For use as a local dependency in another project (e.g., with uv):

```toml
[tool.uv.sources]
vector-core = { path = "../vector-core", editable = true }
```

## Configuration

All settings are configured via environment variables prefixed with `VECTOR_`. Managed by [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).

### Qdrant

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `VECTOR_QDRANT_API_KEY` | `None` | Qdrant API key (optional, for Qdrant Cloud) |
| `VECTOR_COLLECTION_NAME` | `None` | Override collection name instead of auto-generating from path |

### Embeddings

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_EMBEDDING_URL` | `http://localhost:8080` | OpenAI-compatible embedding API base URL |
| `VECTOR_EMBEDDING_MODEL` | `""` | Model name to pass in API requests. **Set this to match your server's model.** |
| `VECTOR_EMBEDDING_DIM` | `0` | Embedding dimensions. `0` = auto-detect at runtime. Common values: 384, 768, 1024, 1536, 4096 |
| `VECTOR_EMBEDDING_BATCH_SIZE` | `8` | Number of texts per embedding API request |
| `VECTOR_EMBEDDING_CONCURRENCY` | `2` | Max concurrent embedding API requests |
| `VECTOR_EMBEDDING_TIMEOUT` | `120` | Timeout in seconds for embedding API requests |
| `VECTOR_EMBEDDING_MAX_TEXT_CHARS` | `None` | Optional complete formatted input character limit; oversized inputs are rejected |
| `VECTOR_EMBEDDING_PROFILE` | `auto` | `raw`, `qwen3`, or `nemotron3`; `auto` recognizes supported model names and otherwise uses `raw` |
| `VECTOR_EMBEDDING_QUERY_INSTRUCTION` | Retrieval instruction | Instruction used by the Qwen3 query formatter |
| `VECTOR_EMBEDDING_QUERY_PREFIX` | Profile default | Exact optional query-prefix override, including whitespace |
| `VECTOR_EMBEDDING_DOCUMENT_PREFIX` | Profile default | Exact optional document-prefix override, including whitespace |
| `VECTOR_EMBEDDING_TOKENIZER_PATH` | `None` | Local tokenizer JSON; requires the `tokenizer` extra and never downloads a model |
| `VECTOR_EMBEDDING_INPUT_ENCODING` | `text` | `text` or `token_ids`; token IDs require a local tokenizer and a server with the identical ID vocabulary |
| `VECTOR_EMBEDDING_MAX_INPUT_TOKENS` | `None` | Optional complete formatted input token limit; requires a matching local tokenizer |
| `VECTOR_EMBEDDING_MAX_INPUT_BYTES` | `None` | Optional complete formatted input UTF-8 byte limit; `0` also disables it |
| `VECTOR_EMBEDDING_TOKENIZER_ADD_SPECIAL_TOKENS` | `True` | Apply the tokenizer's special-token postprocessor; match the backend's tokenization policy |
| `VECTOR_EMBEDDING_RESERVED_TOKENS` | `0` | Additional backend token overhead not produced by the configured tokenizer |
| `VECTOR_EMBEDDING_MAX_REQUEST_BYTES` | `None` | Optional serialized embedding HTTP body budget; batches are divided without shortening inputs |
| `VECTOR_EMBEDDING_CACHE_NAMESPACE` | `None` | Stable model/deployment identity. Setting a non-empty value opts `embed_all()` into persistent reuse; leave unset to disable it |
| `VECTOR_EMBEDDING_GLOBAL_CONCURRENCY` | `0` | Max embedding HTTP attempts across all local processes sharing an endpoint/model. `0` disables cross-process limiting |

All processes using the same endpoint, model, and cache directory must use the
same nonzero `VECTOR_EMBEDDING_GLOBAL_CONCURRENCY` value. During a rolling
configuration change, mixed capacities make the effective limit the largest
capacity any process can see; once an enabled scope has created its capacity
manifest, later clients with a different nonzero value fail fast. A process set
to `0` deliberately opts out and cannot be constrained by other clients. Stop every process
using that backend and remove the scope's limiter directory before changing it.

### Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_CACHE_DIR` | `~/.cache/vector-core` | Directory for embedding cache and other reconstructible data |
| `VECTOR_CACHE_MAX_SIZE_GB` | `10.0` | Max cache size in GB |
| `VECTOR_CACHE_MAX_ENTRIES` | `100000` | Max number of cached embeddings |

### Shared Data

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_SHARED_DATA_DIR` | `~/.local/share/vector-core` | Directory for persistent shared data (glossary.db, facts.db) |

### Indexing

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_MAX_FILE_SIZE_KB` | `500` | Max file size to index (in KB) |
| `VECTOR_MAX_PAYLOAD_CONTENT_CHARS` | `30000` | Legacy compatibility setting; unused by full-source indexing, which retains complete content |

### Search (Hybrid RRF)

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_DENSE_WEIGHT` | `1.0` | Weight for dense (embedding) results in RRF |
| `VECTOR_SPARSE_WEIGHT` | `0.8` | Weight for sparse (TF-IDF) results in RRF |
| `VECTOR_RRF_K` | `60` | RRF smoothing constant |
| `VECTOR_RRF_PREFETCH_LIMIT` | `50` | Number of results to prefetch from each source before fusion |

### Timeouts

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_SEARCH_TIMEOUT` | `30` | Timeout in seconds for hybrid search operations |
| `VECTOR_QDRANT_OPERATION_TIMEOUT` | `60` | Transport timeout in seconds applied to every Qdrant request. Bounds above it bind first: health checks use their own 5s limit, and hybrid search uses `VECTOR_SEARCH_TIMEOUT` |
| `VECTOR_FILE_LOCK_TIMEOUT` | `10.0` | Timeout in seconds for file locking |

### Limits & Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_SCROLL_MAX_RESULTS` | `100000` | Max points returned by scroll operations |
| `VECTOR_GLOBAL_VOCAB_CACHE_TTL` | `5.0` | TTL in seconds for the global TF-IDF vocabulary cache |
| `VECTOR_CONTENT_HASH_DISPLAY_LENGTH` | `16` | Truncated hash length for display/logging |
| `VECTOR_CIRCUIT_BREAKER_THRESHOLD` | `5` | Consecutive embedding failures before circuit opens |
| `VECTOR_CIRCUIT_BREAKER_RESET_SECONDS` | `60.0` | Seconds to wait before retrying after circuit opens |

## Usage

```python
from vector_core.embeddings import EmbeddingClient, EmbeddingCache, SparseVectorizer
from vector_core.storage import QdrantStorage, HybridSearcher
from vector_core.indexing import FileDiscovery, ChangeDetector
from vector_core.search import QueryPreprocessor
from vector_core.settings import settings
```

### Embedding text

```python
client = EmbeddingClient(
    base_url=settings.embedding_url,
    model=settings.embedding_model,
)
vectors = await client.embed_batch(["hello world", "vector search"])
# Or single text:
vector = await client.embed_single("hello world")
# Retrieval queries use the model's query formatting:
query_vector = await client.embed_single("find a greeting", role="query")
```

All asynchronous and synchronous embedding methods accept `role="document"` or `role="query"`. Document is the backward-compatible default. Query and document inputs remain separate in both persistent and in-memory caches. The Nemotron3 profile formats queries with `query: ` and documents with `passage: `; Qwen3 adds its retrieval instruction to queries and leaves documents unprefixed. Formatting is applied once inside the client, so callers should pass raw text. Set an explicit profile when a generic model alias hides its family.

Embedding methods preserve every input character and return exactly one vector per input, or raise an error. They never truncate or implicitly average multiple spans. Formatting profiles do not infer a model's deployed capacity. Configure any character, byte, or token limits explicitly; all include the role prefix. An oversized input raises `EmbeddingInputTooLongError`, a `ValueError`, before it is sent. Backend HTTP 400/422 responses raise `EmbeddingInputRejectedError`, also a `ValueError`, so applications do not mistake rejected input for an outage and silently fall back to sparse search. Empty strings are invalid; nonempty whitespace is retained so source spans remain lossless. Consumers should separately validate blank user queries.

Install `vector-core[tokenizer]` to load a local tokenizer JSON without model downloads. Token limits require that tokenizer, and its special-token policy must match the backend. The tokenizer's own truncation and padding are disabled. `VECTOR_EMBEDDING_RESERVED_TOKENS` accounts for additional backend overhead; no model-family overhead is guessed. Without configured limits, complete inputs are forwarded and backend errors propagate. A client cannot independently verify an unconfigured backend's truncation behavior. Tokenizer content hash, special-token policy, formatting, preprocessing version, and input limits participate in identity; changes invalidate reuse and trigger consumer migration.

Set `input_encoding="token_ids"` on either client, or `VECTOR_EMBEDDING_INPUT_ENCODING=token_ids`, to send a list of token-ID lists instead of strings. This requires a configured local tokenizer and a backend that accepts ID inputs with exactly the same vocabulary-to-ID mapping. Nonzero BPE dropout is rejected to keep tokenization deterministic, and inputs encoding to an empty ID sequence fail before HTTP. The complete role-formatted string is encoded with `tokenizer_add_special_tokens`; transport never adds or removes EOS or other tokens and never falls back to text. The validated IDs are reused for request budgeting, transmission, and retries. Local tokenizer normalization and literal special-token parsing may differ from the server's text tokenizer, so this mode can produce different vectors from text mode even for the same source. Source strings and split offsets remain unmodified. Token-ID mode, tokenizer implementation, package version, and JSON hash bind caches and index generations; text-mode identities remain compatible. Change the deployment namespace if server-side processing changes.

Use `split_text()` explicitly when indexing documents larger than one input. It is synchronous and returns exact half-open source-character spans that partition the complete text. Every resulting input fits the configured limits, including any repeated contextual prefix:

```python
spans = client.split_text(source, context_prefix="Module: example\n")
vectors = await client.embed_all(["Module: example\n" + span.text for span in spans])
assert "".join(span.text for span in spans) == source
```

Each span needs its own searchable representation. Tokenization uses bounded windows rather than repeatedly encoding the remaining document, and does not promise globally maximal spans for every tokenizer. Oversized queries are rejected by the ordinary embedding methods; applications wanting multi-query retrieval must explicitly split and combine searches. HTTP request budgets are independent of model capacity: whole inputs are packed by their actual serialized request size, and an oversized singleton raises `EmbeddingRequestTooLargeError` rather than reducing its model context. Transport batching does not change embedding identity.

In token-ID mode, tokenless source runs are retained in adjacent spans when input limits permit. Source that cannot be partitioned into nonempty ID sequences is rejected explicitly; the splitter never drops whitespace or returns a span that fails solely because its ID sequence is empty.

Persistent reuse applies to `embed_all()` indexing workloads and is deliberately
opt-in. Configure an immutable model-artifact/deployment fingerprint as the
namespace, change it whenever weights or serving behavior change, and configure
the output dimension so reads can begin on the first call. Reusing a namespace
across different weights is unsupported because the OpenAI embeddings API does
not expose a standard model-revision fingerprint:

```bash
export VECTOR_EMBEDDING_CACHE_NAMESPACE="model-artifact-sha256:0123456789abcdef..."
export VECTOR_EMBEDDING_DIM=1024
export VECTOR_EMBEDDING_GLOBAL_CONCURRENCY=4
```

Keys include the namespace, API endpoint, model name, output dimension, cache schema,
preprocessing version, and hash of the complete formatted input. Values are stored
as binary float32 vectors. If the configured dimension is `0`, the first request
infers it and populates the cache; later calls can read it. Cache I/O errors fail
open and do not prevent embedding requests. The global limit covers only active
HTTP attempts, not cache hits or retry backoff.

### Hybrid search

```python
searcher = HybridSearcher(storage)
results = await searcher.search(
    collection="my_collection",
    dense_query=dense_vector,
    sparse_query=sparse_vector,
    limit=10,
)
```

### Automatic embedding-model migration

Consumers can bind a logical index to the current embedding configuration through `ensure_embedding_collection()`. Its identity includes model, deployment namespace, endpoint, resolved dimension, and preprocessing. Endpoint authentication credentials are excluded from stored metadata; an opaque digest distinguishes authenticated deployments. A first uncached embedding request validates the configured dimension or resolves an automatic dimension. An incompatible or unidentified legacy collection is rebuilt into a new physical collection; the old collection remains intact.

```python
from vector_core.storage.embedding_migration import (
    embedding_collection_lock,
    ensure_embedding_collection,
)

async with embedding_collection_lock(storage, logical_name):
    generation = await ensure_embedding_collection(
        storage, logical_name, embedder, resolve_legacy_text, lock_held=True,
    )
    await storage.upsert_batch(generation.physical_name, points)
```

Hold the operation lock through all source mutations and vector writes. The helper creates a task-reentrant local process lock and checks an existing client's binding before returning a physical target. Concurrent clients wait up to at least one hour for an ongoing rebuild, rather than using the short ordinary file-lock timeout. Common loopback endpoint spellings share one lock. Reads can resolve without an outer lock, then retain the returned target throughout their operation. The `logical_name + "__active"` alias is a discovery pointer only: never query or write vectors through it. Shared collections migrate all point types together. An old client fails closed after a different identity becomes active; a fresh client can intentionally change the configured model. Returning to an earlier model copies current content rather than reusing an outdated generation.

New index points retain the complete raw document input before role formatting. Use `embedding_text` when the input is not already retained, or `embedding_text_field="content"` when `content` contains the exact input. References must name a nonempty string in the same payload; invalid references fail closed. An explicit `embedding_text` takes precedence. Legacy documents, facts, and note chunks retain usable text. Legacy note summaries lack their original body excerpt, so the shared resolver creates a canonical summary from retained title, tags, and category and records `embedding_text_source="legacy-note-metadata"`. Their full retained chunks remain independently searchable. Truncated glossary definitions are recovered from the corresponding verified SQLite row. Unavailable required source content fails closed without publishing a partial build.

Oversized retained inputs become independently searchable fragments. The canonical point keeps its original ID, full raw payload, and original sparse vector; its dense vector represents its declared first span. Deterministic derived points hold only bounded snippets and retrieval metadata. The `embedding_fragment` marker records parent identity, complete-source hash, source offsets, ordinal, and count, including for a single-span source. Migration validates old derived lineage, replans canonical raw inputs, and regenerates children rather than recursively splitting previous children. Historical generations remain intact. Fragment sparse vectors use the existing vocabulary without registering additional logical documents.

Writers can use `fragment_point()` and `upsert_fragment_group()` from `vector_core.storage.embedding_fragments`. A group is embedded and byte-bounded writes complete before stale children are pruned. Domain adapters select child metadata rather than duplicating complete raw sources. Source reconstruction must exclude `is_derived_fragment(payload)` records, while search includes them. `fragment_text(payload)` extracts the matching declared span for presentation without modifying stored source fields. `HybridSearcher.search(group_by="document_id")` retrieves distinct groups independently from each enabled branch before fusing by group ID, avoiding fragment crowding in entity-level results. Glossary searches hydrate the canonical expansion and display definition for winning derived hits; their `content` and fragment metadata retain the independently matched snippet.

Generic `HybridSearcher` results contain the representative point's payload, including when grouped; they do not automatically hydrate canonical entities. Fact children retain matching snippets and identity metadata while `object` and `context` remain on the canonical point. Fact presentation callers must fetch those canonical fields for final derived winners, validate their parent lineage and entity identity, and preserve the winning snippet and fragment marker separately. `FactIndexer` writes these groups but does not provide a fact search or presentation API.

Before replacing a group, the writer durably journals its previous IDs, payloads, and all vectors under `VECTOR_CACHE_DIR/embedding-fragment-recovery/`, together with the target and newly introduced IDs. Journal files have mode `0600`. Serialization and filesystem synchronization run in worker threads, which settle before releasing the caller's writer lock even under cancellation. Cancellation during journal creation removes the unused journal without mutating storage. A failed write, stale-child deletion, or cancellation during replacement settles in-flight mutations and restores the previous group before returning the original error. If rollback also fails, `FragmentGroupRecoveryError` includes the recovery path and both errors; bulk fact indexing stops rather than hiding that failure. Successful operations and confirmed rollbacks remove their journals. Cancellation during final journal cleanup waits for cleanup and leaves the already completed storage outcome intact.

This is compensation under the caller's writer lock, not a cross-request Qdrant transaction: concurrent readers can observe intermediate requests, and process death can interrupt replacement or rollback. Retained journals are recovery evidence, not commands to replay automatically. Preserve them after a recovery error, stop competing writers, and inspect the recorded target and current group before restoring prior points and removing introduced IDs. A later successful edit can make an older journal stale; never restore it blindly or discard it with ordinary reconstructible caches before resolving the failure.

Migration avoids duplicating text already retained in `content` and splits upserts using their serialized UTF-8 size, with a conservative 30 MiB request budget. A single point exceeding that budget produces an explicit size error without truncating source content. The original collection remains intact and an incomplete candidate is never selected. Deployments with a smaller server request limit may still require a smaller transport budget; migration never increases server limits automatically.

An optional `finalize_candidate` callback can reconstruct domain-owned source groups before publication; its text resolver may return `None` to omit points belonging to those groups when retained inputs are unavailable. The callback must propagate every incomplete rebuild. Failed candidates stay unselected, retries build a fresh candidate, and originals remain available without their source files. Inventory and cleanup code must distinguish active generations from preserved and incomplete collections, and report canonical source counts separately from derived fragment counts.

Generation metadata and the source lineage are retained in point ID `0`. Ordinary collection metadata updates preserve the identity marker. Collection generations and their retained data are never deleted automatically. Restore a historical corpus only after stopping writers and selecting a compatible embedding identity; simply reverting model configuration rebuilds from the current corpus. Local file locks coordinate upgraded clients sharing a cache directory, not legacy binaries or uncoordinated writers on other hosts.

### Sparse index recovery

Persisted sparse indices can be inspected through the append-only global
vocabulary without guessing:

```python
tokens = vocabulary.get_tokens_by_indices(sparse_vector.indices)
```

The method raises `KeyError` if any requested index is absent. The low-level
`file_lock()` and `async_file_lock()` helpers also accept `shared=True` for
cross-process reader locks; exclusive locking remains the default.

Large recovery jobs can call `register_codebase_frequencies()` with an
already-aggregated token-to-document-frequency mapping instead of retaining one
token set per document. `rebuild_aggregate_doc_frequencies()` restores global
frequencies from all recorded codebase contributions without changing token IDs.

### Glossary

```python
from vector_core.glossary import GlossaryStore, GlossaryIndexer

store = GlossaryStore(db_path)
store.create(term="RRF", expansion="Reciprocal Rank Fusion", definition="A method for combining ranked lists", domain="search")
```

### Facts (knowledge graph)

```python
from vector_core.facts import FactStore, FactIndexer

store = FactStore(db_path)
store.create(subject="vector-core", predicate="provides", object_value="hybrid search")
```

### Settings mixin for downstream servers

Use `VectorCoreSettingsMixin` to inherit all vector-core settings in your server's settings class without duplicating fields:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from vector_core.settings import VectorCoreSettingsMixin

class MyServerSettings(VectorCoreSettingsMixin, BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MYSERVER_")

    # Server-specific settings only
    my_setting: str = "default"

    # Access vector-core settings via attribute delegation:
    # settings.embedding_url -> vector_core.settings.embedding_url
```

## Subsystems

### Glossary

A shared glossary system backed by SQLite with Qdrant indexing for semantic lookup. Multiple MCP servers can read/write the same glossary database. Includes `GlossaryStore` for CRUD, `GlossaryIndexer` for vector indexing, and `GlossaryToolHelper` for MCP tool implementations.

### Facts

A knowledge graph subsystem storing subject-predicate-object triples in SQLite with source tracking and integrity management. Supports semantic search over facts via Qdrant indexing. Includes `FactStore` for storage, `FactIndexer` for vector indexing, and `SourceIntegrityManager` for tracking fact provenance.

## Architecture

See [PATTERNS.md](PATTERNS.md) for detailed documentation of architectural patterns including:

- Singleton patterns (async and sync) for shared resources
- Error handling with `error_response()` and `ErrorCollector`
- Circuit breaker on the embedding client
- SQLite thread safety via `ThreadSafeSQLiteStore`
- Cross-process locking strategies (WAL, fcntl)
- TTL caching and global vocabulary management
- Retry with exponential backoff
- Query preprocessing and synonym expansion

## License

[Apache License 2.0](LICENSE)
