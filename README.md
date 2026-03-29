# vector-core

Shared vector search infrastructure for MCP servers. Provides dense and sparse embeddings, hybrid search with Reciprocal Rank Fusion, Qdrant vector storage, and supporting utilities (caching, file discovery, change detection, glossary, facts) as a reusable Python library.

## Features

- **Dense embeddings** via any OpenAI-compatible API (llama.cpp, vLLM, Ollama, OpenAI, etc.)
- **Sparse embeddings** via TF-IDF with a shared global vocabulary
- **Hybrid search** combining dense + sparse results with RRF (Reciprocal Rank Fusion)
- **Qdrant vector storage** with health checks and automatic reconnection
- **Persistent SQLite-backed embedding cache** to avoid redundant API calls
- **File discovery and change detection** with `.gitignore`-aware path filtering
- **Glossary subsystem** -- shared term definitions stored in SQLite and indexed in Qdrant
- **Facts subsystem** -- knowledge graph storage with subject-predicate-object triples and source integrity tracking
- **Circuit breaker** on the embedding client to fail fast when the upstream API is down
- **Query preprocessing** with synonym expansion (generic and code-specific)
- **Structured error handling** with error codes, collectors, and consistent response formatting
- **Pydantic-based configuration** via environment variables with validation

## Prerequisites

- **Python 3.12+**
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
pip install -e ".[dev]"
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
| `VECTOR_EMBEDDING_MAX_TEXT_CHARS` | `8000` | Max characters before text truncation |

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
| `VECTOR_MAX_PAYLOAD_CONTENT_CHARS` | `30000` | Max chunk content length stored in Qdrant payloads |

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
| `VECTOR_QDRANT_OPERATION_TIMEOUT` | `60` | Timeout in seconds for bulk upsert/delete operations |
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
```

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
