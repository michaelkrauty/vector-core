"""Tests for glossary/indexer module."""

import tempfile
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from vector_core.embeddings.client import EmbeddingClient
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.glossary import indexer as indexer_module
from vector_core.glossary.indexer import (
    GLOSSARY_CODEBASE_ID,
    GLOSSARY_PAYLOAD_INDEXES,
    GlossaryIndexer,
    _generate_embedding_content,
)
from vector_core.glossary.models import GlossaryEntry
from vector_core.glossary.store import GlossaryStore
from vector_core.settings import settings


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def store(temp_db):
    """Create a GlossaryStore instance with temporary database."""
    s = GlossaryStore(db_path=temp_db / "glossary.db")
    yield s
    s.close()


@pytest.fixture
def mock_storage(temp_db, monkeypatch):
    """Create mock QdrantStorage."""
    monkeypatch.setattr(settings, "cache_dir", temp_db)
    storage = MagicMock()
    storage.url = "http://glossary-tests.invalid"
    storage.ensure_collection_with_indexes = AsyncMock(return_value=True)
    storage.upsert_batch = AsyncMock()
    storage.upsert_point = AsyncMock()
    storage.delete_by_filter = AsyncMock()
    storage.query_dense = AsyncMock(return_value=[])
    storage.create_point = MagicMock(return_value=MagicMock())
    storage.close = AsyncMock()
    storage.get_client = AsyncMock(
        return_value=SimpleNamespace(
            retrieve=AsyncMock(return_value=[]),
            upsert=storage.upsert_batch,
            scroll=AsyncMock(return_value=([], None)),
            delete=AsyncMock(),
        )
    )
    return storage


@pytest.fixture
def mock_embedder():
    """Create mock EmbeddingClient."""
    embedder = EmbeddingClient(dim=4096)
    # Return a fake 4096-dim embedding
    embedder.embed_single_cached = AsyncMock(return_value=[0.1] * 4096)
    embedder.embed_all = AsyncMock(
        side_effect=lambda texts, **kwargs: [[0.1] * 4096 for _ in texts]
    )
    return embedder


@pytest.fixture
def mock_vocab(temp_db):
    """Create mock GlobalVocabulary."""
    vocab = GlobalVocabulary(db_path=temp_db / "vocab.db")
    return vocab


@pytest.fixture
def mock_hybrid_searcher():
    """Create mock HybridSearcher."""
    searcher = MagicMock()
    searcher.search = AsyncMock(return_value=[])
    return searcher


@pytest.fixture
def migration(monkeypatch):
    state = SimpleNamespace(locked=False)

    @asynccontextmanager
    async def lock(storage, logical_name):
        assert not state.locked
        state.locked = True
        try:
            yield
        finally:
            state.locked = False

    async def ensure(*args, lock_held=False, **kwargs):
        assert lock_held == state.locked
        return SimpleNamespace(physical_name="test_generation", migrated=True)

    state.ensure = AsyncMock(side_effect=ensure)
    monkeypatch.setattr(indexer_module, "ensure_embedding_collection", state.ensure)
    monkeypatch.setattr(
        indexer_module, "active_embedding_collection", AsyncMock(return_value="test_collection")
    )
    monkeypatch.setattr(indexer_module, "embedding_collection_lock", lock)
    return state


@pytest.fixture
def indexer(store, mock_storage, mock_embedder, mock_vocab, mock_hybrid_searcher, *, migration):
    """Create a GlossaryIndexer with mocks."""
    idx = GlossaryIndexer(
        collection_name="test_collection",
        glossary_store=store,
        storage=mock_storage,
        embedder=mock_embedder,
        global_vocab=mock_vocab,
    )
    # Replace hybrid searcher with mock
    idx.hybrid_searcher = mock_hybrid_searcher
    return idx


class TestGenerateEmbeddingContent:
    """Tests for _generate_embedding_content helper."""

    def test_basic_content(self):
        """Should include term, expansion, and definition."""
        entry = GlossaryEntry(
            id=uuid4(),
            term="USAF",
            expansion="United States Air Force",
            definition="The air service branch",
            domain=None,
            aliases=[],
            created=datetime.now(UTC),
            modified=datetime.now(UTC),
        )

        content = _generate_embedding_content(entry)

        assert "USAF" in content
        assert "United States Air Force" in content
        assert "The air service branch" in content

    def test_includes_domain(self):
        """Should include domain if present."""
        entry = GlossaryEntry(
            id=uuid4(),
            term="USAF",
            expansion="United States Air Force",
            definition="The air service branch",
            domain="military",
            aliases=[],
            created=datetime.now(UTC),
            modified=datetime.now(UTC),
        )

        content = _generate_embedding_content(entry)
        assert "military" in content

    def test_includes_aliases(self):
        """Should include aliases."""
        entry = GlossaryEntry(
            id=uuid4(),
            term="USAF",
            expansion="United States Air Force",
            definition="The air service branch",
            domain=None,
            aliases=["Air Force", "US Air Force"],
            created=datetime.now(UTC),
            modified=datetime.now(UTC),
        )

        content = _generate_embedding_content(entry)
        assert "Air Force" in content
        assert "US Air Force" in content


class TestGlossaryIndexer:
    """Tests for GlossaryIndexer."""

    @pytest.mark.asyncio
    async def test_ensure_collection(self, indexer, mock_storage, migration):
        """Should resolve an embedding-compatible generation with indexes."""
        result = await indexer.ensure_collection()

        assert result is True
        assert migration.ensure.call_args.args[1] == "test_collection"
        assert migration.ensure.call_args.kwargs["payload_indexes"] == GLOSSARY_PAYLOAD_INDEXES

    @pytest.mark.asyncio
    async def test_index_all_empty(self, indexer, mock_storage, migration):
        """Should return 0 for empty store."""
        result = await indexer.index_all()

        assert result == 0
        mock_storage.upsert_batch.assert_not_called()
        migration.ensure.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_index_all_with_entries(self, indexer, store, mock_storage, mock_embedder):
        """Should index all entries."""
        store.create(
            term="API",
            expansion="Application Programming Interface",
            definition="A set of protocols",
        )
        store.create(
            term="SDK", expansion="Software Development Kit", definition="Tools for development"
        )

        result = await indexer.index_all()

        assert result == 2
        assert mock_storage.upsert_batch.await_count == 2
        # Should have called embedder for each entry
        assert mock_embedder.embed_all.await_count == 2

    @pytest.mark.asyncio
    async def test_index_entry(self, indexer, store, mock_storage, mock_embedder):
        """Should index single entry."""
        entry = store.create(
            term="API",
            expansion="Application Programming Interface",
            definition="A set of protocols",
        )

        await indexer.index_entry(entry.id)

        mock_storage.upsert_batch.assert_awaited_once()
        mock_embedder.embed_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_entry_index(self, indexer, store, mock_storage):
        """Should delete entry from index."""
        entry = store.create(
            term="API",
            expansion="Application Programming Interface",
            definition="A set of protocols",
        )

        await indexer.delete_entry_index(entry.id)

        mock_storage.delete_by_filter.assert_called_once()

    @pytest.mark.asyncio
    async def test_search(self, indexer, mock_embedder):
        """Should perform hybrid semantic search."""
        await indexer.search("application interface")

        mock_embedder.embed_single_cached.assert_called_once_with(
            "application interface", role="query"
        )
        indexer.hybrid_searcher.search.assert_called_once()
        assert indexer.hybrid_searcher.search.call_args.kwargs["group_by"] == "glossary_id"

    @pytest.mark.asyncio
    async def test_search_with_domain_filter(self, indexer, mock_embedder):
        """Should pass domain filter to hybrid search."""
        await indexer.search("application interface", domain="tech")

        indexer.hybrid_searcher.search.assert_called_once()
        call_args = indexer.hybrid_searcher.search.call_args
        filter_conditions = call_args.kwargs.get("filter_conditions", [])
        # Should have type=glossary and domain=tech filters
        assert len(filter_conditions) == 2


class TestGlossaryIndexerPayload:
    """Tests for payload creation."""

    def test_payload_structure(self, indexer, store):
        """Should create correct payload structure."""
        entry = store.create(
            term="API",
            expansion="Application Programming Interface",
            definition="A set of protocols",
            domain="tech",
            aliases=["Interface"],
        )

        payload = indexer._create_payload(entry)

        assert payload["type"] == "glossary"
        assert payload["glossary_id"] == str(entry.id)
        assert payload["term"] == "API"
        assert payload["term_normalized"] == "api"
        assert payload["expansion"] == "Application Programming Interface"
        assert payload["domain"] == "tech"
        assert payload["aliases"] == ["Interface"]
        assert "created" in payload
        assert "modified" in payload
        assert "entry_hash" in payload

    def test_payload_truncates_definition(self, indexer, store):
        """Should truncate long definitions."""
        long_def = "x" * 5000
        entry = store.create(
            term="API",
            expansion="Application Programming Interface",
            definition=long_def,
        )

        payload = indexer._create_payload(entry)

        assert len(payload["definition"]) == 2000
        assert payload["embedding_text"] == _generate_embedding_content(entry)


class TestGenerationTargets:
    async def test_mutation_pins_target_under_lock(
        self, indexer, store, mock_storage, mock_embedder, migration
    ):
        entry = store.create("API", "Interface", "definition")

        async def embed(texts, *, role):
            assert migration.locked
            assert role == "document"
            return [[0.1] * 4096 for _ in texts]

        mock_embedder.embed_all.side_effect = embed
        await indexer.index_entry(entry.id)
        assert mock_storage.upsert_batch.call_args.args[0] == "test_generation"
        assert migration.ensure.call_args.kwargs["lock_held"] is True
        assert indexer.collection_name == "test_collection"
        assert not migration.locked

    async def test_each_read_resolves_a_fresh_target(self, indexer, migration):
        migration.ensure.side_effect = [
            SimpleNamespace(physical_name="first", migrated=False),
            SimpleNamespace(physical_name="second", migrated=True),
        ]
        await indexer.search("query")
        await indexer.search("query")
        assert [
            call.kwargs["collection"] for call in indexer.hybrid_searcher.search.call_args_list
        ] == ["first", "second"]
        assert indexer.collection_name == "test_collection"

    async def test_injected_generation_does_not_reenter_lock(
        self, indexer, store, mock_storage, migration
    ):
        entry = store.create("API", "Interface", "definition")
        indexer._generation = SimpleNamespace(physical_name="bound", migrated=False)
        migration.locked = True
        await indexer.index_entry(entry.id)
        migration.ensure.assert_not_awaited()
        assert mock_storage.upsert_batch.call_args.args[0] == "bound"


class TestGlossaryIndexerConstants:
    """Tests for module constants."""

    def test_codebase_id(self):
        """Should have correct codebase ID."""
        assert GLOSSARY_CODEBASE_ID == "glossary"

    def test_payload_indexes(self):
        """Should have correct payload indexes."""
        field_names = [idx[0] for idx in GLOSSARY_PAYLOAD_INDEXES]
        assert "type" in field_names
        assert "term_normalized" in field_names
        assert "domain" in field_names
