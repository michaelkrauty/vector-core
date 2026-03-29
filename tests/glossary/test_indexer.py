"""Tests for glossary/indexer module."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from vector_core.glossary.indexer import (
    GLOSSARY_CODEBASE_ID,
    GLOSSARY_PAYLOAD_INDEXES,
    GlossaryIndexer,
    _generate_embedding_content,
)
from vector_core.glossary.models import GlossaryEntry
from vector_core.glossary.store import GlossaryStore


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
def mock_storage():
    """Create mock QdrantStorage."""
    storage = MagicMock()
    storage.ensure_collection_with_indexes = AsyncMock(return_value=True)
    storage.upsert_batch = AsyncMock()
    storage.upsert_point = AsyncMock()
    storage.delete_by_filter = AsyncMock()
    storage.query_dense = AsyncMock(return_value=[])
    storage.create_point = MagicMock(return_value=MagicMock())
    storage.close = AsyncMock()
    return storage


@pytest.fixture
def mock_embedder():
    """Create mock EmbeddingClient."""
    embedder = MagicMock()
    # Return a fake 4096-dim embedding
    embedder.embed_single_cached = AsyncMock(return_value=[0.1] * 4096)
    return embedder


@pytest.fixture
def mock_vocab(temp_db):
    """Create mock GlobalVocabulary."""
    from vector_core.embeddings.global_vocab import GlobalVocabulary
    vocab = GlobalVocabulary(db_path=temp_db / "vocab.db")
    return vocab


@pytest.fixture
def mock_hybrid_searcher():
    """Create mock HybridSearcher."""
    from vector_core.storage.hybrid import SearchResult

    searcher = MagicMock()
    searcher.search = AsyncMock(return_value=[])
    return searcher


@pytest.fixture
def indexer(store, mock_storage, mock_embedder, mock_vocab, mock_hybrid_searcher):
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
        from datetime import UTC, datetime

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
        from datetime import UTC, datetime

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
        from datetime import UTC, datetime

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
    async def test_ensure_collection(self, indexer, mock_storage):
        """Should call ensure_collection_with_indexes."""
        result = await indexer.ensure_collection()

        assert result is True
        mock_storage.ensure_collection_with_indexes.assert_called_once_with(
            "test_collection",
            GLOSSARY_PAYLOAD_INDEXES,
        )

    @pytest.mark.asyncio
    async def test_index_all_empty(self, indexer, mock_storage):
        """Should return 0 for empty store."""
        result = await indexer.index_all()

        assert result == 0
        mock_storage.upsert_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_index_all_with_entries(self, indexer, store, mock_storage, mock_embedder):
        """Should index all entries."""
        store.create(term="API", expansion="Application Programming Interface", definition="A set of protocols")
        store.create(term="SDK", expansion="Software Development Kit", definition="Tools for development")

        result = await indexer.index_all()

        assert result == 2
        mock_storage.upsert_batch.assert_called_once()
        # Should have called embedder for each entry
        assert mock_embedder.embed_single_cached.call_count == 2

    @pytest.mark.asyncio
    async def test_index_entry(self, indexer, store, mock_storage, mock_embedder):
        """Should index single entry."""
        entry = store.create(term="API", expansion="Application Programming Interface", definition="A set of protocols")

        await indexer.index_entry(entry.id)

        mock_storage.upsert_point.assert_called_once()
        mock_embedder.embed_single_cached.assert_called()

    @pytest.mark.asyncio
    async def test_delete_entry_index(self, indexer, store, mock_storage):
        """Should delete entry from index."""
        entry = store.create(term="API", expansion="Application Programming Interface", definition="A set of protocols")

        await indexer.delete_entry_index(entry.id)

        mock_storage.delete_by_filter.assert_called_once()

    @pytest.mark.asyncio
    async def test_search(self, indexer, mock_embedder):
        """Should perform hybrid semantic search."""
        await indexer.search("application interface")

        mock_embedder.embed_single_cached.assert_called_once()
        indexer.hybrid_searcher.search.assert_called_once()

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
