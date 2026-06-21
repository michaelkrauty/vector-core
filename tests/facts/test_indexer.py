"""Tests for facts/indexer module (FactIndexer.index_all / _train_vocabulary).

These focus on the indexer processing the COMPLETE fact corpus:
- index_all / _train_vocabulary must not be capped at 50 facts.
- the GlobalVocabulary must be registered from every fact, not just the
  incremental batch (register_codebase replaces the whole contribution).
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.facts.database import FactStore
from vector_core.facts.indexer import FACTS_CODEBASE_ID, FactIndexer


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def store(temp_dir):
    s = FactStore(db_path=temp_dir / "facts.db")
    yield s
    s.close()


@pytest.fixture
def vocab(temp_dir):
    """A real GlobalVocabulary so doc-count assertions exercise real state."""
    return GlobalVocabulary(db_path=temp_dir / "vocab.db")


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.collection_exists = AsyncMock(return_value=True)
    storage.create_collection = AsyncMock()
    storage.ensure_payload_indexes = AsyncMock()
    storage.upsert_batch = AsyncMock()
    storage.delete_by_filter = AsyncMock()
    storage.scroll_points = AsyncMock(return_value=[])
    storage.close = AsyncMock()
    return storage


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_single_cached = AsyncMock(return_value=[0.1] * 4096)
    return embedder


@pytest.fixture
def indexer(store, mock_storage, mock_embedder, vocab):
    return FactIndexer(
        fact_store=store,
        storage=mock_storage,
        embedder=mock_embedder,
        global_vocab=vocab,
        collection_name="test_facts",
    )


def _make_facts(store: FactStore, n: int) -> list:
    """Create n distinct facts in the store."""
    return [store.create(f"subject_{i}", "relates_to", f"object_{i}") for i in range(n)]


class TestIndexAllCompleteCorpus:
    """index_all must index every fact, not just the 50 most recent."""

    @pytest.mark.asyncio
    async def test_indexes_more_than_fifty_facts(self, indexer, store, mock_storage):
        """A store with >50 facts indexes all of them (was capped at 50 by
        list_summaries' default limit)."""
        _make_facts(store, 60)

        result = await indexer.index_all(force=True)

        assert result["total"] == 60
        assert result["indexed"] == 60
        # _index_fact upserts one point per fact
        assert mock_storage.upsert_batch.await_count == 60

    @pytest.mark.asyncio
    async def test_vocabulary_trained_on_all_facts(self, indexer, store, vocab):
        """GlobalVocabulary document count reflects every fact, not 50."""
        _make_facts(store, 60)

        await indexer.index_all(force=True)

        assert vocab.get_codebase_doc_count(FACTS_CODEBASE_ID) == 60

    @pytest.mark.asyncio
    async def test_emptying_store_clears_fact_vocabulary(self, indexer, store, vocab):
        """Deleting the last fact and reindexing clears the facts vocabulary
        contribution; otherwise the doc count and IDF stay skewed and
        index_fact() skips retraining on the next add."""
        facts = _make_facts(store, 3)
        await indexer.index_all(force=True)
        assert vocab.get_codebase_doc_count(FACTS_CODEBASE_ID) == 3

        for f in facts:
            store.delete(f.id)
        result = await indexer.index_all(force=True)

        assert result["total"] == 0
        assert vocab.get_codebase_doc_count(FACTS_CODEBASE_ID) == 0


class TestIncrementalVocabularyCorpus:
    """Incremental index_all must register vocabulary from the full corpus."""

    @pytest.mark.asyncio
    async def test_incremental_registers_full_corpus_not_just_new(
        self, indexer, store, vocab, mock_storage
    ):
        """register_codebase replaces the whole 'facts' contribution, so an
        incremental run that registered only the new facts' tokens dropped the
        vocabulary document count to the size of the incremental batch."""
        initial = _make_facts(store, 3)
        await indexer.index_all(force=True)
        assert vocab.get_codebase_doc_count(FACTS_CODEBASE_ID) == 3

        # The first three facts are now "already indexed".
        mock_storage.scroll_points = AsyncMock(
            return_value=[{"fact_id": str(f.id)} for f in initial]
        )
        # Add two more facts and run an incremental pass.
        _make_facts_extra = [
            store.create("subject_new_a", "relates_to", "object_new_a"),
            store.create("subject_new_b", "relates_to", "object_new_b"),
        ]
        assert len(_make_facts_extra) == 2
        mock_storage.upsert_batch.reset_mock()

        result = await indexer.index_all(force=False)

        # Vocabulary reflects all five facts, not just the two new ones.
        assert vocab.get_codebase_doc_count(FACTS_CODEBASE_ID) == 5
        # Only the two new facts are upserted (incremental optimization intact).
        assert result["total"] == 5
        assert result["indexed"] == 2
        assert mock_storage.upsert_batch.await_count == 2


class TestTrainVocabularyCompleteCorpus:
    """_train_vocabulary must tokenize every fact."""

    @pytest.mark.asyncio
    async def test_trains_on_more_than_fifty_facts(self, indexer, store, vocab):
        _make_facts(store, 55)

        await indexer._train_vocabulary()

        assert vocab.get_codebase_doc_count(FACTS_CODEBASE_ID) == 55


class TestIndexAllRobustness:
    """index_all must tolerate individual bad facts and never delete before
    the corpus has been read."""

    @pytest.mark.asyncio
    async def test_skips_fact_whose_tokenization_fails(
        self, indexer, store, vocab, mock_storage, monkeypatch
    ):
        store.create("poison", "relates_to", "value")
        store.create("good", "relates_to", "value2")

        real_tokenize = vocab.tokenize

        def flaky_tokenize(text):
            if "poison" in text:
                raise RuntimeError("tokenizer blew up")
            return real_tokenize(text)

        monkeypatch.setattr(vocab, "tokenize", flaky_tokenize)

        result = await indexer.index_all(force=True)

        # The poison fact is skipped; only the good fact is counted and upserted.
        assert result["total"] == 1
        assert result["indexed"] == 1
        assert mock_storage.upsert_batch.await_count == 1

    @pytest.mark.asyncio
    async def test_force_does_not_delete_when_corpus_read_fails(
        self, indexer, store, mock_storage, monkeypatch
    ):
        """A force reindex must read the corpus before clearing points, so a
        read failure leaves the existing index intact."""
        def boom():
            raise RuntimeError("db locked")
            yield  # pragma: no cover  (make boom a generator)

        monkeypatch.setattr(store, "iter_all", boom)

        with pytest.raises(RuntimeError, match="db locked"):
            await indexer.index_all(force=True)

        mock_storage.delete_by_filter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_force_empty_store_clears_stale_points(self, indexer, mock_storage):
        """force=True with no readable facts still clears stale fact points, so a
        force rebuild never leaves deleted facts searchable."""
        result = await indexer.index_all(force=True)

        assert result["total"] == 0
        mock_storage.delete_by_filter.assert_awaited()


class TestIndexFactAtomic:
    """index_fact must not delete the existing point before re-indexing: the
    point id is stable per fact, so the upsert overwrites in place. Pre-deleting
    would drop the fact from search if the subsequent embed/upsert fails."""

    @pytest.mark.asyncio
    async def test_index_fact_does_not_pre_delete(self, indexer, store, monkeypatch):
        fact = store.create("a", "rel", "b")
        indexer._delete_fact_point = AsyncMock()
        indexer._index_fact = AsyncMock()
        indexer._ensure_global_vocab = AsyncMock()
        indexer.ensure_collection = AsyncMock()
        # Pretend the vocab is already trained so index_fact does not retrain.
        monkeypatch.setattr(
            indexer.global_vocab,
            "get_codebase_doc_count",
            MagicMock(return_value=1),
        )

        await indexer.index_fact(fact)

        # The upsert (inside _index_fact) overwrites the stable-id point; no
        # pre-delete that could strand the fact on a transient upsert failure.
        indexer._delete_fact_point.assert_not_called()
        indexer._index_fact.assert_awaited_once_with(fact)
