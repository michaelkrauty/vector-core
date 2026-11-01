"""Tests for facts/indexer module (FactIndexer.index_all / _train_vocabulary).

These focus on the indexer processing the COMPLETE fact corpus:
- index_all / _train_vocabulary must not be capped at 50 facts.
- the GlobalVocabulary must be registered from every fact, not just the
  incremental batch (register_codebase replaces the whole contribution).
"""

import sqlite3
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from vector_core.embeddings.client import EmbeddingClient, EmbeddingServiceError
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.facts import indexer as indexer_module
from vector_core.facts.database import FactStore
from vector_core.facts.indexer import FACTS_CODEBASE_ID, FactIndexer, generate_fact_text
from vector_core.settings import settings
from vector_core.storage.embedding_fragments import is_derived_fragment
from vector_core.storage.embedding_migration import active_embedding_collection
from vector_core.storage.qdrant import QdrantStorage, generate_point_id


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
def mock_storage(temp_dir, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", temp_dir)
    storage = MagicMock()
    storage.url = "http://facts-tests.invalid"
    storage.collection_exists = AsyncMock(return_value=True)
    storage.create_collection = AsyncMock()
    storage.ensure_payload_indexes = AsyncMock()
    storage.upsert_batch = AsyncMock()
    storage.delete_by_filter = AsyncMock()
    storage.scroll_points = AsyncMock(return_value=[])
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
    embedder = EmbeddingClient(dim=4096)
    embedder.embed_single_cached = AsyncMock(return_value=[0.1] * 4096)
    embedder.embed_all = AsyncMock(
        side_effect=lambda texts, **kwargs: [[0.1] * 4096 for _ in texts]
    )
    return embedder


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
    monkeypatch.setattr(indexer_module, "embedding_collection_lock", lock)
    return state


@pytest.fixture
def indexer(store, mock_storage, mock_embedder, vocab, migration):
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
        indexer._index_fact.assert_awaited_once_with(fact, "test_generation")


class TestGenerationTargets:
    async def test_force_rebuild_reads_deletes_and_upserts_one_generation(
        self, indexer, store, mock_storage, mock_embedder, migration
    ):
        fact = store.create("subject", "relates_to", "object", context="x" * 5000)

        async def embed(texts, *, role):
            assert migration.locked
            assert role == "document"
            return [[0.1] * 4096 for _ in texts]

        mock_embedder.embed_all.side_effect = embed
        await indexer.index_all(force=True)
        assert mock_storage.delete_by_filter.call_args.args[0] == "test_generation"
        assert mock_storage.upsert_batch.call_args.args[0] == "test_generation"
        point = mock_storage.upsert_batch.call_args.args[1][0]
        assert point.payload["content"] == mock_embedder.embed_all.call_args.args[0][0]
        assert point.payload["embedding_text_field"] == "content"
        assert fact.context in point.payload["content"]
        assert indexer.collection_name == "test_facts"
        assert not migration.locked

    async def test_incremental_scan_uses_resolved_target(self, indexer, mock_storage):
        await indexer.index_all()
        assert mock_storage.scroll_points.call_args.args[0] == "test_generation"

    async def test_delete_uses_resolved_target(self, indexer, store, mock_storage, migration):
        fact = store.create("subject", "relates_to", "object")
        await indexer.delete_fact_index(fact.id)
        assert mock_storage.delete_by_filter.call_args.args[0] == "test_generation"
        assert migration.ensure.call_args.kwargs["lock_held"] is True

    async def test_injected_generation_does_not_reenter_lock(
        self, indexer, store, mock_storage, migration
    ):
        fact = store.create("subject", "relates_to", "object")
        indexer._generation = SimpleNamespace(physical_name="bound", migrated=False)
        migration.locked = True
        await indexer.index_fact(fact)
        migration.ensure.assert_not_awaited()
        assert mock_storage.upsert_batch.call_args.args[0] == "bound"


# ---------------------------------------------------------------------------
# Single-fact vocabulary accounting (index_fact / delete_fact_index)
# ---------------------------------------------------------------------------


@pytest.fixture
async def qdrant(temp_dir, monkeypatch):
    """In-process Qdrant, so read-back sees exactly what earlier writes stored."""
    monkeypatch.setattr(settings, "cache_dir", temp_dir)
    storage = QdrantStorage(url="http://facts-vocab-tests.invalid", embedding_dim=2)
    storage._client = AsyncQdrantClient(location=":memory:")
    try:
        yield storage
    finally:
        await storage.close()


def _live_indexer(store, storage, vocab, *, budget=None, name="facts_vocab"):
    embedder = EmbeddingClient(model="A", dim=2, max_input_bytes=budget)
    # Every embedding entry point is stubbed, identity probing included: these
    # tests must never reach a real embedding service.
    embedder.embed_single = AsyncMock(return_value=[1.0, 0.0])
    embedder.embed_single_cached = AsyncMock(return_value=[1.0, 0.0])
    embedder.embed_all = AsyncMock(side_effect=lambda texts, **kwargs: [[1.0, 0.0] for _ in texts])
    return FactIndexer(
        fact_store=store,
        storage=storage,
        embedder=embedder,
        global_vocab=vocab,
        collection_name=name,
    )


@pytest.fixture
def live(store, qdrant, vocab):
    return _live_indexer(store, qdrant, vocab)


def _facts_vocab(vocab):
    """The facts contribution as ``(doc_count, nonzero doc_freqs)``.

    Vocabulary indices are append-only, so a retired term keeps its row at
    doc_freq 0; only the nonzero frequencies describe the indexed corpus.
    """
    doc_freq = {token: freq for token, freq in vocab._get_doc_freq().items() if freq}
    return vocab.get_codebase_doc_count(FACTS_CODEBASE_ID), doc_freq


async def _fact_points(indexer, fact_id):
    target = await active_embedding_collection(indexer.storage, indexer.logical_name)
    client = await indexer.storage.get_client()
    points, _ = await client.scroll(target, limit=1000, with_payload=True, with_vectors=True)
    return [point for point in points if (point.payload or {}).get("fact_id") == str(fact_id)]


def _fail_first_vocab_write(monkeypatch, vocab):
    """Fail only the first update_codebase_incremental call; delegate the rest.

    A mock that failed every call would also make an erroneous compensation
    fail, letting a test pass for the wrong reason.
    """
    real = vocab.update_codebase_incremental
    calls = []

    def side_effect(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return real(*args, **kwargs)

    update = MagicMock(side_effect=side_effect)
    monkeypatch.setattr(vocab, "update_codebase_incremental", update)
    return update


class TestSingleFactVocabulary:
    """index_fact / delete_fact_index keep the facts vocabulary contribution in
    step with the indexed facts after the first training (issue #33)."""

    async def test_repeated_updates_keep_doc_count_and_retire_dropped_terms(
        self, live, store, vocab
    ):
        fact = store.create("kestrel", "hunts", "voles", context="zebracorn")
        store.create("heron", "eats", "fish")
        await live.index_all(force=True)
        assert vocab.get_codebase_doc_count(FACTS_CODEBASE_ID) == 2

        for context in ("walrus", "narwhal", "manatee", "dugong", "beluga"):
            fact = store.update(fact.id, context=context)
            await live.index_fact(fact)

        assert vocab.get_codebase_doc_count(FACTS_CODEBASE_ID) == 2
        doc_freq = vocab._get_doc_freq()
        for retired in ("zebracorn", "walrus", "narwhal", "manatee", "dugong"):
            assert doc_freq.get(retired) == 0, retired
        assert doc_freq["beluga"] == 1
        # Terms in every revision are counted once, not once per edit.
        assert doc_freq["kestrel"] == 1

    async def test_new_term_reaches_the_sparse_vector_of_the_fact_that_introduced_it(
        self, live, store, vocab
    ):
        fact = store.create("kestrel", "hunts", "voles")
        store.create("heron", "eats", "fish")
        await live.index_all(force=True)
        assert "quokkaesque" not in vocab._get_vocab()

        fact = store.update(fact.id, context="quokkaesque")
        await live.index_fact(fact)

        assert vocab._get_doc_freq().get("quokkaesque") == 1
        (point,) = await _fact_points(live, fact.id)
        assert point.payload["content"] == generate_fact_text(fact)
        assert vocab._get_vocab()["quokkaesque"] in point.vector["sparse"].indices

    async def test_add_then_delete_round_trips_the_contribution(self, live, store, vocab):
        first = store.create("kestrel", "hunts", "voles")
        # An empty contribution is trained from the store, which already holds
        # the fact; no delta on top, or it would be counted twice.
        await live.index_fact(first)
        before = _facts_vocab(vocab)
        assert before[0] == 1

        added = store.create("axolotl", "lives_in", "lakes")
        await live.index_fact(added)
        doc_count, doc_freq = _facts_vocab(vocab)
        assert doc_count == 2
        assert doc_freq.get("axolotl") == 1

        await live.delete_fact_index(added.id)
        assert await _fact_points(live, added.id) == []
        assert _facts_vocab(vocab) == before
        assert vocab._get_doc_freq()["axolotl"] == 0

    async def test_deleting_an_unindexed_fact_changes_nothing(self, live, store, vocab):
        store.create("kestrel", "hunts", "voles")
        await live.index_all(force=True)
        unindexed = store.create("axolotl", "lives_in", "lakes")
        before = _facts_vocab(vocab)

        await live.delete_fact_index(unindexed.id)
        await live.delete_fact_index(uuid4())

        assert _facts_vocab(vocab) == before

    async def test_failed_delete_retires_nothing(self, live, store, vocab, monkeypatch):
        store.create("kestrel", "hunts", "voles")
        doomed = store.create("axolotl", "lives_in", "lakes")
        await live.index_all(force=True)
        before = _facts_vocab(vocab)
        monkeypatch.setattr(
            live.storage,
            "delete_by_filter",
            AsyncMock(side_effect=RuntimeError("delete failed")),
        )

        with pytest.raises(RuntimeError, match="delete failed"):
            await live.delete_fact_index(doomed.id)

        assert _facts_vocab(vocab) == before
        assert len(await _fact_points(live, doomed.id)) == 1

    @pytest.mark.parametrize("stage", ["embed", "upsert"])
    @pytest.mark.parametrize("existing", [True, False], ids=["update", "new"])
    async def test_failed_write_after_the_delta_compensates_exactly(
        self, live, vocab, monkeypatch, stage, existing
    ):
        store = live.fact_store
        fact = store.create("kestrel", "hunts", "voles", context="meadow")
        await live.index_all(force=True)
        if existing:
            fact = store.update(fact.id, context="quokkaesque")
        else:
            fact = store.create("axolotl", "lives_in", "lakes", context="quokkaesque")
        before = _facts_vocab(vocab)
        stored_before = [point.payload for point in await _fact_points(live, fact.id)]

        error: type[Exception]
        if stage == "embed":
            error = EmbeddingServiceError
            live.embedder.embed_all.side_effect = EmbeddingServiceError("embedding down")
        else:
            error = RuntimeError
            client = await live.storage.get_client()
            real_upsert = client.upsert
            calls = []

            async def upsert(*args, **kwargs):
                calls.append(args)
                if len(calls) == 1:
                    raise RuntimeError("upsert failed")
                return await real_upsert(*args, **kwargs)

            monkeypatch.setattr(client, "upsert", upsert)
        update = MagicMock(wraps=vocab.update_codebase_incremental)
        monkeypatch.setattr(vocab, "update_codebase_incremental", update)

        with pytest.raises(error):
            await live.index_fact(fact)

        # The forward delta reached the database and was then inverted.
        assert update.call_count == 2
        assert _facts_vocab(vocab) == before
        assert [point.payload for point in await _fact_points(live, fact.id)] == stored_before

    async def test_vocabulary_write_failure_still_indexes_the_fact(
        self, live, store, vocab, monkeypatch
    ):
        store.create("kestrel", "hunts", "voles")
        await live.index_all(force=True)
        before = _facts_vocab(vocab)
        update = _fail_first_vocab_write(monkeypatch, vocab)
        added = store.create("axolotl", "lives_in", "lakes")

        await live.index_fact(added)

        assert update.call_count == 1
        assert _facts_vocab(vocab) == before
        (point,) = await _fact_points(live, added.id)
        assert point.payload["content"] == generate_fact_text(added)

    async def test_unapplied_delta_is_never_compensated(self, live, store, vocab, monkeypatch):
        store.create("kestrel", "hunts", "voles")
        await live.index_all(force=True)
        before = _facts_vocab(vocab)
        update = _fail_first_vocab_write(monkeypatch, vocab)
        live.embedder.embed_all.side_effect = EmbeddingServiceError("embedding down")
        added = store.create("axolotl", "lives_in", "lakes")

        with pytest.raises(EmbeddingServiceError):
            await live.index_fact(added)

        # Later calls reach the real method, so inverting the delta that never
        # landed would subtract a document and tokens that were never added.
        assert update.call_count == 1
        assert _facts_vocab(vocab) == before

    async def test_point_without_stored_text_still_counts_as_indexed(self, live, store, vocab):
        fact = store.create("kestrel", "hunts", "voles")
        store.create("heron", "eats", "fish")
        await live.index_all(force=True)
        # A point predating the current payload layout: no readable fact text.
        target = await active_embedding_collection(live.storage, live.logical_name)
        client = await live.storage.get_client()
        await client.delete_payload(
            target,
            keys=["content", "embedding_text_field", "embedding_fragment"],
            points=[generate_point_id(f"fact:{fact.id}")],
        )

        fact = store.update(fact.id, context="orchard")
        await live.index_fact(fact)

        assert vocab.get_codebase_doc_count(FACTS_CODEBASE_ID) == 2
        assert vocab._get_doc_freq().get("orchard") == 1
        # Its unreadable old tokens stay counted until a force rebuild repairs them.
        await live.index_all(force=True)
        assert vocab._get_doc_freq()["kestrel"] == 1

    async def test_fragment_children_are_not_read_back_as_fact_text(self, store, qdrant, vocab):
        live = _live_indexer(store, qdrant, vocab, budget=64)
        fact = store.create("kestrel", "hunts", "voles", context="alpha " * 30 + "tailneedle")
        store.create("heron", "eats", "decoyterm")
        await live.index_all(force=True)
        children = [
            point
            for point in await _fact_points(live, fact.id)
            if is_derived_fragment(point.payload)
        ]
        assert children
        # A child holds a dense slice of the fact (which may split a word), not
        # its vocabulary document. Plant a term only another fact contributes.
        target = await active_embedding_collection(live.storage, live.logical_name)
        client = await live.storage.get_client()
        await client.set_payload(target, payload={"content": "decoyterm"}, points=[children[0].id])

        fact = store.update(fact.id, context="beta")
        await live.index_fact(fact)

        doc_freq = vocab._get_doc_freq()
        assert doc_freq["decoyterm"] == 1
        assert doc_freq["alpha"] == 0
        assert doc_freq.get("beta") == 1
