"""Shared indexers operate on isolated real Qdrant generations."""

from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, SparseVector

from vector_core.embeddings.client import EmbeddingClient
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.facts.database import FactStore
from vector_core.facts.indexer import FactIndexer, generate_fact_text
from vector_core.glossary.indexer import GlossaryIndexer, _generate_embedding_content
from vector_core.glossary.store import GlossaryStore
from vector_core.glossary.tools import GlossaryToolHelper
from vector_core.settings import settings
from vector_core.storage.embedding_migration import (
    EmbeddingMigrationError,
    active_embedding_collection,
)
from vector_core.storage.qdrant import QdrantStorage, generate_point_id


@pytest.fixture
async def resources(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    storage = QdrantStorage(url="http://shared-test.invalid", embedding_dim=2)
    storage._client = AsyncQdrantClient(location=":memory:")
    glossary = GlossaryStore(db_path=tmp_path / "glossary.db")
    facts = FactStore(db_path=tmp_path / "facts.db")
    vocab = GlobalVocabulary(db_path=tmp_path / "vocab.db")
    try:
        yield storage, glossary, facts, vocab
    finally:
        await storage.close()
        glossary.close()
        facts.close()
        vocab.close()


def embedder(model="new"):
    client = EmbeddingClient(
        base_url="http://embedding-test.invalid", model=model, dim=3, cache_namespace="revision"
    )
    client.embed_single = AsyncMock(return_value=[1.0, 0.0, 0.0])
    client.embed_single_cached = AsyncMock(return_value=[1.0, 0.0, 0.0])
    client.embed_all = AsyncMock(
        side_effect=lambda texts, **kwargs: [[1.0, 0.0, 0.0] for _ in texts]
    )
    return client


async def seed(storage, payload, *, point_id: int | str = 1):
    await storage.create_collection("shared", dense_dim=2)
    raw = await storage.get_client()
    await raw.upsert(
        "shared",
        [
            PointStruct(
                id=point_id,
                vector={"dense": [1.0, 0.0], "sparse": SparseVector(indices=[7], values=[2.0])},
                payload=payload,
            )
        ],
        wait=True,
    )


async def points(storage, collection):
    raw = await storage.get_client()
    records, _ = await raw.scroll(collection, with_payload=True, with_vectors=True)
    return {record.id: record for record in records}


async def test_glossary_search_migrates_exact_long_input_and_retains_source(resources):
    storage, glossary, _, vocab = resources
    entry = glossary.create("API", "Interface", "full " * 1000, "software", ["First", "Second"])
    payload = GlossaryIndexer._create_payload(entry)
    del payload["embedding_text"]
    await seed(storage, payload)
    before = await points(storage, "shared")
    client = embedder()
    indexer = GlossaryIndexer("shared", glossary, storage, client, vocab)
    indexer.hybrid_searcher.search = AsyncMock(return_value=[])

    await indexer.search("interface")

    target = await active_embedding_collection(storage, "shared")
    migrated = await points(storage, target)
    assert target != "shared"
    assert indexer.hybrid_searcher.search.call_args.kwargs["collection"] == target
    assert await points(storage, "shared") == before
    assert migrated[1].payload["embedding_text"] == _generate_embedding_content(entry)
    assert migrated[1].vector["sparse"] == before[1].vector["sparse"]
    client.embed_all.assert_awaited_once_with([_generate_embedding_content(entry)], role="document")


async def test_mismatched_glossary_row_does_not_publish_or_write(resources):
    storage, glossary, _, vocab = resources
    entry = glossary.create("API", "Interface", "definition", "software", ["First"])
    payload = GlossaryIndexer._create_payload(entry)
    del payload["embedding_text"]
    payload["aliases"] = ["stale"]
    await seed(storage, payload)
    before = await points(storage, "shared")
    indexer = GlossaryIndexer("shared", glossary, storage, embedder(), vocab)

    with pytest.raises(EmbeddingMigrationError, match="does not match"):
        await indexer.index_entry(entry.id)

    assert await active_embedding_collection(storage, "shared") == "shared"
    assert await points(storage, "shared") == before


async def test_fact_write_preserves_legacy_and_other_shared_types(resources):
    storage, _, facts, vocab = resources
    payload = {"type": "document", "content": "retained document", "extra": {"preserved": True}}
    await seed(storage, payload)
    before = await points(storage, "shared")
    fact = facts.create("subject", "relates_to", "object", context="exact " * 1000)
    client = embedder()
    indexer = FactIndexer(facts, storage, client, vocab, collection_name="shared")

    await indexer.index_fact(fact)

    target = await active_embedding_collection(storage, "shared")
    current = await points(storage, target)
    fact_id = generate_point_id(f"fact:{fact.id}")
    assert target != "shared"
    assert await points(storage, "shared") == before
    assert current[1].payload == {
        **payload,
        "embedding_text": payload["content"],
        "embedding_text_source": "legacy-reconstruction",
    }
    assert current[1].vector["sparse"] == before[1].vector["sparse"]
    assert current[fact_id].payload["embedding_text"] == generate_fact_text(fact)
    await indexer.delete_fact_index(fact.id)
    assert fact_id not in await points(storage, target)
    assert await points(storage, "shared") == before


async def test_old_indexer_cannot_write_after_space_is_superseded(resources):
    storage, _, facts, vocab = resources
    fact = facts.create("subject", "relates_to", "object")
    first = FactIndexer(facts, storage, embedder("first"), vocab, collection_name="shared")
    await first.index_fact(fact)
    second = FactIndexer(facts, storage, embedder("second"), vocab, collection_name="shared")
    await second.ensure_collection()
    target = await active_embedding_collection(storage, "shared")
    before = await points(storage, target)

    with pytest.raises(EmbeddingMigrationError, match="superseded"):
        await first.delete_fact_index(fact.id)

    assert await points(storage, target) == before


async def test_glossary_tool_updates_only_after_recovering_legacy_row(resources):
    storage, glossary, _, vocab = resources
    entry = glossary.create("API", "Interface", "old " * 1000, "software", ["First"])
    payload = GlossaryIndexer._create_payload(entry)
    del payload["embedding_text"]
    point_id = generate_point_id(f"glossary:{entry.id}")
    await seed(storage, payload, point_id=point_id)
    before = await points(storage, "shared")
    client = embedder()
    helper = GlossaryToolHelper(
        glossary, GlossaryIndexer("shared", glossary, storage, client, vocab)
    )

    result = await helper.update_entry("API", definition="updated definition", aliases=["Second"])

    assert result["definition"] == "updated definition"
    target = await active_embedding_collection(storage, "shared")
    current = await points(storage, target)
    assert current[point_id].payload["embedding_text"] == _generate_embedding_content(
        glossary.read(entry.id)
    )
    client.embed_all.assert_awaited_once_with([_generate_embedding_content(entry)], role="document")
    assert await points(storage, "shared") == before


async def test_glossary_tool_leaves_sqlite_unchanged_when_migration_fails(resources):
    storage, glossary, _, vocab = resources
    entry = glossary.create("API", "Interface", "definition", "software", ["First"])
    payload = GlossaryIndexer._create_payload(entry)
    del payload["embedding_text"]
    payload["domain"] = "stale"
    await seed(storage, payload)
    helper = GlossaryToolHelper(
        glossary, GlossaryIndexer("shared", glossary, storage, embedder(), vocab)
    )

    with pytest.raises(EmbeddingMigrationError, match="does not match"):
        await helper.update_entry("API", definition="changed")

    assert glossary.read(entry.id) == entry
    with pytest.raises(EmbeddingMigrationError, match="does not match"):
        await helper.delete_entry("API")
    assert glossary.read(entry.id) == entry
