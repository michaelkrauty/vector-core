"""Complete-source dense coverage and reversible fragment generations, fully isolated."""

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.api.points_api import jsonable_encoder
from qdrant_client.models import PointsList, PointStruct, SparseVector, WriteOrdering

from vector_core.embeddings.client import EmbeddingClient
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.facts.database import FactStore
from vector_core.facts.indexer import FACTS_CODEBASE_ID, FactIndexer, generate_fact_text
from vector_core.glossary.indexer import GLOSSARY_CODEBASE_ID, GlossaryIndexer
from vector_core.glossary.store import GlossaryStore
from vector_core.settings import settings
from vector_core.storage import embedding_migration as migration
from vector_core.storage.embedding_fragments import (
    FRAGMENT_KEY,
    fragment_id,
    fragment_point,
    fragment_text,
    is_derived_fragment,
    source_hash,
    upsert_fragment_group,
)
from vector_core.storage.embedding_sources import stored_embedding_text
from vector_core.storage.qdrant import QdrantStorage, generate_point_id


@pytest.fixture
async def resources(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    monkeypatch.setattr(GlobalVocabulary, "_instance", None)
    storage = QdrantStorage(url="http://fragment-tests.invalid", embedding_dim=2)
    storage._client = AsyncQdrantClient(location=":memory:")
    vocab = GlobalVocabulary(db_path=tmp_path / "global_vocabulary.db")
    try:
        yield storage, vocab
    finally:
        await storage.close()
        vocab.close()


def embedder(model="A", budget=40):
    client = EmbeddingClient(model=model, dim=2, max_input_bytes=budget)
    client.embed_single = AsyncMock(return_value=[1.0, 0.0])
    client.embed_single_cached = AsyncMock(return_value=[1.0, 0.0])
    client.embed_all = AsyncMock(
        side_effect=lambda texts, **kwargs: [
            [0.0, 1.0] if "TAILNEEDLE" in text else [1.0, 0.0] for text in texts
        ]
    )
    return client


async def all_points(storage, collection):
    client = await storage.get_client()
    points, _ = await client.scroll(collection, limit=1000, with_payload=True, with_vectors=True)
    return {point.id: point for point in points}


async def seed(storage, raw):
    await storage.create_collection("corpus", dense_dim=2)
    client = await storage.get_client()
    await client.upsert(
        "corpus",
        [
            PointStruct(
                id=7,
                vector={"dense": [1.0, 0.0], "sparse": SparseVector(indices=[77], values=[2.0])},
                payload={
                    "type": "document",
                    "document_id": "source",
                    "content": raw,
                    "extra": {"kept": True},
                },
            )
        ],
        wait=True,
    )


async def resolve(payload):
    return payload["content"]


async def test_fragments_preserve_canonical_raw_and_sparse_with_real_snippet_vectors(resources):
    _, vocab = resources
    raw = "  alpha 🦊\n" * 12 + "TAILNEEDLE " + " omega " * 8
    vocab.register_codebase("source", [set(vocab.tokenize(raw))])
    sparse = SparseVector(indices=[999], values=[2.0])
    payload = {
        "type": "doc_chunk",
        "document_id": "doc",
        "filename": "source.txt",
        "path": "/source.txt",
        "doc_type": "text",
        "title": "Source",
        "tags": ["test"],
        "chunk_index": 4,
        "section_title": "Section",
        "char_start": 400,
        "char_end": 999,
        "content": raw,
        "body": raw,
        "unrelated": {"original": True},
    }
    original = deepcopy(payload)
    client = embedder()
    points = await fragment_point(client, point_id=7, payload=payload, sparse=sparse, text=raw)
    assert payload == original
    canonical = points[0]
    for key, value in original.items():
        assert canonical.payload[key] == value
    assert canonical.payload["embedding_text_field"] == "content"
    assert canonical.vector["sparse"] == sparse
    assert len(points) > 1
    assert "".join(fragment_text(point.payload) for point in points) == raw
    assert not is_derived_fragment(canonical.payload)
    for index, point in enumerate(points):
        marker = point.payload[FRAGMENT_KEY]
        assert marker["source_hash"] == source_hash(raw)
        assert marker["index"] == index and marker["count"] == len(points)
        if index:
            assert is_derived_fragment(point.payload)
            assert "body" not in point.payload and "unrelated" not in point.payload
            assert "embedding_text" not in point.payload
            assert point.payload["char_start"] == 400 + marker["start"]
            assert point.payload["char_end"] == 400 + marker["end"]
            assert point.payload["document_id"] == "doc"
            actual_sparse = vocab.vectorize_document(point.payload["content"])
            assert point.vector["sparse"] == SparseVector(
                indices=actual_sparse.indices,
                values=actual_sparse.values,
            )
            assert len(point.payload["content"].encode()) <= 40
    assert vocab.total_docs == 1


async def test_single_span_and_whitespace_have_markers_and_exact_retained_inputs(resources):
    storage, _ = resources
    await storage.create_collection("corpus", dense_dim=2)
    points = await fragment_point(
        embedder(),
        point_id=7,
        payload={"type": "note", "content": " \n\t"},
        text=" \n\t",
        sparse=SparseVector(indices=[], values=[]),
    )
    assert len(points) == 1
    assert points[0].payload[FRAGMENT_KEY]["count"] == 1
    assert stored_embedding_text(points[0].payload) == " \n\t"
    await upsert_fragment_group(storage, "corpus", points)
    assert fragment_text((await all_points(storage, "corpus"))[7].payload) == " \n\t"


async def test_A_B_A_regenerates_only_canonical_source_and_keeps_every_old_generation(resources):
    storage, vocab = resources
    raw = "alpha beta gamma " * 12 + "TAILNEEDLE "
    vocab.register_codebase("source", [set(vocab.tokenize(raw))])
    await seed(storage, raw)
    original = await all_points(storage, "corpus")
    snapshots = []
    for model, budget in (("A", 40), ("B", 65), ("A", 40)):
        client = embedder(model, budget)
        generation = await migration.ensure_embedding_collection(storage, "corpus", client, resolve)
        current = await all_points(storage, generation.physical_name)
        group = [point for point in current.values() if point.id != 0]
        assert len(group) == len(client.split_text(raw))
        assert current[7].payload["content"] == raw
        assert current[7].vector["sparse"] == original[7].vector["sparse"]
        assert client.embed_all.await_count == 1
        assert "".join(client.embed_all.await_args.args[0]) == raw
        assert all(point.payload[FRAGMENT_KEY]["parent_id"] == 7 for point in group)
        snapshots.append((generation, current))
    assert snapshots[0][0].physical_name != snapshots[2][0].physical_name
    assert set(snapshots[0][1]) == set(snapshots[2][1])
    for generation, snapshot in snapshots:
        assert await all_points(storage, generation.physical_name) == snapshot
    assert await all_points(storage, "corpus") == original
    assert vocab.total_docs == 1
    # The tail has its own dense hit, not just a preserved but unembedded payload.
    raw_client = await storage.get_client()
    hits = await raw_client.query_points(
        snapshots[-1][0].physical_name,
        query=[0.0, 1.0],
        using="dense",
        limit=1,
    )
    assert "TAILNEEDLE" in fragment_text(hits.points[0].payload)


@pytest.mark.parametrize("damage", ["orphan", "hash", "id", "malformed"])
async def test_forged_or_orphaned_children_never_publish(resources, damage):
    storage, vocab = resources
    raw = "alpha beta gamma " * 6
    vocab.register_codebase("source", [set(vocab.tokenize(raw))])
    await seed(storage, raw)
    first = await migration.ensure_embedding_collection(storage, "corpus", embedder(), resolve)
    client = await storage.get_client()
    current = await all_points(storage, first.physical_name)
    child = next(point for point in current.values() if is_derived_fragment(point.payload))
    if damage == "orphan":
        await client.delete(first.physical_name, points_selector=[7], wait=True)
    elif damage == "hash":
        marker = {**child.payload[FRAGMENT_KEY], "source_hash": "0" * 64}
        await client.set_payload(first.physical_name, {FRAGMENT_KEY: marker}, points=[child.id])
    elif damage == "malformed":
        await client.set_payload(
            first.physical_name, {FRAGMENT_KEY: {"index": 1}}, points=[child.id]
        )
    else:
        await client.upsert(
            first.physical_name,
            [
                PointStruct(
                    id=8,
                    vector=child.vector,
                    payload=child.payload,
                )
            ],
            wait=True,
        )
    before = await all_points(storage, first.physical_name)
    with pytest.raises(migration.EmbeddingMigrationError, match="fragment"):
        await migration.ensure_embedding_collection(storage, "corpus", embedder("B"), resolve)
    assert await migration.active_embedding_collection(storage, "corpus") == first.physical_name
    assert await all_points(storage, first.physical_name) == before


async def test_generated_child_id_collision_cannot_overwrite_existing_source(resources):
    storage, vocab = resources
    raw = "alpha beta gamma " * 6
    vocab.register_codebase("source", [set(vocab.tokenize(raw))])
    await seed(storage, raw)
    span = embedder().split_text(raw)[1]
    collision_id = fragment_id(7, source_hash(raw), span.start, span.end, 1)
    client = await storage.get_client()
    await client.upsert(
        "corpus",
        [
            PointStruct(
                id=collision_id,
                vector={"dense": [1.0, 0.0], "sparse": SparseVector(indices=[], values=[])},
                payload={"type": "document", "content": "another real source"},
            )
        ],
        wait=True,
    )
    before = await all_points(storage, "corpus")
    with pytest.raises(migration.EmbeddingMigrationError, match="collides with source"):
        await migration.ensure_embedding_collection(storage, "corpus", embedder(), resolve)
    assert await migration.active_embedding_collection(storage, "corpus") == "corpus"
    assert await all_points(storage, "corpus") == before


async def test_normal_update_uses_byte_bounded_strong_writes_then_retires_only_its_stale_children(
    resources,
    monkeypatch,
):
    storage, vocab = resources
    await storage.create_collection("corpus", dense_dim=2)
    raw = "alpha beta gamma " * 400
    vocab.register_codebase("source", [set(vocab.tokenize(raw))])
    sparse = vocab.vectorize_document(raw)
    old = await fragment_point(
        embedder(), point_id=7, payload={"content": raw}, sparse=sparse, text=raw
    )
    other = await fragment_point(
        embedder(), point_id=8, payload={"content": raw}, sparse=sparse, text=raw
    )
    await upsert_fragment_group(storage, "corpus", old)
    await upsert_fragment_group(storage, "corpus", other)
    updated = await fragment_point(
        embedder(), point_id=7, payload={"content": "alpha"}, sparse=sparse, text="alpha"
    )
    budget = max(len(jsonable_encoder(PointsList(points=[point])).encode()) for point in updated)
    monkeypatch.setattr(migration, "_MAX_UPSERT_BYTES", budget)
    client = await storage.get_client()
    upsert = AsyncMock(wraps=client.upsert)
    deletion = AsyncMock(wraps=client.delete)
    monkeypatch.setattr(client, "upsert", upsert)
    monkeypatch.setattr(client, "delete", deletion)
    await upsert_fragment_group(storage, "corpus", updated)
    after = await all_points(storage, "corpus")
    assert set(after) == {7, *(point.id for point in other)}
    assert after[7].payload["content"] == "alpha"
    deleted = []
    for call in deletion.await_args_list:
        selector = call.kwargs["points_selector"]
        deleted.extend(selector.points)
        assert call.kwargs["wait"] is True and call.kwargs["ordering"] == WriteOrdering.STRONG
        assert len(jsonable_encoder(selector).encode()) <= budget
    assert set(deleted) == {point.id for point in old[1:]}
    for call in upsert.await_args_list:
        assert call.kwargs["wait"] is True and call.kwargs["ordering"] == WriteOrdering.STRONG
        assert len(jsonable_encoder(PointsList(points=call.args[1])).encode()) <= budget


async def test_failed_normal_group_upsert_never_prunes_old_children(resources, monkeypatch):
    storage, vocab = resources
    await storage.create_collection("corpus", dense_dim=2)
    raw = "alpha beta gamma " * 12
    vocab.register_codebase("source", [set(vocab.tokenize(raw))])
    points = await fragment_point(
        embedder(),
        point_id=7,
        payload={"content": raw},
        sparse=vocab.vectorize_document(raw),
        text=raw,
    )
    await upsert_fragment_group(storage, "corpus", points)
    before = await all_points(storage, "corpus")
    client = await storage.get_client()
    monkeypatch.setattr(client, "upsert", AsyncMock(side_effect=RuntimeError("write rejected")))
    deletion = AsyncMock()
    monkeypatch.setattr(client, "delete", deletion)
    updated = await fragment_point(
        embedder(),
        point_id=7,
        payload={"content": "alpha"},
        sparse=vocab.vectorize_document("alpha"),
        text="alpha",
    )
    with pytest.raises(RuntimeError, match="write rejected"):
        await upsert_fragment_group(storage, "corpus", updated)
    deletion.assert_not_awaited()
    assert await all_points(storage, "corpus") == before


async def test_partial_migration_transport_cannot_hide_behind_finalizer(resources, monkeypatch):
    storage, vocab = resources
    raw = "alpha beta gamma " * 6
    vocab.register_codebase("source", [set(vocab.tokenize(raw))])
    original_vocabulary = (vocab.vocab_size, vocab.total_docs, dict(vocab._get_doc_freq()))
    await seed(storage, raw)
    writer = migration._upsert_copy_points

    async def drop_tail(client, collection, points):
        await writer(client, collection, points[:-1])

    monkeypatch.setattr(migration, "_upsert_copy_points", drop_tail)
    finalizer = AsyncMock()
    with pytest.raises(migration.EmbeddingMigrationError, match="point count"):
        await migration.ensure_embedding_collection(
            storage,
            "corpus",
            embedder(),
            resolve,
            finalize_candidate=finalizer,
        )
    finalizer.assert_not_awaited()
    assert await migration.active_embedding_collection(storage, "corpus") == "corpus"
    assert (vocab.vocab_size, vocab.total_docs, dict(vocab._get_doc_freq())) == original_vocabulary


async def test_finalizer_cannot_publish_an_incomplete_fragment_group(resources):
    storage, vocab = resources
    raw = "alpha beta gamma " * 6
    vocab.register_codebase("source", [set(vocab.tokenize(raw))])
    await seed(storage, raw)

    async def finalize(target):
        points = await all_points(storage, target)
        child = next(point for point in points.values() if is_derived_fragment(point.payload))
        client = await storage.get_client()
        await client.delete(target, points_selector=[child.id], wait=True)

    with pytest.raises(migration.EmbeddingMigrationError, match="group is incomplete"):
        await migration.ensure_embedding_collection(
            storage,
            "corpus",
            embedder(),
            resolve,
            finalize_candidate=finalize,
        )
    assert await migration.active_embedding_collection(storage, "corpus") == "corpus"


async def test_fact_and_glossary_writers_keep_entity_counts_and_cover_tail(
    resources,
    tmp_path,
    monkeypatch,
):
    storage, vocab = resources
    facts = FactStore(db_path=tmp_path / "facts.db")
    glossary = GlossaryStore(db_path=tmp_path / "glossary.db")
    try:
        fact = facts.create("subject", "relates_to", "object", context="alpha " * 30 + "TAILNEEDLE")
        fact_indexer = FactIndexer(
            facts, storage, embedder(budget=64), vocab, collection_name="shared"
        )
        await fact_indexer.index_fact(fact)
        entry = glossary.create("TERM", "Expansion", "alpha " * 400 + "TAILNEEDLE", "test")
        glossary_indexer = GlossaryIndexer("shared", glossary, storage, embedder(budget=64), vocab)
        assert await glossary_indexer.index_all() == 1
        target = await migration.active_embedding_collection(storage, "shared")
        points = await all_points(storage, target)
        fact_id = generate_point_id(f"fact:{fact.id}")
        assert stored_embedding_text(points[fact_id].payload) == generate_fact_text(fact)
        assert vocab.get_codebase_doc_count(FACTS_CODEBASE_ID) == 1
        assert vocab.get_codebase_doc_count(GLOSSARY_CODEBASE_ID) == 1
        for field, value in (("fact_id", str(fact.id)), ("glossary_id", str(entry.id))):
            group = [point for point in points.values() if point.payload.get(field) == value]
            assert len(group) > 1
            assert any("TAILNEEDLE" in fragment_text(point.payload) for point in group)
            assert any(point.vector["dense"] == [0.0, 1.0] for point in group)
        glossary_indexer.embedder.embed_single_cached = AsyncMock(return_value=[0.0, 1.0])
        raw_client = await storage.get_client()
        retrieve = AsyncMock(wraps=raw_client.retrieve)
        monkeypatch.setattr(raw_client, "retrieve", retrieve)
        results = await glossary_indexer.search("TAILNEEDLE", limit=1)
        assert len(results) == 1
        assert results[0]["glossary_id"] == str(entry.id)
        assert results[0]["expansion"] == entry.expansion
        assert results[0]["definition"] == entry.definition[:2000]
        assert "TAILNEEDLE" not in results[0]["definition"]
        assert "TAILNEEDLE" in results[0]["content"]
        assert is_derived_fragment(results[0])
        assert fragment_text(results[0]) == results[0]["content"]
        parent_id = generate_point_id(f"glossary:{entry.id}")
        hydration = [
            call for call in retrieve.await_args_list if call.kwargs.get("ids") == [parent_id]
        ]
        assert len(hydration) == 1
        assert set(hydration[0].kwargs["with_payload"]) == {
            "type",
            "glossary_id",
            "expansion",
            "definition",
            FRAGMENT_KEY,
        }
        for point in points.values():
            if point.payload.get("glossary_id") == str(entry.id) and is_derived_fragment(
                point.payload
            ):
                assert "expansion" not in point.payload
                assert "definition" not in point.payload
        await fact_indexer.delete_fact_index(fact.id)
        await glossary_indexer.delete_entry_index(entry.id)
        assert set(await all_points(storage, target)) == {0}
    finally:
        facts.close()
        glossary.close()


@pytest.mark.parametrize("damage", ["missing", "changed_source"])
async def test_glossary_tail_hydration_rejects_missing_or_changed_parent(
    resources, tmp_path, damage
):
    storage, vocab = resources
    store = GlossaryStore(db_path=tmp_path / "glossary.db")
    try:
        entry = store.create("TERM", "Expansion", "alpha " * 400 + "TAILNEEDLE", "test")
        indexer = GlossaryIndexer("glossary", store, storage, embedder(budget=64), vocab)
        await indexer.index_all()
        indexer.embedder.embed_single_cached = AsyncMock(return_value=[0.0, 1.0])
        target = await migration.active_embedding_collection(storage, "glossary")
        client = await storage.get_client()
        parent_id = generate_point_id(f"glossary:{entry.id}")
        if damage == "missing":
            await client.delete(target, points_selector=[parent_id], wait=True)
        else:
            parent = (await client.retrieve(target, ids=[parent_id], with_payload=True))[0]
            marker = {**parent.payload[FRAGMENT_KEY], "source_hash": "0" * 64}
            await client.set_payload(target, payload={FRAGMENT_KEY: marker}, points=[parent_id])
        with pytest.raises(ValueError, match="matching canonical source"):
            await indexer.search("TAILNEEDLE", limit=1)
    finally:
        store.close()


async def test_large_fragment_group_reads_and_hashes_parent_only_once_per_migration(
    resources,
    monkeypatch,
):
    storage, vocab = resources
    raw = "alpha beta " * ((24 * 1024 * 1024) // 11) + "TAILNEEDLE"
    vocab.register_codebase("source", [{"alpha", "beta", "tailneedle"}])
    await seed(storage, raw)
    first = await migration.ensure_embedding_collection(
        storage, "corpus", embedder(budget=65536), resolve
    )
    raw_client = await storage.get_client()
    retrieve = AsyncMock(wraps=raw_client.retrieve)
    monkeypatch.setattr(raw_client, "retrieve", retrieve)
    hash_calls = []
    original_hash = migration.source_hash

    def count_hash(text):
        hash_calls.append(len(text))
        return original_hash(text)

    monkeypatch.setattr(migration, "source_hash", count_hash)
    second = await migration.ensure_embedding_collection(
        storage,
        "corpus",
        embedder("B", budget=65536),
        resolve,
    )
    assert hash_calls == [len(raw)]
    parent_reads = [
        call
        for call in retrieve.await_args_list
        if call.args[0] == first.physical_name and call.kwargs.get("ids") == [7]
    ]
    assert len(parent_reads) == 1
    assert parent_reads[0].kwargs["with_payload"] == [FRAGMENT_KEY]
    points = await all_points(storage, second.physical_name)
    assert len(points) > 300
    assert stored_embedding_text(points[7].payload) == raw
    assert all(
        len(point.payload["content"]) <= 65536
        for point in points.values()
        if is_derived_fragment(point.payload)
    )


async def test_duplicate_ordinals_and_gap_ranges_fail_even_when_group_count_matches(resources):
    storage, vocab = resources
    raw = "alpha beta gamma " * 6
    vocab.register_codebase("source", [set(vocab.tokenize(raw))])
    await seed(storage, raw)
    first = await migration.ensure_embedding_collection(storage, "corpus", embedder(), resolve)
    client = await storage.get_client()
    points = await all_points(storage, first.physical_name)
    children = sorted(
        (point for point in points.values() if is_derived_fragment(point.payload)),
        key=lambda point: point.payload[FRAGMENT_KEY]["index"],
    )
    first_child, second_child = children[:2]
    old_marker = second_child.payload[FRAGMENT_KEY]
    marker = {**old_marker, "index": first_child.payload[FRAGMENT_KEY]["index"]}
    duplicate_id = fragment_id(7, source_hash(raw), marker["start"], marker["end"], marker["index"])
    await client.delete(first.physical_name, points_selector=[second_child.id], wait=True)
    await client.upsert(
        first.physical_name,
        [
            PointStruct(
                id=duplicate_id,
                vector=second_child.vector,
                payload={**second_child.payload, FRAGMENT_KEY: marker},
            )
        ],
        wait=True,
    )
    with pytest.raises(migration.EmbeddingMigrationError, match="invalid parent lineage"):
        await migration.ensure_embedding_collection(storage, "corpus", embedder("B"), resolve)
    # Restore a distinct ordinal, but leave a gap in the source partition.
    await client.delete(first.physical_name, points_selector=[duplicate_id], wait=True)
    marker = {**old_marker, "start": old_marker["start"] + 1}
    gap_id = fragment_id(7, source_hash(raw), marker["start"], marker["end"], marker["index"])
    await client.upsert(
        first.physical_name,
        [
            PointStruct(
                id=gap_id,
                vector=second_child.vector,
                payload={
                    **second_child.payload,
                    "content": raw[marker["start"] : marker["end"]],
                    FRAGMENT_KEY: marker,
                },
            )
        ],
        wait=True,
    )
    with pytest.raises(migration.EmbeddingMigrationError, match="leave source uncovered"):
        await migration.ensure_embedding_collection(storage, "corpus", embedder("B"), resolve)
    assert await migration.active_embedding_collection(storage, "corpus") == first.physical_name


async def test_plain_singleton_write_adds_marker_and_retires_old_children(resources):
    storage, vocab = resources
    await storage.create_collection("corpus", dense_dim=2)
    raw = "alpha beta gamma " * 6
    vocab.register_codebase("source", [set(vocab.tokenize(raw))])
    group = await fragment_point(
        embedder(),
        point_id=7,
        payload={"content": raw},
        text=raw,
        sparse=vocab.vectorize_document(raw),
    )
    await upsert_fragment_group(storage, "corpus", group)
    payload = {"type": "note", "note_id": "entity", "content": "alpha"}
    original = deepcopy(payload)
    point = PointStruct(
        id=7,
        payload=payload,
        vector={"dense": [0.0, 1.0], "sparse": SparseVector(indices=[77], values=[2.0])},
    )
    await upsert_fragment_group(storage, "corpus", [point])
    points = await all_points(storage, "corpus")
    assert set(points) == {7}
    assert payload == original and point.payload == original
    assert points[7].payload[FRAGMENT_KEY]["count"] == 1
    assert points[7].vector == point.vector
    for key, value in original.items():
        assert points[7].payload[key] == value


async def test_public_group_writer_rejects_forged_child_before_any_write(resources, monkeypatch):
    storage, vocab = resources
    await storage.create_collection("corpus", dense_dim=2)
    raw = "alpha beta gamma " * 6
    vocab.register_codebase("source", [set(vocab.tokenize(raw))])
    group = await fragment_point(
        embedder(),
        point_id=7,
        payload={"content": raw},
        text=raw,
        sparse=vocab.vectorize_document(raw),
    )
    child = group[1]
    marker = {**child.payload[FRAGMENT_KEY], "parent_id": 99}
    group[1] = child.model_copy(update={"payload": {**child.payload, FRAGMENT_KEY: marker}})
    client = await storage.get_client()
    upsert = AsyncMock()
    deletion = AsyncMock()
    monkeypatch.setattr(client, "upsert", upsert)
    monkeypatch.setattr(client, "delete", deletion)
    with pytest.raises(ValueError, match="group lineage"):
        await upsert_fragment_group(storage, "corpus", group)
    upsert.assert_not_awaited()
    deletion.assert_not_awaited()
