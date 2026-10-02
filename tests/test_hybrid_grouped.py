"""Distinct-group retrieval and rank-based representative selection."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    GroupsResult,
    MatchValue,
    PointGroup,
    PointStruct,
    ScoredPoint,
    SparseVectorParams,
    VectorParams,
)

from vector_core.embeddings.sparse import SparseVector
from vector_core.storage.hybrid import HybridSearcher, settings


def group(key, point_id, score=1.0, **payload):
    return PointGroup(
        id=key,
        hits=[ScoredPoint(id=point_id, version=0, score=score, payload=payload)],
    )


def searcher_with(client, **kwargs):
    storage = Mock()
    storage._get_client = AsyncMock(return_value=client)
    return HybridSearcher(storage, **kwargs)


async def test_grouped_retrieval_avoids_chunk_starvation():
    """Even a document dominating point ranks cannot consume the group budget."""
    client = AsyncQdrantClient(location=":memory:")
    try:
        await client.create_collection(
            "chunks",
            vectors_config={"dense": VectorParams(size=2, distance=Distance.DOT)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        await client.upsert(
            "chunks",
            points=[
                PointStruct(id=i, vector={"dense": [100.0 - i, 1.0]}, payload={"file": "large"})
                for i in range(20)
            ]
            + [
                PointStruct(id=20, vector={"dense": [2.0, 1.0]}, payload={"file": "small"}),
                PointStruct(id=21, vector={"dense": [1.0, 1.0]}, payload={"file": "tiny"}),
            ],
        )
        searcher = searcher_with(client, dense_weight=1, sparse_weight=0)
        results = await searcher.search(
            "chunks",
            [1.0, 0.0],
            SparseVector(indices=[], values=[]),
            group_by="file",
            limit=3,
            prefetch_limit=1,
        )
        assert [result.payload["file"] for result in results] == ["large", "small", "tiny"]
        assert [result.id for result in results] == [0, 20, 21]
    finally:
        await client.close()


@pytest.mark.parametrize("weights", [(0.5, 0.5), (0.2, 0.8)])
async def test_group_fusion_and_best_representative(weights):
    dense = GroupsResult(
        groups=[
            group("a", 10, title="dense a"),
            group("b", 11, title="dense b"),
        ]
    )
    sparse = GroupsResult(
        groups=[
            group("b", 20, score=900, title="sparse b"),
            group("a", 21, score=1000, title="sparse a"),
        ]
    )
    client = Mock()
    client.query_points_groups = AsyncMock(side_effect=[dense, sparse])
    client.query_points = AsyncMock()
    searcher = searcher_with(client, dense_weight=weights[0], sparse_weight=weights[1])
    condition = FieldCondition(key="project", match=MatchValue(value="test"))
    results = await searcher.search(
        "chunks",
        [1.0],
        SparseVector(indices=[2], values=[3.0]),
        group_by="file",
        limit=2,
        prefetch_limit=5,
        filter_conditions=[condition],
    )
    assert len(results) == 2
    by_id = {result.id: result for result in results}
    assert by_id[20].payload == {"title": "sparse b"}
    assert by_id[20].score == pytest.approx(weights[0] / 62 + weights[1] / 61)
    a_id = 10 if weights[0] == weights[1] else 21
    assert by_id[a_id].score == pytest.approx(weights[0] / 61 + weights[1] / 62)
    assert by_id[a_id].payload["title"] == ("dense a" if a_id == 10 else "sparse a")
    if weights[1] > weights[0]:
        assert results[0].id == 20
    assert [call.kwargs["using"] for call in client.query_points_groups.await_args_list] == [
        "dense",
        "sparse",
    ]
    assert all(call.kwargs["limit"] == 5 for call in client.query_points_groups.await_args_list)
    assert all(
        call.kwargs["query_filter"].must == [condition]
        for call in client.query_points_groups.await_args_list
    )
    client.query_points.assert_not_called()


@pytest.mark.parametrize(
    "using,weights",
    [
        ("dense", (1.0, 0.0)),
        ("sparse", (0.0, 1.0)),
        ("sparse", (-1.0, 1.0)),
    ],
)
async def test_grouped_one_sided_filter_and_limit(using, weights):
    client = Mock()
    client.query_points_groups = AsyncMock(return_value=GroupsResult(groups=[group("a", 7)]))
    searcher = searcher_with(client, dense_weight=weights[0], sparse_weight=weights[1])
    condition = FieldCondition(key="project", match=MatchValue(value="test"))
    results = await searcher.search(
        "chunks",
        [1.0],
        SparseVector(indices=[2], values=[3.0]),
        group_by="file",
        limit=4,
        prefetch_limit=2,
        filter_conditions=[condition],
    )
    assert results[0].id == 7
    assert results[0].score == pytest.approx(1 / 61)
    client.query_points_groups.assert_awaited_once()
    call = client.query_points_groups.await_args
    assert call.args == ("chunks",)
    assert call.kwargs["using"] == using
    assert call.kwargs["group_by"] == "file"
    assert call.kwargs["group_size"] == 1
    assert call.kwargs["limit"] == 4
    assert call.kwargs["query_filter"].must == [condition]
    assert call.kwargs["with_payload"] is True
    if using == "dense":
        assert call.kwargs["query"] == [1.0]
    else:
        assert call.kwargs["query"].indices == [2]
        assert call.kwargs["query"].values == [3.0]


async def test_grouped_both_disabled_skip_queries():
    client = Mock()
    client.query_points_groups = AsyncMock()
    searcher = searcher_with(client, dense_weight=0, sparse_weight=0)
    assert (
        await searcher.search(
            "chunks",
            [],
            SparseVector(indices=[], values=[]),
            group_by="file",
        )
        == []
    )
    client.query_points_groups.assert_not_called()


async def test_grouped_timeout_cancels_branches_and_preserves_context(monkeypatch):
    monkeypatch.setattr(settings, "search_timeout", 0.01)
    cancelled = []

    async def blocked(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(kwargs["using"])

    client = Mock()
    client.query_points_groups = AsyncMock(side_effect=blocked)
    searcher = searcher_with(client, dense_weight=0.5, sparse_weight=0.5)
    with pytest.raises(TimeoutError, match="Grouped hybrid search.*group_by='file'"):
        await searcher.search(
            "chunks",
            [1.0],
            SparseVector(indices=[2], values=[3.0]),
            group_by="file",
        )
    assert sorted(cancelled) == ["dense", "sparse"]
