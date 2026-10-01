"""Collection-scoped metadata dimensions against isolated real Qdrant storage."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from qdrant_client import AsyncQdrantClient

from vector_core.embeddings.client import EmbeddingClient
from vector_core.settings import settings
from vector_core.storage.embedding_migration import (
    active_embedding_collection,
    ensure_embedding_collection,
)
from vector_core.storage.qdrant import QdrantStorage


@pytest.fixture(params=[0, 2])
async def storage(request, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    result = QdrantStorage(url="http://isolated.invalid", embedding_dim=request.param)
    result._client = AsyncQdrantClient(location=":memory:")
    yield result
    await result.close()


def embedder(dimension):
    result = EmbeddingClient(model=f"dimension-{dimension}", dim=dimension)
    result.embed_single = AsyncMock(return_value=[1.0] + [0.0] * (dimension - 1))
    result.embed_all = AsyncMock(return_value=[])
    return result


async def metadata_point(storage, collection):
    points = await storage.retrieve_points(collection, [0], with_vectors=True)
    assert len(points) == 1
    return points[0]


async def test_shared_storage_metadata_uses_each_target_schema(storage):
    configured_dimension = storage.embedding_dim
    resolver = AsyncMock(side_effect=AssertionError("empty collections need no source"))
    first = await ensure_embedding_collection(storage, "first", embedder(3), resolver)
    second = await ensure_embedding_collection(storage, "second", embedder(5), resolver)

    await asyncio.gather(
        storage.store_metadata(first.physical_name, {"label": "first"}),
        storage.store_metadata(second.physical_name, {"label": "second"}),
    )

    for generation, dimension in [(first, 3), (second, 5)]:
        point = await metadata_point(storage, generation.physical_name)
        assert len(point.vector["dense"]) == dimension
        metadata = await storage.get_metadata(generation.physical_name)
        assert metadata["label"] == generation.logical_name
        assert metadata["embedding_generation"]["identity"]["dimension"] == dimension
        assert metadata["embedding_generation"]["state"] == "ready"
    assert storage.embedding_dim == configured_dimension


async def test_concurrent_finalizers_preserve_dimensions_and_sources(storage):
    configured_dimension = storage.embedding_dim
    sources = {}
    for logical_name in ["first", "second"]:
        await storage.create_collection(logical_name, dense_dim=2)
        await storage.store_metadata(logical_name, {"original": logical_name})
        sources[logical_name] = await metadata_point(storage, logical_name)

    arrived = 0
    both_finalizing = asyncio.Event()

    async def finalize(target):
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            both_finalizing.set()
        await both_finalizing.wait()
        await storage.store_metadata(target, {"finalized": True})

    resolver = AsyncMock(side_effect=AssertionError("metadata is not embedding source text"))
    async with asyncio.timeout(10):
        generations = await asyncio.gather(
            *(
                ensure_embedding_collection(
                    storage,
                    logical_name,
                    embedder(dimension),
                    resolver,
                    finalize_candidate=finalize,
                )
                for logical_name, dimension in [("first", 3), ("second", 5)]
            )
        )

    for generation, dimension in zip(generations, [3, 5], strict=True):
        assert (
            await metadata_point(storage, generation.logical_name)
            == sources[generation.logical_name]
        )
        assert await active_embedding_collection(storage, generation.logical_name) == (
            generation.physical_name
        )
        point = await metadata_point(storage, generation.physical_name)
        assert len(point.vector["dense"]) == dimension
        metadata = await storage.get_metadata(generation.physical_name)
        manifest = metadata["embedding_generation"]
        assert metadata["finalized"] is True
        assert manifest["state"] == "ready"
        assert manifest["identity"]["dimension"] == dimension
        with pytest.raises(ValueError, match="identity cannot be overwritten"):
            await storage.store_metadata(
                generation.physical_name, {"embedding_generation": {"state": "building"}}
            )
        assert await metadata_point(storage, generation.physical_name) == point
    assert storage.embedding_dim == configured_dimension
