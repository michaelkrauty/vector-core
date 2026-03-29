"""Hybrid search with RRF (Reciprocal Rank Fusion)."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    Prefetch,
    ScoredPoint,
)
from qdrant_client.models import (
    SparseVector as QdrantSparseVector,
)

from vector_core.embeddings.sparse import SparseVector
from vector_core.settings import settings
from vector_core.storage.qdrant import QdrantStorage

# Threshold for considering dense/sparse weights as equal (enables fast path)
# Weights within 0.1% of each other use Qdrant's built-in RRF fusion
WEIGHT_EQUALITY_THRESHOLD = 0.001


@dataclass
class SearchResult:
    """Generic search result with score and payload."""

    id: int | str
    score: float
    payload: dict[str, Any]


class HybridSearcher:
    """
    Combines dense and sparse search results using RRF.

    Reciprocal Rank Fusion formula:
    score = dense_weight/(k + dense_rank) + sparse_weight/(k + sparse_rank)

    Features:
    - Configurable weights for dense vs sparse
    - Fast path using Qdrant's built-in RRF when weights are equal
    - Slow path with custom weighted RRF for unequal weights
    """

    def __init__(
        self,
        storage: QdrantStorage,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        rrf_k: int | None = None,
    ):
        """
        Initialize hybrid searcher.

        Args:
            storage: QdrantStorage instance
            dense_weight: Weight for dense (semantic) vectors. Default from settings.
            sparse_weight: Weight for sparse (TF-IDF) vectors. Default from settings.
            rrf_k: RRF constant (higher = smoother ranking). Default from settings.
        """
        self.storage = storage
        self.dense_weight = dense_weight if dense_weight is not None else settings.dense_weight
        self.sparse_weight = sparse_weight if sparse_weight is not None else settings.sparse_weight
        self.rrf_k = rrf_k if rrf_k is not None else settings.rrf_k

    def _weighted_rrf(
        self,
        dense_results: list[ScoredPoint],
        sparse_results: list[ScoredPoint],
        dense_weight: float,
        sparse_weight: float,
        k: int,
    ) -> list[tuple[ScoredPoint, float]]:
        """
        Compute weighted Reciprocal Rank Fusion.

        Args:
            dense_results: Results from dense vector search
            sparse_results: Results from sparse vector search
            dense_weight: Weight for dense results
            sparse_weight: Weight for sparse results
            k: RRF constant

        Returns:
            List of (point, rrf_score) tuples sorted by score descending
        """
        # Build rank maps (1-indexed ranks)
        # Cast point.id to int | str (we never use UUID in our code)
        dense_ranks: dict[int | str, int] = {}
        for rank, point in enumerate(dense_results, start=1):
            dense_ranks[cast(int | str, point.id)] = rank

        sparse_ranks: dict[int | str, int] = {}
        for rank, point in enumerate(sparse_results, start=1):
            sparse_ranks[cast(int | str, point.id)] = rank

        # Combine all unique points
        all_points: dict[int | str, ScoredPoint] = {}
        for point in dense_results + sparse_results:
            pid = cast(int | str, point.id)
            if pid not in all_points:
                all_points[pid] = point

        # Compute weighted RRF scores
        rrf_scores: list[tuple[ScoredPoint, float]] = []
        for point_id, point in all_points.items():
            score = 0.0
            if point_id in dense_ranks:
                score += dense_weight / (k + dense_ranks[point_id])
            if point_id in sparse_ranks:
                score += sparse_weight / (k + sparse_ranks[point_id])
            rrf_scores.append((point, score))

        # Sort by score descending
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        return rrf_scores

    async def search(
        self,
        collection: str,
        dense_query: list[float],
        sparse_query: SparseVector,
        limit: int = 10,
        prefetch_limit: int | None = None,
        filter_conditions: Sequence[FieldCondition] | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> list[SearchResult]:
        """
        Perform hybrid search with RRF fusion.

        Args:
            collection: Collection name
            dense_query: Dense query vector
            sparse_query: Sparse query vector
            limit: Max results to return
            prefetch_limit: How many candidates to fetch from each search type.
                           Higher = better quality but slower. Default from settings.
            filter_conditions: Optional filter conditions
            dense_weight: Override default dense weight
            sparse_weight: Override default sparse weight

        Returns:
            List of SearchResult ordered by relevance
        """
        client = await self.storage._get_client()
        prefetch_limit = prefetch_limit or settings.rrf_prefetch_limit
        dense_weight = dense_weight if dense_weight is not None else self.dense_weight
        sparse_weight = sparse_weight if sparse_weight is not None else self.sparse_weight

        query_filter = Filter(must=list(filter_conditions)) if filter_conditions else None

        # Fast path: If one weight is 0, skip that search entirely
        if dense_weight <= 0 and sparse_weight > 0:
            # Sparse-only search
            async with asyncio.timeout(settings.search_timeout):
                response = await client.query_points(
                    collection,
                    query=QdrantSparseVector(
                        indices=sparse_query.indices,
                        values=sparse_query.values,
                    ),
                    using="sparse",
                    limit=limit,
                    query_filter=query_filter,
                )
            return [
                SearchResult(
                    id=cast(int | str, p.id),
                    score=p.score or 0.0,
                    payload=dict(p.payload) if p.payload else {},
                )
                for p in response.points
            ]

        if sparse_weight <= 0 and dense_weight > 0:
            # Dense-only search
            async with asyncio.timeout(settings.search_timeout):
                response = await client.query_points(
                    collection,
                    query=dense_query,
                    using="dense",
                    limit=limit,
                    query_filter=query_filter,
                )
            return [
                SearchResult(
                    id=cast(int | str, p.id),
                    score=p.score or 0.0,
                    payload=dict(p.payload) if p.payload else {},
                )
                for p in response.points
            ]

        # If equal weights, use fast Qdrant built-in RRF fusion
        if abs(dense_weight - sparse_weight) < WEIGHT_EQUALITY_THRESHOLD:
            async with asyncio.timeout(settings.search_timeout):
                points = await client.query_points(
                    collection,
                    prefetch=[
                        Prefetch(
                            query=QdrantSparseVector(
                                indices=sparse_query.indices,
                                values=sparse_query.values,
                            ),
                            using="sparse",
                            limit=prefetch_limit,
                            filter=query_filter,
                        ),
                        Prefetch(
                            query=dense_query,
                            using="dense",
                            limit=prefetch_limit,
                            filter=query_filter,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=limit,
                )
            scored_points = [(p, p.score or 0.0) for p in points.points]
        else:
            # Run separate searches for weighted fusion
            async with asyncio.timeout(settings.search_timeout):
                dense_response, sparse_response = await asyncio.gather(
                    client.query_points(
                        collection,
                        query=dense_query,
                        using="dense",
                        limit=prefetch_limit,
                        query_filter=query_filter,
                    ),
                    client.query_points(
                        collection,
                        query=QdrantSparseVector(
                            indices=sparse_query.indices,
                            values=sparse_query.values,
                        ),
                        using="sparse",
                        limit=prefetch_limit,
                        query_filter=query_filter,
                    ),
                )

            # Apply weighted RRF
            scored_points = self._weighted_rrf(
                dense_response.points,
                sparse_response.points,
                dense_weight,
                sparse_weight,
                k=self.rrf_k,
            )[:limit]

        # Convert to SearchResult
        results = []
        for point, score in scored_points:
            results.append(SearchResult(
                id=cast(int | str, point.id),
                score=score,
                payload=dict(point.payload) if point.payload else {},
            ))

        return results

    def fuse_results(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        limit: int = 10,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> list[SearchResult]:
        """
        Fuse pre-computed search results using weighted RRF.

        This is useful when you've already retrieved results separately.

        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            limit: Max results to return
            dense_weight: Override default dense weight
            sparse_weight: Override default sparse weight

        Returns:
            Fused results ordered by RRF score
        """
        dense_weight = dense_weight if dense_weight is not None else self.dense_weight
        sparse_weight = sparse_weight if sparse_weight is not None else self.sparse_weight

        # Build rank maps (1-indexed)
        dense_ranks: dict[int | str, int] = {}
        dense_payloads: dict[int | str, dict] = {}
        for rank, result in enumerate(dense_results, start=1):
            dense_ranks[result.id] = rank
            dense_payloads[result.id] = result.payload

        sparse_ranks: dict[int | str, int] = {}
        sparse_payloads: dict[int | str, dict] = {}
        for rank, result in enumerate(sparse_results, start=1):
            sparse_ranks[result.id] = rank
            sparse_payloads[result.id] = result.payload

        # Combine all unique IDs
        all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        # Compute RRF scores
        fused: list[SearchResult] = []
        for result_id in all_ids:
            score = 0.0
            if result_id in dense_ranks:
                score += dense_weight / (self.rrf_k + dense_ranks[result_id])
            if result_id in sparse_ranks:
                score += sparse_weight / (self.rrf_k + sparse_ranks[result_id])

            # Use payload from whichever search found it
            payload = dense_payloads.get(result_id, sparse_payloads.get(result_id, {}))
            fused.append(SearchResult(id=result_id, score=score, payload=payload))

        # Sort by score descending
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:limit]
