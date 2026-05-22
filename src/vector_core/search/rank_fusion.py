"""Generic ranked-list fusion utilities.

These helpers are intentionally independent of Qdrant or any specific search
result model so downstream projects can fuse dense, sparse, lexical, and other
ranked retrieval outputs without copying RRF implementations.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)


@dataclass(frozen=True)
class RankFusionResult(Generic[T, K]):
    """One fused ranked-list result."""

    key: K
    item: T
    score: float
    best_rank: int
    ranks: dict[int, int]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[T]],
    *,
    key: Callable[[T], K],
    limit: int = 10,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[RankFusionResult[T, K]]:
    """Fuse ranked result lists using weighted Reciprocal Rank Fusion.

    Args:
        ranked_lists: Ranked lists ordered from most to least relevant.
        key: Stable identity function used to deduplicate items across lists.
        limit: Maximum fused results to return. Clamped to at least one.
        k: RRF smoothing constant. Higher values make rank differences smoother.
        weights: Optional per-list weights. Defaults to ``1.0`` for each list.

    Returns:
        Fused results ordered by descending RRF score, then best rank, then key.
        The representative ``item`` is the first-seen item for that key.
    """

    if weights is None:
        normalized_weights = [1.0] * len(ranked_lists)
    else:
        normalized_weights = [float(weight) for weight in weights]
        if len(normalized_weights) != len(ranked_lists):
            raise ValueError("weights length must match ranked_lists length")

    rrf_k = max(0, int(k))
    scores: dict[K, float] = {}
    representatives: dict[K, T] = {}
    best_ranks: dict[K, int] = {}
    ranks_by_list: dict[K, dict[int, int]] = {}

    for list_index, ranked in enumerate(ranked_lists):
        weight = normalized_weights[list_index]
        if weight == 0:
            continue
        for rank, item in enumerate(ranked, start=1):
            item_key = key(item)
            scores[item_key] = scores.get(item_key, 0.0) + weight / (rrf_k + rank)
            representatives.setdefault(item_key, item)
            best_ranks[item_key] = min(best_ranks.get(item_key, rank), rank)
            ranks_by_list.setdefault(item_key, {})[list_index] = rank

    def sort_key(item_key: K) -> tuple[float, int, str]:
        return (-scores[item_key], best_ranks[item_key], str(item_key))

    fused: list[RankFusionResult[T, K]] = []
    for item_key in sorted(scores, key=sort_key)[: max(1, int(limit or 1))]:
        fused.append(
            RankFusionResult(
                key=item_key,
                item=representatives[item_key],
                score=scores[item_key],
                best_rank=best_ranks[item_key],
                ranks=dict(ranks_by_list[item_key]),
            )
        )
    return fused
