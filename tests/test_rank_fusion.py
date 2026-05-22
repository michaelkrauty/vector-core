"""Tests for generic rank fusion utilities."""

from dataclasses import dataclass

from vector_core.search.rank_fusion import RankFusionResult, reciprocal_rank_fusion


@dataclass(frozen=True)
class Hit:
    """Small fake search hit for rank-fusion tests."""

    key: str
    label: str


def test_rrf_promotes_items_that_appear_in_multiple_ranked_lists() -> None:
    """RRF ranks repeated cross-list hits above single-list hits."""
    dense = [Hit("dense-only", "Dense only"), Hit("shared", "Shared from dense")]
    sparse = [Hit("shared", "Shared from sparse"), Hit("sparse-only", "Sparse only")]

    fused = reciprocal_rank_fusion([dense, sparse], key=lambda hit: hit.key, k=60, limit=3)

    assert [result.key for result in fused] == ["shared", "dense-only", "sparse-only"]
    assert all(isinstance(result, RankFusionResult) for result in fused)
    assert fused[0].item == dense[1]
    assert fused[0].ranks == {0: 2, 1: 1}


def test_rrf_supports_per_list_weights() -> None:
    """Weights let callers bias one retrieval source over another."""
    dense = [Hit("dense-top", "Dense top"), Hit("shared", "Shared")]
    sparse = [Hit("sparse-top", "Sparse top"), Hit("shared", "Shared")]

    fused = reciprocal_rank_fusion(
        [dense, sparse],
        key=lambda hit: hit.key,
        weights=[0.01, 1.0],
        k=60,
        limit=3,
    )

    assert [result.key for result in fused] == ["sparse-top", "shared", "dense-top"]


def test_rrf_tie_break_is_deterministic_by_best_rank_then_key() -> None:
    """Equal RRF scores use best rank and stable key ordering."""
    first = [Hit("b", "B"), Hit("a", "A")]
    second = [Hit("c", "C")]

    fused = reciprocal_rank_fusion([first, second], key=lambda hit: hit.key, k=60, limit=3)

    assert [result.key for result in fused] == ["b", "c", "a"]


def test_rrf_limit_is_clamped_to_at_least_one() -> None:
    """Limit 0 still returns the top item instead of surprising empty output."""
    fused = reciprocal_rank_fusion(
        [[Hit("a", "A"), Hit("b", "B")]], key=lambda hit: hit.key, limit=0
    )

    assert [result.key for result in fused] == ["a"]


def test_rrf_rejects_weight_count_mismatch() -> None:
    """Weight lists must match ranked-list count to avoid silent misranking."""
    try:
        reciprocal_rank_fusion(
            [[Hit("a", "A")], [Hit("b", "B")]], key=lambda hit: hit.key, weights=[1.0]
        )
    except ValueError as exc:
        assert "weights length" in str(exc)
    else:  # pragma: no cover - makes failure message clearer than pytest.raises import
        raise AssertionError("expected ValueError")


def test_rrf_deduplicates_keys_within_each_ranked_list() -> None:
    """A source list contributes only the first rank for each key."""
    ranked = [
        Hit("duplicate", "first occurrence"),
        Hit("duplicate", "second occurrence"),
        Hit("unique", "unique hit"),
    ]

    fused = reciprocal_rank_fusion([ranked], key=lambda hit: hit.key, k=60, limit=2)

    assert [result.key for result in fused] == ["duplicate", "unique"]
    assert fused[0].ranks == {0: 1}
    assert fused[0].score == 1 / 61
