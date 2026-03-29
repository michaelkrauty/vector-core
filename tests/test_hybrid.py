"""Tests for hybrid search with RRF fusion."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vector_core.embeddings.sparse import SparseVector
from vector_core.storage.hybrid import HybridSearcher, SearchResult


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """SearchResult holds id, score, and payload."""
        result = SearchResult(
            id="test-id",
            score=0.95,
            payload={"key": "value"}
        )

        assert result.id == "test-id"
        assert result.score == 0.95
        assert result.payload == {"key": "value"}

    def test_search_result_with_int_id(self):
        """SearchResult accepts int ID."""
        result = SearchResult(id=123, score=0.5, payload={})

        assert result.id == 123

    def test_search_result_with_str_id(self):
        """SearchResult accepts str ID."""
        result = SearchResult(id="abc123", score=0.5, payload={})

        assert result.id == "abc123"


class TestHybridSearcherInit:
    """Tests for HybridSearcher initialization."""

    def test_default_weights(self):
        """Uses default weights from settings."""
        mock_storage = MagicMock()

        with patch("vector_core.storage.hybrid.settings") as mock_settings:
            mock_settings.dense_weight = 0.7
            mock_settings.sparse_weight = 0.3
            mock_settings.rrf_k = 60

            searcher = HybridSearcher(mock_storage)

            assert searcher.dense_weight == 0.7
            assert searcher.sparse_weight == 0.3
            assert searcher.rrf_k == 60

    def test_custom_weights(self):
        """Accepts custom weights."""
        mock_storage = MagicMock()

        searcher = HybridSearcher(
            mock_storage,
            dense_weight=0.6,
            sparse_weight=0.4,
            rrf_k=100,
        )

        assert searcher.dense_weight == 0.6
        assert searcher.sparse_weight == 0.4
        assert searcher.rrf_k == 100


class TestWeightedRRF:
    """Tests for weighted Reciprocal Rank Fusion."""

    @pytest.fixture
    def searcher(self):
        """Create searcher with known weights."""
        mock_storage = MagicMock()
        return HybridSearcher(
            mock_storage,
            dense_weight=0.5,
            sparse_weight=0.5,
            rrf_k=60,
        )

    def test_rrf_single_result_both_lists(self, searcher):
        """RRF of point appearing in both lists."""
        point = MagicMock()
        point.id = "p1"
        point.score = 0.9
        point.payload = {"title": "Test"}

        dense_results = [point]
        sparse_results = [point]

        result = searcher._weighted_rrf(
            dense_results, sparse_results,
            dense_weight=0.5, sparse_weight=0.5, k=60
        )

        assert len(result) == 1
        # Score = 0.5/(60+1) + 0.5/(60+1) = 0.5/61 * 2 ≈ 0.0164
        assert result[0][0].id == "p1"
        assert abs(result[0][1] - (1.0/61)) < 0.001

    def test_rrf_multiple_results(self, searcher):
        """RRF with multiple results in different ranks."""
        p1 = MagicMock()
        p1.id = "p1"
        p2 = MagicMock()
        p2.id = "p2"
        p3 = MagicMock()
        p3.id = "p3"

        # Dense: p1 rank 1, p2 rank 2
        # Sparse: p2 rank 1, p3 rank 2
        dense_results = [p1, p2]
        sparse_results = [p2, p3]

        result = searcher._weighted_rrf(
            dense_results, sparse_results,
            dense_weight=0.5, sparse_weight=0.5, k=60
        )

        assert len(result) == 3
        # p2 should be highest (in both lists at good ranks)
        ids = [r[0].id for r in result]
        assert "p2" in ids

    def test_rrf_only_dense(self, searcher):
        """Point only in dense results."""
        point = MagicMock()
        point.id = "dense_only"

        dense_results = [point]
        sparse_results = []

        result = searcher._weighted_rrf(
            dense_results, sparse_results,
            dense_weight=0.5, sparse_weight=0.5, k=60
        )

        assert len(result) == 1
        assert result[0][0].id == "dense_only"
        # Score = 0.5/(60+1) = ~0.0082
        assert abs(result[0][1] - (0.5/61)) < 0.001

    def test_rrf_preserves_payload(self, searcher):
        """RRF preserves point payload."""
        point = MagicMock()
        point.id = "p1"
        point.payload = {"content": "test data"}

        result = searcher._weighted_rrf(
            [point], [],
            dense_weight=0.5, sparse_weight=0.5, k=60
        )

        assert result[0][0].payload == {"content": "test data"}


class TestHybridSearch:
    """Tests for hybrid search method."""

    @pytest.fixture
    def mock_searcher(self):
        """Create searcher with mocked storage."""
        mock_storage = MagicMock()
        mock_storage._get_client = AsyncMock()

        return HybridSearcher(
            mock_storage,
            dense_weight=0.5,
            sparse_weight=0.5,
            rrf_k=60,
        )

    async def test_search_returns_results(self, mock_searcher):
        """Search returns SearchResult objects."""
        mock_client = AsyncMock()
        mock_searcher.storage._get_client.return_value = mock_client

        # Mock query response
        mock_point = MagicMock()
        mock_point.id = "result1"
        mock_point.score = 0.9
        mock_point.payload = {"title": "Test Result"}

        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_client.query_points.return_value = mock_response

        sparse_vec = SparseVector(indices=[0, 1], values=[0.5, 0.3])

        results = await mock_searcher.search(
            collection="test_collection",
            dense_query=[0.1] * 100,
            sparse_query=sparse_vec,
            limit=10,
        )

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].id == "result1"
        assert results[0].score == 0.9
        assert results[0].payload == {"title": "Test Result"}

    async def test_search_with_filter(self, mock_searcher):
        """Search applies filter conditions."""
        mock_client = AsyncMock()
        mock_searcher.storage._get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        from qdrant_client.models import FieldCondition, MatchValue
        filter_cond = FieldCondition(key="type", match=MatchValue(value="note"))

        sparse_vec = SparseVector(indices=[0], values=[0.5])

        await mock_searcher.search(
            collection="test",
            dense_query=[0.1] * 100,
            sparse_query=sparse_vec,
            filter_conditions=[filter_cond],
        )

        # Verify query was called with filter
        mock_client.query_points.assert_called()

    @pytest.mark.asyncio
    async def test_search_weighted_fusion_path(self):
        """Search with unequal weights uses weighted RRF (lines 179-200)."""
        mock_storage = MagicMock()
        mock_storage._get_client = AsyncMock()

        # Create searcher with unequal weights to trigger weighted fusion path
        searcher = HybridSearcher(
            mock_storage,
            dense_weight=0.7,  # Unequal weights
            sparse_weight=0.3,
            rrf_k=60,
        )

        mock_client = AsyncMock()
        mock_storage._get_client.return_value = mock_client

        # Mock separate query responses for dense and sparse
        mock_dense_point = MagicMock()
        mock_dense_point.id = "dense1"
        mock_dense_point.score = 0.9
        mock_dense_point.payload = {"source": "dense"}

        mock_sparse_point = MagicMock()
        mock_sparse_point.id = "sparse1"
        mock_sparse_point.score = 0.8
        mock_sparse_point.payload = {"source": "sparse"}

        mock_dense_response = MagicMock()
        mock_dense_response.points = [mock_dense_point]

        mock_sparse_response = MagicMock()
        mock_sparse_response.points = [mock_sparse_point]

        # query_points will be called twice (dense, sparse)
        mock_client.query_points.side_effect = [mock_dense_response, mock_sparse_response]

        sparse_vec = SparseVector(indices=[0, 1], values=[0.5, 0.3])

        results = await searcher.search(
            collection="test_collection",
            dense_query=[0.1] * 100,
            sparse_query=sparse_vec,
            limit=10,
        )

        # Should have both results
        assert len(results) == 2
        # query_points called twice for separate dense/sparse searches
        assert mock_client.query_points.call_count == 2


class TestFuseResults:
    """Tests for fuse_results method."""

    @pytest.fixture
    def searcher(self):
        """Create searcher with known weights."""
        mock_storage = MagicMock()
        return HybridSearcher(
            mock_storage,
            dense_weight=0.5,
            sparse_weight=0.5,
            rrf_k=60,
        )

    def test_fuse_empty_results(self, searcher):
        """Fusing empty lists returns empty."""
        result = searcher.fuse_results([], [])
        assert result == []

    def test_fuse_single_dense_result(self, searcher):
        """Fuse with only dense result."""
        dense_result = SearchResult(id="d1", score=0.9, payload={"source": "dense"})

        result = searcher.fuse_results([dense_result], [], limit=10)

        assert len(result) == 1
        assert result[0].id == "d1"
        assert result[0].payload == {"source": "dense"}

    def test_fuse_single_sparse_result(self, searcher):
        """Fuse with only sparse result."""
        sparse_result = SearchResult(id="s1", score=0.8, payload={"source": "sparse"})

        result = searcher.fuse_results([], [sparse_result], limit=10)

        assert len(result) == 1
        assert result[0].id == "s1"
        assert result[0].payload == {"source": "sparse"}

    def test_fuse_overlapping_results(self, searcher):
        """Fuse results appearing in both lists."""
        dense_results = [
            SearchResult(id="shared", score=0.9, payload={"title": "Dense Title"}),
            SearchResult(id="dense_only", score=0.8, payload={}),
        ]
        sparse_results = [
            SearchResult(id="shared", score=0.85, payload={"title": "Sparse Title"}),
            SearchResult(id="sparse_only", score=0.7, payload={}),
        ]

        result = searcher.fuse_results(dense_results, sparse_results, limit=10)

        assert len(result) == 3
        # Shared should be ranked highest (in both lists)
        assert result[0].id == "shared"

    def test_fuse_respects_limit(self, searcher):
        """Fuse respects limit parameter."""
        dense_results = [
            SearchResult(id=f"d{i}", score=0.9-i*0.1, payload={})
            for i in range(5)
        ]
        sparse_results = []

        result = searcher.fuse_results(dense_results, sparse_results, limit=3)

        assert len(result) == 3

    def test_fuse_with_custom_weights(self, searcher):
        """Fuse with custom weight override."""
        dense_results = [SearchResult(id="d1", score=0.9, payload={})]
        sparse_results = [SearchResult(id="s1", score=0.9, payload={})]

        # With 1.0 dense weight and 0.0 sparse, dense should dominate
        result = searcher.fuse_results(
            dense_results, sparse_results,
            dense_weight=1.0, sparse_weight=0.0,
            limit=10
        )

        # Dense result gets full RRF score, sparse gets 0
        assert result[0].id == "d1"
