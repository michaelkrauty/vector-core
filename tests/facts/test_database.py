"""Tests for facts/database module."""

import tempfile
import time
from pathlib import Path

import pytest

from vector_core.facts.database import FactStore


def _pause() -> None:
    """Guarantee distinct `modified` timestamps between writes."""
    time.sleep(0.002)


@pytest.fixture
def store():
    """Create a FactStore instance with temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = FactStore(db_path=Path(tmpdir) / "test_facts.db")
        yield s
        s.close()


class TestReadBatchOrdering:
    """_read_batch must preserve the caller's ID order.

    Callers like query() select IDs with ORDER BY modified DESC; SQL
    `IN (...)` reads return rows in arbitrary order, so the batch read
    has to restore the caller's order.
    """

    def test_query_returns_most_recently_modified_first(self, store):
        """query() results follow ORDER BY modified DESC through _read_batch."""
        fact_a = store.create("alpha", "relates_to", "one")
        _pause()
        fact_b = store.create("beta", "relates_to", "two")
        _pause()
        fact_c = store.create("gamma", "relates_to", "three")
        _pause()

        # Touch the oldest fact so it becomes the most recently modified.
        store.update(fact_a.id, context="touched")

        results = store.query(predicate="relates_to")

        assert [f.id for f in results] == [fact_a.id, fact_c.id, fact_b.id]

    def test_query_order_survives_source_attachment(self, store):
        """Facts with sources keep their order through the two-query batch read."""
        facts = []
        for i in range(5):
            facts.append(store.create(f"subject_{i}", "ordered_by", f"obj_{i}"))
            _pause()

        results = store.query(predicate="ordered_by")

        expected = [f.id for f in reversed(facts)]  # newest first
        assert [f.id for f in results] == expected
