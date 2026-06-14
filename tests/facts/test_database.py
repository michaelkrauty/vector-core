"""Tests for facts/database module."""

import sqlite3
import tempfile
import time
from datetime import date
from pathlib import Path

import pytest

from vector_core.facts.database import FactStore
from vector_core.facts.models import FactNotFoundError


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


class TestInputValidation:
    """create()/update() reject garbage before any database access.

    Blank SPO/type fields would corrupt spo_hash deduplication and
    entity adjacency; confidence is documented as 0.0-1.0.
    """

    @pytest.mark.parametrize(
        "field",
        ["subject", "predicate", "object_value", "subject_type", "object_type"],
    )
    @pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
    def test_create_rejects_blank_string_fields(self, store, field, blank):
        """Each required string field rejects empty/whitespace-only values."""
        kwargs = {
            "subject": "John",
            "predicate": "works_at",
            "object_value": "Acme",
            "subject_type": "person",
            "object_type": "organization",
            field: blank,
        }
        with pytest.raises(ValueError, match=field):
            store.create(**kwargs)
        assert store.count() == 0

    @pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
    def test_create_rejects_out_of_range_confidence(self, store, bad):
        """Confidence outside [0.0, 1.0] raises before any row is written."""
        with pytest.raises(ValueError, match="confidence"):
            store.create("John", "works_at", "Acme", confidence=bad)
        assert store.count() == 0

    @pytest.mark.parametrize("ok", [0.0, 0.5, 1.0])
    def test_create_accepts_boundary_confidence(self, store, ok):
        """0.0 and 1.0 are inclusive bounds."""
        fact = store.create("John", "works_at", f"Acme-{ok}", confidence=ok)
        assert fact.confidence == ok

    @pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
    def test_update_rejects_out_of_range_confidence(self, store, bad):
        """update() validates confidence and leaves the fact unchanged."""
        fact = store.create("John", "works_at", "Acme", confidence=0.8)

        with pytest.raises(ValueError, match="confidence"):
            store.update(fact.id, confidence=bad)

        assert store.read(fact.id).confidence == 0.8

    def test_update_accepts_valid_confidence(self, store):
        """In-range confidence still updates normally."""
        fact = store.create("John", "works_at", "Acme", confidence=0.8)

        store.update(fact.id, confidence=1.0)

        assert store.read(fact.id).confidence == 1.0

    def test_create_does_not_normalize_accepted_values(self, store):
        """The store validates but stores values exactly as given."""
        fact = store.create("  John  ", "works_at", "Acme")
        assert store.read(fact.id).subject == "  John  "


class TestFindConnectionsTypeFilterCase:
    """find_connections type filters must match case-insensitively.

    The entity_adjacency table stores entity names AND types lowercased
    (_update_adjacency), so filters compared against those rows have to be
    normalized the same way. Passing a type exactly as facts display it
    (e.g. "Person") must not silently yield zero paths.
    """

    @pytest.fixture
    def peopled_store(self, store):
        store.create(
            "Alice", "works_at", "Acme",
            subject_type="Person", object_type="Company",
        )
        store.create(
            "Bob", "works_at", "Acme",
            subject_type="Person", object_type="Company",
        )
        return store

    def test_source_type_filter_is_case_insensitive(self, peopled_store):
        paths = peopled_store.find_connections(
            "alice", "bob", source_type="Person"
        )
        assert len(paths) == 1

    def test_target_type_filter_is_case_insensitive(self, peopled_store):
        paths = peopled_store.find_connections(
            "alice", "bob", target_type="Person"
        )
        assert len(paths) == 1

    def test_lowercase_filters_still_match(self, peopled_store):
        paths = peopled_store.find_connections(
            "alice", "bob", source_type="person", target_type="person"
        )
        assert len(paths) == 1

    def test_wrong_type_still_excludes(self, peopled_store):
        """Normalization must not loosen the filter into a no-op."""
        assert peopled_store.find_connections(
            "alice", "bob", source_type="Company"
        ) == []
        assert peopled_store.find_connections(
            "alice", "bob", target_type="Robot"
        ) == []


class TestIterAllResilience:
    """iter_all snapshots ids then reads lazily; a fact deleted in between
    must be skipped, not raise (a force reindex deletes points up front and
    would otherwise abort with the index emptied)."""

    def test_skips_fact_deleted_during_iteration(self, store, monkeypatch):
        f1 = store.create("alpha", "relates_to", "one")
        _pause()
        f2 = store.create("beta", "relates_to", "two")

        real_read = store.read

        def flaky_read(fact_id):
            # Simulate f1 vanishing between the id snapshot and its read.
            if fact_id == f1.id:
                raise FactNotFoundError(str(fact_id))
            return real_read(fact_id)

        monkeypatch.setattr(store, "read", flaky_read)

        facts = list(store.iter_all())

        assert [f.id for f in facts] == [f2.id]

    def test_propagates_systemic_db_error(self, store, monkeypatch):
        """A systemic DB failure (e.g. locked) must propagate, not be swallowed
        as a single bad row — otherwise a force reindex sees an empty corpus and
        deletes every point."""
        store.create("alpha", "relates_to", "one")

        def locked_read(fact_id):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(store, "read", locked_read)

        with pytest.raises(sqlite3.OperationalError):
            list(store.iter_all())

    def test_skips_fact_that_fails_to_load(self, store, monkeypatch):
        """A malformed row (non-FactNotFoundError read failure) is skipped, not fatal."""
        f1 = store.create("alpha", "relates_to", "one")
        _pause()
        f2 = store.create("beta", "relates_to", "two")

        real_read = store.read

        def flaky_read(fact_id):
            if fact_id == f1.id:
                raise ValueError("malformed stored row")
            return real_read(fact_id)

        monkeypatch.setattr(store, "read", flaky_read)

        facts = list(store.iter_all())

        assert [f.id for f in facts] == [f2.id]


class TestQueryTypeFilterCase:
    """query() and list_summaries() type filters must be case-insensitive.

    The facts table stores subject_type/object_type case-preserving, but every
    other filter in these methods (subject/predicate/object) and the graph
    methods (get_entity_facts/find_connections, normalized via adjacency) match
    case-insensitively. A fact stored with type "Person" must still be found by
    subject_type="person".
    """

    @pytest.fixture
    def typed_store(self, store):
        store.create(
            "Alice", "works_at", "Acme",
            subject_type="Person", object_type="Company",
        )
        store.create(
            "Bob", "founded", "Beta",
            subject_type="person", object_type="company",
        )
        return store

    def test_query_subject_type_case_insensitive(self, typed_store):
        assert len(typed_store.query(subject_type="person")) == 2
        assert len(typed_store.query(subject_type="PERSON")) == 2

    def test_query_object_type_case_insensitive(self, typed_store):
        assert len(typed_store.query(object_type="company")) == 2

    def test_query_wrong_type_still_excludes(self, typed_store):
        assert typed_store.query(subject_type="robot") == []

    def test_list_summaries_subject_type_case_insensitive(self, typed_store):
        assert len(typed_store.list_summaries(subject_type="person")) == 2

    def test_list_summaries_object_type_case_insensitive(self, typed_store):
        assert len(typed_store.list_summaries(object_type="COMPANY")) == 2


class TestTemporalRangeValidation:
    """create()/update() reject an inverted valid_from > valid_to interval."""

    def test_create_rejects_inverted_range(self, store):
        with pytest.raises(ValueError, match="valid_from"):
            store.create(
                "A", "r", "B",
                valid_from=date(2025, 1, 1), valid_to=date(2024, 1, 1),
            )
        assert store.count() == 0

    def test_create_accepts_equal_bounds(self, store):
        fact = store.create(
            "A", "r", "B",
            valid_from=date(2024, 1, 1), valid_to=date(2024, 1, 1),
        )
        assert fact.valid_from == fact.valid_to

    def test_create_accepts_ordered_range(self, store):
        fact = store.create(
            "A", "r", "B",
            valid_from=date(2024, 1, 1), valid_to=date(2025, 1, 1),
        )
        assert fact.valid_from < fact.valid_to

    def test_create_accepts_one_sided_bounds(self, store):
        assert store.create("A", "r", "B", valid_from=date(2024, 1, 1)).valid_to is None
        assert store.create("C", "r", "D", valid_to=date(2024, 1, 1)).valid_from is None

    def test_update_rejects_inverting_via_valid_from(self, store):
        fact = store.create("A", "r", "B", valid_to=date(2024, 1, 1))
        with pytest.raises(ValueError, match="valid_from"):
            store.update(fact.id, valid_from=date(2025, 1, 1))
        # nothing was written
        assert store.read(fact.id).valid_from is None

    def test_update_rejects_inverting_via_valid_to(self, store):
        fact = store.create("A", "r", "B", valid_from=date(2025, 1, 1))
        with pytest.raises(ValueError, match="valid_from"):
            store.update(fact.id, valid_to=date(2024, 1, 1))
        assert store.read(fact.id).valid_to is None

    def test_update_allows_ordered_change(self, store):
        fact = store.create("A", "r", "B", valid_from=date(2024, 1, 1))
        store.update(fact.id, valid_to=date(2025, 1, 1))
        assert store.read(fact.id).valid_to == date(2025, 1, 1)

    def test_update_clearing_a_bound_avoids_inversion(self, store):
        fact = store.create(
            "A", "r", "B",
            valid_from=date(2024, 1, 1), valid_to=date(2025, 1, 1),
        )
        store.update(fact.id, valid_to=None)
        assert store.read(fact.id).valid_to is None
