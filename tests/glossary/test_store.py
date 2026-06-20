"""Tests for glossary/store module."""

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from vector_core.glossary.models import (
    GlossaryEntry,
    GlossaryEntrySummary,
    GlossaryNotFoundError,
    TermExistsError,
)
from vector_core.glossary.store import GlossaryStore, _compute_entry_hash


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_glossary.db"


@pytest.fixture
def store(temp_db):
    """Create a GlossaryStore instance with temporary database."""
    s = GlossaryStore(db_path=temp_db)
    yield s
    s.close()


class TestGlossaryStoreCreate:
    """Tests for GlossaryStore.create method."""

    def test_create_entry(self, store):
        """Should create entry with all fields."""
        entry = store.create(
            term="USAF",
            expansion="United States Air Force",
            definition="The air service branch of the US Armed Forces.",
            domain="military",
            aliases=["Air Force", "US Air Force"],
        )

        assert entry.term == "USAF"
        assert entry.expansion == "United States Air Force"
        assert entry.definition == "The air service branch of the US Armed Forces."
        assert entry.domain == "military"
        assert set(entry.aliases) == {"Air Force", "US Air Force"}
        assert entry.id is not None

    def test_create_minimal_entry(self, store):
        """Should create entry with only required fields."""
        entry = store.create(
            term="API",
            expansion="Application Programming Interface",
            definition="A set of protocols for building software.",
        )

        assert entry.term == "API"
        assert entry.domain is None
        assert entry.aliases == []

    def test_create_duplicate_term_raises(self, store):
        """Should raise TermExistsError for duplicate term."""
        store.create(
            term="USAF",
            expansion="United States Air Force",
            definition="Military branch",
        )

        with pytest.raises(TermExistsError) as exc:
            store.create(
                term="USAF",
                expansion="Different expansion",
                definition="Different definition",
            )

        assert exc.value.term == "USAF"

    def test_create_term_case_insensitive(self, store):
        """Should detect duplicate terms case-insensitively."""
        store.create(
            term="USAF",
            expansion="United States Air Force",
            definition="Military branch",
        )

        with pytest.raises(TermExistsError):
            store.create(
                term="usaf",
                expansion="Different expansion",
                definition="Different definition",
            )

    def test_create_alias_conflicts_with_term(self, store):
        """Should raise TermExistsError if alias matches existing term."""
        store.create(
            term="USAF",
            expansion="United States Air Force",
            definition="Military branch",
        )

        with pytest.raises(TermExistsError):
            store.create(
                term="Military",
                expansion="Military expansion",
                definition="Definition",
                aliases=["USAF"],
            )

    def test_create_alias_conflicts_with_alias(self, store):
        """Should raise TermExistsError if alias matches existing alias."""
        store.create(
            term="USAF",
            expansion="United States Air Force",
            definition="Military branch",
            aliases=["Air Force"],
        )

        with pytest.raises(TermExistsError):
            store.create(
                term="Other",
                expansion="Other expansion",
                definition="Definition",
                aliases=["Air Force"],
            )


class TestGlossaryStoreRead:
    """Tests for GlossaryStore.read method."""

    def test_read_entry(self, store):
        """Should read entry by ID."""
        created = store.create(
            term="API",
            expansion="Application Programming Interface",
            definition="A set of protocols.",
        )

        entry = store.read(created.id)

        assert entry.id == created.id
        assert entry.term == "API"

    def test_read_not_found(self, store):
        """Should raise GlossaryNotFoundError for unknown ID."""
        with pytest.raises(GlossaryNotFoundError):
            store.read(uuid4())

    def test_read_includes_aliases(self, store):
        """Should include aliases in read entry."""
        created = store.create(
            term="USAF",
            expansion="United States Air Force",
            definition="Military branch",
            aliases=["Air Force", "US Air Force"],
        )

        entry = store.read(created.id)
        assert set(entry.aliases) == {"Air Force", "US Air Force"}


class TestGlossaryStoreUpdate:
    """Tests for GlossaryStore.update method."""

    def test_update_term(self, store):
        """Should update term."""
        entry = store.create(
            term="API",
            expansion="Application Programming Interface",
            definition="A set of protocols.",
        )

        updated = store.update(entry.id, term="REST API")

        assert updated.term == "REST API"
        assert store.read(entry.id).term == "REST API"

    def test_update_expansion(self, store):
        """Should update expansion."""
        entry = store.create(
            term="API",
            expansion="Old expansion",
            definition="Definition",
        )

        updated = store.update(entry.id, expansion="New expansion")
        assert updated.expansion == "New expansion"

    def test_update_definition(self, store):
        """Should update definition."""
        entry = store.create(
            term="API",
            expansion="Expansion",
            definition="Old definition",
        )

        updated = store.update(entry.id, definition="New definition")
        assert updated.definition == "New definition"

    def test_update_domain(self, store):
        """Should update domain."""
        entry = store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
            domain="old",
        )

        updated = store.update(entry.id, domain="new")
        assert updated.domain == "new"

    def test_update_domain_to_none(self, store):
        """Should allow clearing domain with None."""
        entry = store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
            domain="tech",
        )

        updated = store.update(entry.id, domain=None)
        assert updated.domain is None

    def test_update_aliases(self, store):
        """Should replace aliases."""
        entry = store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
            aliases=["old1", "old2"],
        )

        updated = store.update(entry.id, aliases=["new1", "new2"])
        assert set(updated.aliases) == {"new1", "new2"}

    def test_update_aliases_to_empty(self, store):
        """Should allow clearing aliases."""
        entry = store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
            aliases=["alias1"],
        )

        updated = store.update(entry.id, aliases=[])
        assert updated.aliases == []

    def test_update_term_conflict(self, store):
        """Should raise TermExistsError if new term exists."""
        store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
        )
        entry2 = store.create(
            term="SDK",
            expansion="Software Development Kit",
            definition="Definition",
        )

        with pytest.raises(TermExistsError):
            store.update(entry2.id, term="API")

    def test_update_not_found(self, store):
        """Should raise GlossaryNotFoundError for unknown ID."""
        with pytest.raises(GlossaryNotFoundError):
            store.update(uuid4(), term="new")

    def test_update_modified_timestamp(self, store):
        """Should update modified timestamp."""
        entry = store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
        )
        original_modified = entry.modified

        updated = store.update(entry.id, expansion="New expansion")
        assert updated.modified > original_modified


class TestGlossaryStoreDelete:
    """Tests for GlossaryStore.delete method."""

    def test_delete_entry(self, store):
        """Should delete entry."""
        entry = store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
        )

        result = store.delete(entry.id)

        assert result is True
        with pytest.raises(GlossaryNotFoundError):
            store.read(entry.id)

    def test_delete_cascades_aliases(self, store):
        """Should delete aliases when entry deleted."""
        entry = store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
            aliases=["alias1", "alias2"],
        )

        store.delete(entry.id)

        # Aliases should no longer block new entries
        new_entry = store.create(
            term="NEW",
            expansion="New",
            definition="New",
            aliases=["alias1"],
        )
        assert "alias1" in new_entry.aliases

    def test_delete_not_found(self, store):
        """Should return False for unknown ID."""
        result = store.delete(uuid4())
        assert result is False


class TestGlossaryStoreLookup:
    """Tests for GlossaryStore.lookup method."""

    def test_lookup_by_term(self, store):
        """Should find entry by term."""
        created = store.create(
            term="API",
            expansion="Application Programming Interface",
            definition="A set of protocols.",
        )

        entry = store.lookup("API")
        assert entry is not None
        assert entry.id == created.id

    def test_lookup_by_term_case_insensitive(self, store):
        """Should find entry case-insensitively."""
        created = store.create(
            term="API",
            expansion="Application Programming Interface",
            definition="A set of protocols.",
        )

        assert store.lookup("api").id == created.id
        assert store.lookup("Api").id == created.id
        assert store.lookup("API").id == created.id

    def test_lookup_by_alias(self, store):
        """Should find entry by alias."""
        created = store.create(
            term="USAF",
            expansion="United States Air Force",
            definition="Military branch",
            aliases=["Air Force"],
        )

        entry = store.lookup("Air Force")
        assert entry is not None
        assert entry.id == created.id

    def test_lookup_by_alias_case_insensitive(self, store):
        """Should find entry by alias case-insensitively."""
        created = store.create(
            term="USAF",
            expansion="United States Air Force",
            definition="Military branch",
            aliases=["Air Force"],
        )

        assert store.lookup("air force").id == created.id

    def test_lookup_not_found(self, store):
        """Should return None for unknown term."""
        result = store.lookup("nonexistent")
        assert result is None


class TestGlossaryStoreFindByTermOrId:
    """Tests for GlossaryStore.find_by_term_or_id method."""

    def test_find_by_uuid(self, store):
        """Should find entry by UUID string."""
        created = store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
        )

        entry = store.find_by_term_or_id(str(created.id))
        assert entry is not None
        assert entry.id == created.id

    def test_find_by_term(self, store):
        """Should find entry by term."""
        created = store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
        )

        entry = store.find_by_term_or_id("API")
        assert entry is not None
        assert entry.id == created.id

    def test_find_not_found(self, store):
        """Should return None if not found."""
        result = store.find_by_term_or_id("nonexistent")
        assert result is None


class TestGlossaryStoreExistsTerm:
    """Tests for GlossaryStore.exists_term method."""

    def test_exists_term(self, store):
        """Should return True for existing term."""
        store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
        )

        assert store.exists_term("API") is True
        assert store.exists_term("api") is True

    def test_exists_alias(self, store):
        """Should return True for existing alias."""
        store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
            aliases=["Interface"],
        )

        assert store.exists_term("Interface") is True
        assert store.exists_term("interface") is True

    def test_not_exists(self, store):
        """Should return False for unknown term."""
        assert store.exists_term("nonexistent") is False


class TestGlossaryStoreListAll:
    """Tests for GlossaryStore.list_all method."""

    def test_list_all(self, store):
        """Should list all entries."""
        store.create(term="A", expansion="A exp", definition="A def")
        store.create(term="B", expansion="B exp", definition="B def")
        store.create(term="C", expansion="C exp", definition="C def")

        entries = store.list_all()

        assert len(entries) == 3
        assert all(isinstance(e, GlossaryEntrySummary) for e in entries)

    def test_list_all_by_domain(self, store):
        """Should filter by domain."""
        store.create(term="A", expansion="A exp", definition="A def", domain="tech")
        store.create(term="B", expansion="B exp", definition="B def", domain="tech")
        store.create(term="C", expansion="C exp", definition="C def", domain="military")

        tech_entries = store.list_all(domain="tech")
        assert len(tech_entries) == 2

    def test_list_all_with_limit(self, store):
        """Should respect limit."""
        for i in range(10):
            store.create(term=f"Term{i}", expansion=f"Exp{i}", definition=f"Def{i}")

        entries = store.list_all(limit=5)
        assert len(entries) == 5

    def test_list_all_limit_zero_returns_no_entries(self, store):
        """limit=0 means zero rows (SQLite LIMIT 0 semantics), not unlimited."""
        for i in range(3):
            store.create(term=f"Term{i}", expansion=f"Exp{i}", definition=f"Def{i}")

        assert store.list_all(limit=0) == []
        # None still means no limit.
        assert len(store.list_all(limit=None)) == 3

    def test_list_all_sorted_by_term(self, store):
        """Should return entries sorted by term."""
        store.create(term="Zebra", expansion="Zebra exp", definition="Zebra def")
        store.create(term="Apple", expansion="Apple exp", definition="Apple def")
        store.create(term="Mango", expansion="Mango exp", definition="Mango def")

        entries = store.list_all()
        terms = [e.term for e in entries]
        assert terms == ["Apple", "Mango", "Zebra"]


class TestGlossaryStoreIterAll:
    """Tests for GlossaryStore.iter_all method."""

    def test_iter_all(self, store):
        """Should iterate over all entries."""
        store.create(term="A", expansion="A exp", definition="A def")
        store.create(term="B", expansion="B exp", definition="B def")

        entries = list(store.iter_all())

        assert len(entries) == 2
        assert all(isinstance(e, GlossaryEntry) for e in entries)


class TestGlossaryStoreCount:
    """Tests for GlossaryStore.count method."""

    def test_count_empty(self, store):
        """Should return 0 for empty store."""
        assert store.count() == 0

    def test_count(self, store):
        """Should return correct count."""
        store.create(term="A", expansion="A exp", definition="A def")
        store.create(term="B", expansion="B exp", definition="B def")

        assert store.count() == 2


class TestGlossaryStoreGetDomains:
    """Tests for GlossaryStore.get_domains method."""

    def test_get_domains(self, store):
        """Should return unique domains."""
        store.create(term="A", expansion="A exp", definition="A def", domain="tech")
        store.create(term="B", expansion="B exp", definition="B def", domain="military")
        store.create(term="C", expansion="C exp", definition="C def", domain="tech")

        domains = store.get_domains()
        assert set(domains) == {"tech", "military"}

    def test_get_domains_excludes_none(self, store):
        """Should not include None in domains."""
        store.create(term="A", expansion="A exp", definition="A def", domain="tech")
        store.create(term="B", expansion="B exp", definition="B def")

        domains = store.get_domains()
        assert domains == ["tech"]

    def test_get_domains_empty(self, store):
        """Should return empty list for empty store."""
        domains = store.get_domains()
        assert domains == []


class TestGlossaryStoreContextManager:
    """Tests for GlossaryStore context manager."""

    def test_context_manager(self, temp_db):
        """Should support context manager protocol."""
        with GlossaryStore(db_path=temp_db) as store:
            store.create(term="API", expansion="Expansion", definition="Definition")

        # Verify data persisted
        store2 = GlossaryStore(db_path=temp_db)
        entry = store2.lookup("API")
        assert entry is not None
        store2.close()


class TestGlossaryStoreEntryHash:
    """Tests for entry_hash maintenance on update."""

    def test_alias_update_refreshes_entry_hash(self, store):
        """entry_hash must reflect alias changes (hash covers aliases)."""
        entry = store.create(
            term="USAF",
            expansion="United States Air Force",
            definition="The air service branch.",
            aliases=["Air Force"],
        )
        hash_before = store.get_entry_hash(entry.id)

        store.update(entry.id, aliases=["Air Force", "US Air Force"])

        hash_after = store.get_entry_hash(entry.id)
        assert hash_after != hash_before

    def test_entry_hash_matches_stored_content_after_alias_update(self, store):
        """Stored hash equals a fresh hash of the re-read entry."""
        entry = store.create(
            term="API",
            expansion="Application Programming Interface",
            definition="A set of protocols.",
            aliases=["interface"],
        )
        store.update(entry.id, aliases=["interface", "endpoint"])

        reread = store.read(entry.id)
        assert store.get_entry_hash(entry.id) == _compute_entry_hash(reread)

    def test_alias_only_update_changes_nothing_else(self, store):
        """Alias-only update leaves other fields intact and applies aliases."""
        entry = store.create(
            term="SLA",
            expansion="Service Level Agreement",
            definition="A service commitment.",
            domain="business",
            aliases=["agreement"],
        )
        updated = store.update(entry.id, aliases=["uptime promise"])

        assert updated.term == "SLA"
        assert updated.domain == "business"
        assert updated.aliases == ["uptime promise"]
        reread = store.read(entry.id)
        assert set(reread.aliases) == {"uptime promise"}


class TestGlossaryStoreAtomicity:
    """Failed mutations must leave the store fully unchanged.

    The store uses long-lived per-thread connections, so a write that
    raises mid-mutation leaves partial state pending in the implicit
    transaction — and the next successful operation on the same
    connection would commit it. Each test therefore performs a
    subsequent successful (committing) operation before asserting.
    """

    def test_update_alias_collision_with_term_leaves_entry_unchanged(self, store):
        """Alias colliding with another entry's term must not clear aliases."""
        store.create(term="API", expansion="E", definition="D")
        entry = store.create(
            term="SDK",
            expansion="Software Development Kit",
            definition="D",
            aliases=["devkit", "kit"],
        )

        with pytest.raises(TermExistsError):
            store.update(entry.id, aliases=["toolkit", "API"])

        # Commit anything pending on the shared connection.
        store.create(term="CLI", expansion="E", definition="D")

        reread = store.read(entry.id)
        assert set(reread.aliases) == {"devkit", "kit"}

    def test_update_alias_collision_with_alias_leaves_entry_unchanged(self, store):
        """Alias colliding with another entry's alias must not clear aliases."""
        store.create(term="API", expansion="E", definition="D", aliases=["interface"])
        entry = store.create(
            term="SDK",
            expansion="Software Development Kit",
            definition="D",
            aliases=["devkit"],
        )

        with pytest.raises(TermExistsError):
            store.update(entry.id, aliases=["interface"])

        store.create(term="CLI", expansion="E", definition="D")

        reread = store.read(entry.id)
        assert set(reread.aliases) == {"devkit"}

    def test_update_duplicate_aliases_raise_term_exists_and_leave_entry_unchanged(
        self, store
    ):
        """Case-normalized duplicates in the list raise TermExistsError, not
        a raw IntegrityError after the old aliases are already deleted."""
        entry = store.create(
            term="SDK",
            expansion="Software Development Kit",
            definition="D",
            aliases=["devkit"],
        )

        with pytest.raises(TermExistsError):
            store.update(entry.id, aliases=["kit", "KIT"])

        store.create(term="CLI", expansion="E", definition="D")

        reread = store.read(entry.id)
        assert set(reread.aliases) == {"devkit"}

    def test_failed_update_leaves_hash_and_modified_consistent(self, store):
        """After a failed update, stored hash still matches stored content."""
        entry = store.create(
            term="SDK",
            expansion="Software Development Kit",
            definition="D",
            aliases=["devkit"],
        )
        store.create(term="API", expansion="E", definition="D")

        with pytest.raises(TermExistsError):
            store.update(entry.id, expansion="changed", aliases=["API"])

        store.create(term="CLI", expansion="E", definition="D")

        reread = store.read(entry.id)
        assert reread.expansion == "Software Development Kit"
        assert reread.modified == entry.modified
        assert store.get_entry_hash(entry.id) == _compute_entry_hash(reread)

    def test_create_duplicate_aliases_raise_term_exists_and_leave_no_entry(
        self, store
    ):
        """Duplicate aliases in create() raise TermExistsError before the
        entry row is written — no dangling entry survives a later commit."""
        with pytest.raises(TermExistsError):
            store.create(
                term="Foo",
                expansion="E",
                definition="D",
                aliases=["bar", "BAR"],
            )

        store.create(term="CLI", expansion="E", definition="D")

        assert store.lookup("Foo") is None
        assert store.lookup("bar") is None
        assert store.count() == 1


class TestGlossaryStoreSelfRename:
    """Term renames that only collide with the entry's OWN rows must work."""

    def test_update_term_case_only_rename(self, store):
        """Changing only the casing of a term is a legitimate rename."""
        entry = store.create(term="USAF", expansion="E", definition="D")

        updated = store.update(entry.id, term="Usaf")

        assert updated.term == "Usaf"
        reread = store.read(entry.id)
        assert reread.term == "Usaf"
        assert store.lookup("usaf").id == entry.id

    def test_update_term_to_own_alias(self, store):
        """Renaming a term to one of the entry's own aliases is allowed."""
        entry = store.create(
            term="USAF",
            expansion="United States Air Force",
            definition="D",
            aliases=["Air Force"],
        )

        updated = store.update(entry.id, term="Air Force")

        assert updated.term == "Air Force"
        found = store.lookup("air force")
        assert found is not None and found.id == entry.id

    def test_update_term_still_rejects_other_entrys_alias(self, store):
        """Cross-entry semantics unchanged: another entry's alias conflicts."""
        store.create(term="USAF", expansion="E", definition="D", aliases=["Air Force"])
        entry = store.create(term="SDK", expansion="E", definition="D")

        with pytest.raises(TermExistsError):
            store.update(entry.id, term="air force")

    def test_case_only_rename_updates_entry_hash(self, store):
        """The stored hash must reflect the re-cased term."""
        entry = store.create(term="USAF", expansion="E", definition="D")

        store.update(entry.id, term="Usaf")

        reread = store.read(entry.id)
        assert store.get_entry_hash(entry.id) == _compute_entry_hash(reread)
