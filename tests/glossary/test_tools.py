"""Tests for glossary/tools module."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from vector_core.glossary.models import GlossaryEntry, GlossaryNotFoundError, TermExistsError
from vector_core.glossary.store import GlossaryStore
from vector_core.glossary.tools import GlossaryToolHelper


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def store(temp_db):
    """Create a GlossaryStore instance with temporary database."""
    s = GlossaryStore(db_path=temp_db / "glossary.db")
    yield s
    s.close()


@pytest.fixture
def mock_indexer():
    """Create mock GlossaryIndexer."""
    indexer = MagicMock()
    indexer.index_entry = AsyncMock()
    indexer.delete_entry_index = AsyncMock()
    indexer.search = AsyncMock(return_value=[])
    return indexer


@pytest.fixture
def helper(store, mock_indexer):
    """Create a GlossaryToolHelper with mocks."""
    return GlossaryToolHelper(store=store, indexer=mock_indexer)


class TestGlossaryToolHelperAddEntry:
    """Tests for GlossaryToolHelper.add_entry method."""

    @pytest.mark.asyncio
    async def test_add_entry(self, helper, mock_indexer):
        """Should add entry and index it."""
        result = await helper.add_entry(
            term="API",
            expansion="Application Programming Interface",
            definition="A set of protocols",
        )

        assert result["term"] == "API"
        assert result["expansion"] == "Application Programming Interface"
        assert "id" in result
        mock_indexer.index_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_entry_with_optional_fields(self, helper, mock_indexer):
        """Should add entry with domain and aliases."""
        result = await helper.add_entry(
            term="USAF",
            expansion="United States Air Force",
            definition="Military branch",
            domain="military",
            aliases=["Air Force", "US Air Force"],
        )

        assert result["term"] == "USAF"
        assert result["domain"] == "military"
        assert "Air Force" in result["aliases"]

    @pytest.mark.asyncio
    async def test_add_entry_duplicate_returns_error(self, helper, store):
        """Should return error dict for duplicate term."""
        store.create(
            term="API",
            expansion="First expansion",
            definition="First definition",
        )

        result = await helper.add_entry(
            term="API",
            expansion="Second expansion",
            definition="Second definition",
        )

        assert "error_code" in result
        assert "exists" in result["message"].lower()


class TestGlossaryToolHelperLookup:
    """Tests for GlossaryToolHelper.lookup method."""

    @pytest.mark.asyncio
    async def test_lookup_by_term(self, helper, store):
        """Should find entry by term."""
        entry = store.create(
            term="API",
            expansion="Application Programming Interface",
            definition="A set of protocols",
        )

        result = await helper.lookup("API")

        assert result["id"] == str(entry.id)
        assert result["term"] == "API"

    @pytest.mark.asyncio
    async def test_lookup_case_insensitive(self, helper, store):
        """Should find entry case-insensitively."""
        entry = store.create(
            term="API",
            expansion="Application Programming Interface",
            definition="A set of protocols",
        )

        result = await helper.lookup("api")
        assert result["id"] == str(entry.id)

    @pytest.mark.asyncio
    async def test_lookup_by_alias(self, helper, store):
        """Should find entry by alias."""
        entry = store.create(
            term="USAF",
            expansion="United States Air Force",
            definition="Military branch",
            aliases=["Air Force"],
        )

        result = await helper.lookup("Air Force")
        assert result["id"] == str(entry.id)

    @pytest.mark.asyncio
    async def test_lookup_not_found(self, helper):
        """Should return error dict for unknown term."""
        result = await helper.lookup("nonexistent")

        assert "error_code" in result


class TestGlossaryToolHelperSearch:
    """Tests for GlossaryToolHelper.search method."""

    @pytest.mark.asyncio
    async def test_search_calls_indexer(self, helper, mock_indexer):
        """Should delegate to indexer.search."""
        mock_indexer.search.return_value = [
            {"glossary_id": str(uuid4()), "term": "API", "score": 0.9}
        ]

        result = await helper.search("application interface")

        # indexer.search is called with positional args
        mock_indexer.search.assert_called_once_with("application interface", None, 10)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_search_with_domain_filter(self, helper, mock_indexer):
        """Should pass domain filter to indexer."""
        await helper.search("application interface", domain="tech", limit=5)

        # indexer.search is called with positional args
        mock_indexer.search.assert_called_once_with("application interface", "tech", 5)

    @pytest.mark.asyncio
    async def test_search_empty_results(self, helper, mock_indexer):
        """Should return empty list when no results."""
        mock_indexer.search.return_value = []

        result = await helper.search("nonexistent query")

        assert result == []


class TestGlossaryToolHelperListEntries:
    """Tests for GlossaryToolHelper.list_entries method."""

    @pytest.mark.asyncio
    async def test_list_entries(self, helper, store):
        """Should list all entries."""
        store.create(term="A", expansion="A exp", definition="A def")
        store.create(term="B", expansion="B exp", definition="B def")

        result = await helper.list_entries()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_entries_with_domain(self, helper, store):
        """Should filter by domain."""
        store.create(term="A", expansion="A exp", definition="A def", domain="tech")
        store.create(term="B", expansion="B exp", definition="B def", domain="military")

        result = await helper.list_entries(domain="tech")

        assert len(result) == 1
        assert result[0]["term"] == "A"

    @pytest.mark.asyncio
    async def test_list_entries_with_limit(self, helper, store):
        """Should respect limit."""
        for i in range(10):
            store.create(term=f"Term{i}", expansion=f"Exp{i}", definition=f"Def{i}")

        result = await helper.list_entries(limit=5)

        assert len(result) == 5


class TestGlossaryToolHelperUpdateEntry:
    """Tests for GlossaryToolHelper.update_entry method."""

    @pytest.mark.asyncio
    async def test_update_entry_by_term(self, helper, store, mock_indexer):
        """Should update entry by term."""
        entry = store.create(
            term="API",
            expansion="Old expansion",
            definition="Old definition",
        )

        result = await helper.update_entry("API", expansion="New expansion")

        assert result["expansion"] == "New expansion"
        mock_indexer.index_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_entry_by_uuid(self, helper, store, mock_indexer):
        """Should update entry by UUID."""
        entry = store.create(
            term="API",
            expansion="Old expansion",
            definition="Old definition",
        )

        result = await helper.update_entry(str(entry.id), definition="New definition")

        assert result["definition"] == "New definition"

    @pytest.mark.asyncio
    async def test_update_entry_multiple_fields(self, helper, store, mock_indexer):
        """Should update multiple fields."""
        entry = store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
        )

        result = await helper.update_entry(
            "API",
            term="REST API",
            expansion="New expansion",
            domain="tech",
        )

        assert result["term"] == "REST API"
        assert result["expansion"] == "New expansion"
        assert result["domain"] == "tech"

    @pytest.mark.asyncio
    async def test_update_entry_not_found(self, helper):
        """Should return error dict for unknown entry."""
        result = await helper.update_entry("nonexistent", expansion="New")

        assert "error_code" in result

    @pytest.mark.asyncio
    async def test_update_entry_clear_domain(self, helper, store, mock_indexer):
        """Should allow clearing domain with None."""
        store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
            domain="tech",
        )

        result = await helper.update_entry("API", domain=None)

        assert result["domain"] is None

    @pytest.mark.asyncio
    async def test_update_aliases(self, helper, store, mock_indexer):
        """Should update aliases."""
        store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
            aliases=["old_alias"],
        )

        result = await helper.update_entry("API", aliases=["new_alias1", "new_alias2"])

        assert set(result["aliases"]) == {"new_alias1", "new_alias2"}


class TestGlossaryToolHelperDeleteEntry:
    """Tests for GlossaryToolHelper.delete_entry method."""

    @pytest.mark.asyncio
    async def test_delete_entry_by_term(self, helper, store, mock_indexer):
        """Should delete entry by term."""
        entry = store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
        )

        result = await helper.delete_entry("API")

        assert result["success"] is True
        assert result["deleted_id"] == str(entry.id)
        mock_indexer.delete_entry_index.assert_called_once_with(entry.id)

    @pytest.mark.asyncio
    async def test_delete_entry_by_uuid(self, helper, store, mock_indexer):
        """Should delete entry by UUID."""
        entry = store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
        )

        result = await helper.delete_entry(str(entry.id))

        assert result["success"] is True
        assert result["deleted_id"] == str(entry.id)

    @pytest.mark.asyncio
    async def test_delete_entry_not_found(self, helper):
        """Should return error dict for unknown entry."""
        result = await helper.delete_entry("nonexistent")

        assert "error_code" in result


class TestGlossaryToolHelperWithoutIndexer:
    """Tests for GlossaryToolHelper without indexer."""

    @pytest.mark.asyncio
    async def test_add_entry_no_indexer(self, store):
        """Should work without indexer."""
        helper = GlossaryToolHelper(store=store, indexer=None)

        result = await helper.add_entry(
            term="API",
            expansion="Application Programming Interface",
            definition="A set of protocols",
        )

        assert result["term"] == "API"

    @pytest.mark.asyncio
    async def test_update_entry_no_indexer(self, store):
        """Should work without indexer."""
        helper = GlossaryToolHelper(store=store, indexer=None)
        store.create(
            term="API",
            expansion="Old",
            definition="Definition",
        )

        result = await helper.update_entry("API", expansion="New")

        assert result["expansion"] == "New"

    @pytest.mark.asyncio
    async def test_delete_entry_no_indexer(self, store):
        """Should work without indexer."""
        helper = GlossaryToolHelper(store=store, indexer=None)
        store.create(
            term="API",
            expansion="Expansion",
            definition="Definition",
        )

        result = await helper.delete_entry("API")

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_search_no_indexer_returns_error(self, store):
        """Should return error when searching without indexer."""
        helper = GlossaryToolHelper(store=store, indexer=None)

        result = await helper.search("query")

        assert len(result) == 1
        assert "error_code" in result[0]


class TestGlossaryToolHelperInputValidation:
    """Blank/whitespace-only inputs must fail fast with INVALID_INPUT."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("term", "expansion", "definition", "field"),
        [
            ("", "Expansion", "Definition", "term"),
            ("   ", "Expansion", "Definition", "term"),
            ("TERM", "", "Definition", "expansion"),
            ("TERM", "Expansion", "\t\n", "definition"),
        ],
    )
    async def test_add_entry_blank_required_field(
        self, helper, store, term, expansion, definition, field
    ):
        result = await helper.add_entry(term, expansion, definition)

        assert result["error_code"] == "invalid_input"
        assert field in result["message"]
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_add_entry_blank_domain(self, helper, store):
        result = await helper.add_entry("TERM", "Expansion", "Definition", domain="  ")

        assert result["error_code"] == "invalid_input"
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_add_entry_blank_alias(self, helper, store):
        result = await helper.add_entry(
            "TERM", "Expansion", "Definition", aliases=["ok", "  "]
        )

        assert result["error_code"] == "invalid_input"
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_add_entry_strips_whitespace(self, helper, store):
        result = await helper.add_entry(
            "  USAF  ", "  United States Air Force ", " Air branch. ",
            domain=" military ", aliases=[" Air Force "],
        )

        assert "error_code" not in result
        entry = store.lookup("USAF")
        assert entry is not None
        assert entry.term == "USAF"
        assert entry.expansion == "United States Air Force"
        assert entry.definition == "Air branch."
        assert entry.domain == "military"
        assert entry.aliases == ["Air Force"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kwargs", [
        {"term": " "},
        {"expansion": ""},
        {"definition": "\n"},
        {"domain": "   "},
        {"aliases": ["valid", ""]},
    ])
    async def test_update_entry_blank_field(self, helper, store, kwargs):
        store.create(term="API", expansion="Expansion", definition="Definition")

        result = await helper.update_entry("API", **kwargs)

        assert result["error_code"] == "invalid_input"
        # Entry unchanged
        entry = store.lookup("API")
        assert entry.expansion == "Expansion"

    @pytest.mark.asyncio
    async def test_update_entry_none_domain_still_clears(self, helper, store):
        """None remains the documented way to clear domain."""
        store.create(term="API", expansion="Exp", definition="Def", domain="tech")

        result = await helper.update_entry("API", domain=None)

        assert "error_code" not in result
        assert store.lookup("API").domain is None

    @pytest.mark.asyncio
    async def test_update_entry_strips_whitespace(self, helper, store):
        store.create(term="API", expansion="Exp", definition="Def")

        result = await helper.update_entry("API", expansion="  New Expansion  ")

        assert "error_code" not in result
        assert store.lookup("API").expansion == "New Expansion"
