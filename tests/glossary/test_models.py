"""Tests for glossary/models module."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from vector_core.glossary.models import (
    GlossaryEntry,
    GlossaryEntrySummary,
    GlossaryError,
    GlossaryNotFoundError,
    TermExistsError,
)


class TestGlossaryEntry:
    """Tests for GlossaryEntry dataclass."""

    def test_create_entry(self):
        """Should create entry with all fields."""
        entry_id = uuid4()
        now = datetime.now(UTC)
        entry = GlossaryEntry(
            id=entry_id,
            term="USAF",
            expansion="United States Air Force",
            definition="The air service branch of the US Armed Forces.",
            domain="military",
            aliases=["Air Force", "US Air Force"],
            created=now,
            modified=now,
        )

        assert entry.id == entry_id
        assert entry.term == "USAF"
        assert entry.expansion == "United States Air Force"
        assert entry.definition == "The air service branch of the US Armed Forces."
        assert entry.domain == "military"
        assert entry.aliases == ["Air Force", "US Air Force"]

    def test_to_dict(self):
        """Should convert to dict correctly."""
        entry_id = uuid4()
        now = datetime.now(UTC)
        entry = GlossaryEntry(
            id=entry_id,
            term="API",
            expansion="Application Programming Interface",
            definition="A set of protocols for building software.",
            domain="tech",
            aliases=["interface"],
            created=now,
            modified=now,
        )

        result = entry.to_dict()

        assert result["id"] == str(entry_id)
        assert result["term"] == "API"
        assert result["expansion"] == "Application Programming Interface"
        assert result["definition"] == "A set of protocols for building software."
        assert result["domain"] == "tech"
        assert result["aliases"] == ["interface"]
        assert result["created"] == now.isoformat()
        assert result["modified"] == now.isoformat()

    def test_to_dict_with_none_domain(self):
        """Should handle None domain in to_dict."""
        entry = GlossaryEntry(
            id=uuid4(),
            term="Test",
            expansion="Test Expansion",
            definition="Test definition",
            domain=None,
            aliases=[],
            created=datetime.now(UTC),
            modified=datetime.now(UTC),
        )

        result = entry.to_dict()
        assert result["domain"] is None

    def test_to_dict_with_empty_aliases(self):
        """Should handle empty aliases list."""
        entry = GlossaryEntry(
            id=uuid4(),
            term="Test",
            expansion="Test Expansion",
            definition="Test definition",
            domain="test",
            aliases=[],
            created=datetime.now(UTC),
            modified=datetime.now(UTC),
        )

        result = entry.to_dict()
        assert result["aliases"] == []


class TestGlossaryEntrySummary:
    """Tests for GlossaryEntrySummary dataclass."""

    def test_create_summary(self):
        """Should create summary with all fields."""
        entry_id = uuid4()
        summary = GlossaryEntrySummary(
            id=entry_id,
            term="USAF",
            expansion="United States Air Force",
            domain="military",
        )

        assert summary.id == entry_id
        assert summary.term == "USAF"
        assert summary.expansion == "United States Air Force"
        assert summary.domain == "military"

    def test_to_dict(self):
        """Should convert to dict correctly."""
        entry_id = uuid4()
        summary = GlossaryEntrySummary(
            id=entry_id,
            term="API",
            expansion="Application Programming Interface",
            domain="tech",
        )

        result = summary.to_dict()

        assert result["id"] == str(entry_id)
        assert result["term"] == "API"
        assert result["expansion"] == "Application Programming Interface"
        assert result["domain"] == "tech"

    def test_to_dict_with_none_domain(self):
        """Should handle None domain in to_dict."""
        summary = GlossaryEntrySummary(
            id=uuid4(),
            term="Test",
            expansion="Test Expansion",
            domain=None,
        )

        result = summary.to_dict()
        assert result["domain"] is None


class TestGlossaryExceptions:
    """Tests for glossary exceptions."""

    def test_glossary_error_base(self):
        """GlossaryError should be base exception."""
        error = GlossaryError("test error")
        assert str(error) == "test error"

    def test_glossary_not_found_error(self):
        """GlossaryNotFoundError should include identifier."""
        error = GlossaryNotFoundError("test-uuid-123")
        assert error.identifier == "test-uuid-123"
        assert "test-uuid-123" in str(error)

    def test_glossary_not_found_is_glossary_error(self):
        """GlossaryNotFoundError should be subclass of GlossaryError."""
        error = GlossaryNotFoundError("test")
        assert isinstance(error, GlossaryError)

    def test_term_exists_error(self):
        """TermExistsError should include term."""
        error = TermExistsError("USAF")
        assert error.term == "USAF"
        assert "USAF" in str(error)

    def test_term_exists_is_glossary_error(self):
        """TermExistsError should be subclass of GlossaryError."""
        error = TermExistsError("test")
        assert isinstance(error, GlossaryError)
