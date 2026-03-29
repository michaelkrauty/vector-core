"""Tests for typed payload models."""

import pytest
from pydantic import ValidationError

from vector_core.models import (
    CodeChunkPayload,
    CodeFilePayload,
    NoteChunkPayload,
    NotePayload,
    payload_to_dict,
)


class TestNotePayload:
    """Tests for NotePayload model."""

    def test_create_note_payload(self):
        """Should create note payload with required fields."""
        payload = NotePayload(
            note_id="abc-123",
            title="Test Note",
            created="2024-01-01T00:00:00Z",
            modified="2024-01-01T00:00:00Z",
            note_hash="hash123",
        )
        assert payload.type == "note"
        assert payload.note_id == "abc-123"
        assert payload.tags == []
        assert payload.category is None

    def test_note_payload_with_optional_fields(self):
        """Should create note payload with optional fields."""
        payload = NotePayload(
            note_id="abc-123",
            title="Test Note",
            tags=["python", "mcp"],
            category="tech/tools",
            created="2024-01-01T00:00:00Z",
            modified="2024-01-01T00:00:00Z",
            note_hash="hash123",
        )
        assert payload.tags == ["python", "mcp"]
        assert payload.category == "tech/tools"

    def test_payload_to_dict(self):
        """Should convert payload to dict for Qdrant."""
        payload = NotePayload(
            note_id="abc-123",
            title="Test Note",
            tags=["python"],
            created="2024-01-01T00:00:00Z",
            modified="2024-01-01T00:00:00Z",
            note_hash="hash123",
        )
        result = payload_to_dict(payload)
        assert isinstance(result, dict)
        assert result["type"] == "note"
        assert result["note_id"] == "abc-123"
        assert result["tags"] == ["python"]


class TestNoteChunkPayload:
    """Tests for NoteChunkPayload model."""

    def test_create_chunk_payload(self):
        """Should create chunk payload with required fields."""
        payload = NoteChunkPayload(
            note_id="abc-123",
            title="Test Note",
            chunk_index=0,
            start_line=10,
            end_line=20,
            content="Some content here",
        )
        assert payload.type == "chunk"
        assert payload.chunk_index == 0
        assert payload.section_title is None

    def test_chunk_payload_validation(self):
        """Should validate line numbers are non-negative."""
        with pytest.raises(ValidationError):
            NoteChunkPayload(
                note_id="abc-123",
                title="Test Note",
                chunk_index=0,
                start_line=-1,  # Invalid
                end_line=20,
                content="Content",
            )


class TestCodePayloads:
    """Tests for code payload models."""

    def test_create_file_payload(self):
        """Should create code file payload."""
        payload = CodeFilePayload(
            path="src/main.py",
            abs_path="/home/user/project/src/main.py",
            language="python",
            file_hash="hash123",
            line_count=100,
            size_bytes=2048,
            mtime=1700000000.0,
        )
        assert payload.type == "file"
        assert payload.language == "python"
        assert payload.summary == ""

    def test_create_chunk_payload(self):
        """Should create code chunk payload."""
        payload = CodeChunkPayload(
            path="src/main.py",
            abs_path="/home/user/project/src/main.py",
            language="python",
            file_hash="hash123",
            chunk_type="function",
            name="my_function",
            start_line=10,
            end_line=25,
            content="def my_function(): pass",
        )
        assert payload.type == "chunk"
        assert payload.chunk_type == "function"
        assert payload.context is None


class TestPayloadParsing:
    """Tests for parsing payloads from Qdrant results."""

    def test_parse_note_payload_from_dict(self):
        """Should parse note payload from dict (simulating Qdrant result)."""
        raw = {
            "type": "note",
            "note_id": "abc-123",
            "title": "Test",
            "tags": ["tag1"],
            "category": None,
            "created": "2024-01-01T00:00:00Z",
            "modified": "2024-01-01T00:00:00Z",
            "note_hash": "hash",
        }
        payload = NotePayload.model_validate(raw)
        assert payload.note_id == "abc-123"
        assert payload.tags == ["tag1"]

    def test_parse_with_extra_fields(self):
        """Should allow extra fields (forward compatibility)."""
        raw = {
            "type": "note",
            "note_id": "abc-123",
            "title": "Test",
            "created": "2024-01-01T00:00:00Z",
            "modified": "2024-01-01T00:00:00Z",
            "note_hash": "hash",
            "future_field": "value",  # Unknown field
        }
        payload = NotePayload.model_validate(raw)
        assert payload.note_id == "abc-123"
