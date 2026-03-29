"""Typed Pydantic models for Qdrant payloads.

These models provide type safety for the payload dictionaries stored in Qdrant.
Use them when constructing or parsing payloads to catch type errors early.

Usage:
    # Creating a payload
    payload = NotePayload(
        note_id="abc123",
        title="My Note",
        tags=["python", "mcp"],
        category="tech",
        created="2024-01-01T00:00:00Z",
        modified="2024-01-01T00:00:00Z",
        note_hash="hash123",
    )
    point = PointStruct(id=1, vector=vec, payload=payload_to_dict(payload))

    # Parsing a payload from search results
    payload = NotePayload.model_validate(point.payload)
    print(payload.title)  # Type-safe access
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class BasePayload(BaseModel):
    """Base payload with common fields."""

    type: str = Field(description="Payload type discriminator")

    model_config = {"extra": "allow"}  # Allow additional fields for flexibility


class ChunkPayloadMixin(BaseModel):
    """Mixin for chunk-level payloads."""

    start_line: int = Field(ge=0, description="Starting line number in source")
    end_line: int = Field(ge=0, description="Ending line number in source")
    content: str = Field(description="Chunk text content (may be truncated)")


# --- Note payloads (mcp-notes) ---


class NotePayload(BasePayload):
    """File-level payload for notes."""

    type: Literal["note"] = "note"
    note_id: str = Field(description="Note UUID as string")
    title: str = Field(description="Note title from frontmatter")
    tags: list[str] = Field(default_factory=list, description="Note tags")
    category: str | None = Field(default=None, description="Category from path")
    created: str = Field(description="Created timestamp (ISO format)")
    modified: str = Field(description="Modified timestamp (ISO format)")
    note_hash: str = Field(description="Content hash for change detection")


class NoteChunkPayload(BasePayload, ChunkPayloadMixin):
    """Chunk-level payload for note sections."""

    type: Literal["chunk"] = "chunk"
    note_id: str = Field(description="Parent note UUID as string")
    title: str = Field(description="Parent note title")
    chunk_index: int = Field(ge=0, description="Chunk index within note")
    section_title: str | None = Field(
        default=None, description="Heading for this section"
    )
    tags: list[str] = Field(default_factory=list, description="Parent note tags")
    category: str | None = Field(default=None, description="Category from path")


# --- Code payloads (mcp-codesearch) ---


class CodeFilePayload(BasePayload):
    """File-level payload for code files."""

    type: Literal["file"] = "file"
    path: str = Field(description="Relative path from codebase root")
    abs_path: str = Field(description="Absolute filesystem path")
    language: str = Field(description="Programming language")
    file_hash: str = Field(description="Content hash for change detection")
    summary: str = Field(default="", description="AI-generated file summary")
    line_count: int = Field(ge=0, description="Total lines in file")
    size_bytes: int = Field(ge=0, description="File size in bytes")
    mtime: float = Field(description="Modification time (Unix timestamp)")


class CodeChunkPayload(BasePayload, ChunkPayloadMixin):
    """Chunk-level payload for code symbols/functions."""

    type: Literal["chunk"] = "chunk"
    path: str = Field(description="Relative path from codebase root")
    abs_path: str = Field(description="Absolute filesystem path")
    language: str = Field(description="Programming language")
    file_hash: str = Field(description="Parent file content hash")
    chunk_type: str = Field(description="Symbol type (function, class, method, etc.)")
    name: str = Field(description="Symbol name")
    context: str | None = Field(
        default=None, description="Surrounding context (class name for methods)"
    )


def payload_to_dict(payload: BasePayload) -> dict[str, Any]:
    """Convert a typed payload to dict for Qdrant.

    Args:
        payload: Pydantic model instance

    Returns:
        Dictionary suitable for Qdrant payload field
    """
    return payload.model_dump(mode="json")
