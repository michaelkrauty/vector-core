"""Recover exact embedding inputs from shared glossary and fact payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from vector_core.utils.hashing import hash_content

if TYPE_CHECKING:
    from vector_core.glossary.store import GlossaryStore


def _legacy_note_text(payload: dict[str, Any]) -> str:
    """Canonical metadata-only reconstruction, not the lost original note body."""
    title = payload.get("title")
    tags = payload.get("tags", [])
    category = payload.get("category")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Legacy note payload has no meaningful title")
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise ValueError("Legacy note payload has invalid tags")
    if category is not None and not isinstance(category, str):
        raise ValueError("Legacy note payload has invalid category")
    parts = [title]
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    if category:
        parts.append(f"Category: {category}")
    return "\n".join(parts)


def stored_embedding_text(payload: dict[str, Any]) -> str | None:
    """Read an explicit raw input or validated same-payload field reference."""
    if "embedding_text" in payload:
        text = payload["embedding_text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid persisted embedding text")
        return text
    if "embedding_text_field" in payload:
        field = payload["embedding_text_field"]
        text = payload.get(field) if isinstance(field, str) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid embedding text field reference")
        return text
    return None


async def resolve_shared_embedding_text(
    payload: dict[str, Any], *, glossary_store: GlossaryStore | None = None
) -> str:
    """Recover retained inputs or explicitly reconstructed legacy note summaries.

    Glossary definitions were truncated in Qdrant. Their complete SQLite row
    is usable only when every embedding-bearing field still matches the point.
    """
    text = stored_embedding_text(payload)
    if text is not None:
        return text
    if payload.get("type") == "note":
        return _legacy_note_text(payload)
    if payload.get("type") in {"fact", "document", "doc_chunk"} or (
        payload.get("type") == "chunk" and payload.get("note_id")
    ):
        text = payload.get("content")
        if not isinstance(text, str) or not text:
            raise ValueError("Shared payload has no exact embedding content")
        return text
    if payload.get("type") != "glossary":
        raise ValueError("Unsupported shared embedding payload type")

    required = ("glossary_id", "entry_hash", "term", "expansion", "definition", "domain", "aliases")
    if any(key not in payload for key in required):
        raise ValueError("Glossary payload is missing identity fields")
    # Import lazily: glossary's package exports its indexer, which uses this resolver.
    from vector_core.glossary.models import GlossaryNotFoundError  # noqa: PLC0415
    from vector_core.glossary.store import GlossaryStore  # noqa: PLC0415

    store = glossary_store or GlossaryStore()
    try:
        try:
            entry = store.read(UUID(payload["glossary_id"]))
        except (GlossaryNotFoundError, TypeError, ValueError) as exc:
            raise ValueError("Glossary source row is unavailable") from exc
        expected_hash = hash_content(f"{entry.term}|{entry.expansion}|{entry.definition}")
        if (
            payload["entry_hash"] != expected_hash
            or payload["term"] != entry.term
            or payload["expansion"] != entry.expansion
            or payload["definition"] != entry.definition[:2000]
            or payload["domain"] != entry.domain
            or payload["aliases"] != entry.aliases
        ):
            raise ValueError("Glossary source row does not match indexed payload")
        parts = [entry.term, entry.expansion, entry.definition]
        if entry.domain:
            parts.append(entry.domain)
        parts.extend(entry.aliases)
        return " ".join(parts)
    finally:
        if glossary_store is None:
            store.close()
