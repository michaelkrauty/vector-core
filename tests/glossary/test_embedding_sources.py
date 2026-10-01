"""Legacy shared input recovery must prove an exact historical input."""

import pytest

from vector_core.glossary.indexer import GlossaryIndexer, _generate_embedding_content
from vector_core.glossary.store import GlossaryStore
from vector_core.storage.embedding_sources import resolve_shared_embedding_text


@pytest.fixture
def glossary(tmp_path):
    store = GlossaryStore(db_path=tmp_path / "glossary.db")
    try:
        yield store
    finally:
        store.close()


def legacy_payload(entry):
    payload = GlossaryIndexer._create_payload(entry)
    del payload["embedding_text"]
    return payload


async def test_recovers_full_long_definition(glossary):
    entry = glossary.create("API", "Interface", "x" * 5000, "software", ["First", "Second"])
    payload = legacy_payload(entry)
    assert len(payload["definition"]) == 2000
    assert await resolve_shared_embedding_text(payload, glossary_store=glossary) == (
        _generate_embedding_content(entry)
    )


@pytest.mark.parametrize(
    "field", ["entry_hash", "term", "expansion", "definition", "domain", "aliases"]
)
async def test_rejects_changed_embedding_fields(glossary, field):
    entry = glossary.create("API", "Interface", "definition", "software", ["First", "Second"])
    payload = legacy_payload(entry)
    payload[field] = ["Second", "First"] if field == "aliases" else "changed"
    with pytest.raises(ValueError, match="does not match"):
        await resolve_shared_embedding_text(payload, glossary_store=glossary)


@pytest.mark.parametrize("field", ["glossary_id", "entry_hash", "domain", "aliases"])
async def test_rejects_missing_proof(glossary, field):
    entry = glossary.create("API", "Interface", "definition")
    payload = legacy_payload(entry)
    del payload[field]
    with pytest.raises(ValueError, match="missing"):
        await resolve_shared_embedding_text(payload, glossary_store=glossary)


async def test_rejects_deleted_source_row(glossary):
    entry = glossary.create("API", "Interface", "definition")
    payload = legacy_payload(entry)
    glossary.delete(entry.id)
    with pytest.raises(ValueError, match="unavailable"):
        await resolve_shared_embedding_text(payload, glossary_store=glossary)


async def test_persisted_text_survives_source_changes(glossary):
    entry = glossary.create("API", "Interface", "x" * 5000)
    payload = GlossaryIndexer._create_payload(entry)
    glossary.delete(entry.id)
    assert await resolve_shared_embedding_text(payload, glossary_store=glossary) == (
        _generate_embedding_content(entry)
    )


async def test_fact_uses_exact_content_without_reconstruction():
    content = "subject  original_predicate\nobject context "
    assert await resolve_shared_embedding_text({"type": "fact", "content": content}) == content


async def test_legacy_note_has_canonical_metadata_only_reconstruction():
    payload = {"type": "note", "title": "Title", "tags": ["one", "two"], "category": "Work"}
    assert await resolve_shared_embedding_text(payload) == "Title\nTags: one, two\nCategory: Work"


async def test_legacy_note_chunk_uses_retained_content():
    payload = {"type": "chunk", "note_id": "note", "content": "original retained chunk"}
    assert await resolve_shared_embedding_text(payload) == payload["content"]


@pytest.mark.parametrize(
    "payload", [{"type": "fact"}, {"type": "fact", "content": 1}, {"type": "unknown"}]
)
async def test_rejects_unrecoverable_payloads(payload):
    with pytest.raises(ValueError):
        await resolve_shared_embedding_text(payload)
