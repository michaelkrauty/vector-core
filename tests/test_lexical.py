"""Tests for generic lexical ranking utilities."""

from vector_core.search.lexical import LexicalRankedItem, rank_lexical_items, tokenize_literal_query


def test_tokenize_literal_query_keeps_underscores_and_splits_paths_versions_and_hyphens() -> None:
    """Literal tokenizer preserves config keys while splitting common separators."""
    tokens = tokenize_literal_query("model.context_length /foo/bar gpt-5.5 HERMES_HOME snake_case")

    assert tokens == [
        "model",
        "context_length",
        "foo",
        "bar",
        "gpt",
        "5",
        "5",
        "hermes_home",
        "snake_case",
    ]


def test_rank_lexical_items_weights_title_path_tags_and_text() -> None:
    """Payload fields are weighted to favor exact title/path/tag matches over body-only hits."""
    items = [
        {
            "id": "body-only",
            "title": "Unrelated",
            "path": "concepts/other.md",
            "tags": [],
            "text": "The user asked about hermes memory provider details.",
        },
        {
            "id": "title-path",
            "title": "Hermes Memory Provider",
            "path": "concepts/hermes-memory-provider.md",
            "tags": ["memory"],
            "text": "Short page.",
        },
    ]

    ranked = rank_lexical_items(
        "Hermes memory provider",
        items,
        text=lambda item: item["text"],
        title=lambda item: item["title"],
        path=lambda item: item["path"],
        tags=lambda item: item["tags"],
    )

    assert [result.item["id"] for result in ranked] == ["title-path", "body-only"]
    assert all(isinstance(result, LexicalRankedItem) for result in ranked)
    assert ranked[0].score > ranked[1].score


def test_rank_lexical_items_filters_zero_score_items() -> None:
    """Items with no literal overlap are omitted."""
    ranked = rank_lexical_items(
        "codex review",
        [{"id": "miss", "text": "nothing related"}],
        text=lambda item: item["text"],
    )

    assert ranked == []


def test_rank_lexical_items_applies_limit_and_deterministic_tie_break() -> None:
    """Equal scores are ordered deterministically by caller-provided key."""
    items = [
        {"id": "b", "text": "codex"},
        {"id": "a", "text": "codex"},
        {"id": "c", "text": "codex"},
    ]

    ranked = rank_lexical_items(
        "codex",
        items,
        text=lambda item: item["text"],
        key=lambda item: item["id"],
        limit=2,
    )

    assert [result.key for result in ranked] == ["a", "b"]


def test_rank_lexical_items_clamps_repeated_term_counts() -> None:
    """Term-frequency saturation prevents spammy text from dominating without bound."""
    normal = {"id": "normal", "text": "codex"}
    spammy = {"id": "spammy", "text": "codex " * 100}

    ranked = rank_lexical_items(
        "codex",
        [normal, spammy],
        text=lambda item: item["text"],
        key=lambda item: item["id"],
        max_term_frequency=5,
    )

    assert ranked[0].key == "spammy"
    assert ranked[0].score == 5.0
