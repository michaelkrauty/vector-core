"""Generic deterministic lexical ranking utilities."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)

_LITERAL_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class LexicalRankedItem(Generic[T, K]):
    """One item ranked by deterministic lexical overlap."""

    item: T
    score: float
    key: K | None = None


def tokenize_literal_query(text: str) -> list[str]:
    """Tokenize text for literal lexical matching.

    Underscores stay inside tokens for config keys and environment variables,
    while punctuation, path separators, hyphens, and version dots split into
    searchable literal parts.
    """

    return [token.lower() for token in _LITERAL_TOKEN_RE.findall(text or "")]


def _coerce_tags(tags: Iterable[Any] | str | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        return [tags]
    return [str(tag) for tag in tags]


def rank_lexical_items(
    query: str,
    items: Sequence[T],
    *,
    text: Callable[[T], str],
    title: Callable[[T], str] | None = None,
    path: Callable[[T], str] | None = None,
    tags: Callable[[T], Iterable[Any] | str | None] | None = None,
    key: Callable[[T], K] | None = None,
    limit: int = 10,
    title_weight: int = 2,
    tag_weight: int = 2,
    path_weight: int = 1,
    text_weight: int = 1,
    max_term_frequency: int = 5,
) -> list[LexicalRankedItem[T, K]]:
    """Rank items by deterministic literal term overlap.

    This is a lightweight local ranker for exact names, paths, IDs, commands,
    config keys, and acronyms. It is intentionally simple and dependency-free so
    callers can use it as a sparse/lexical fallback beside dense vector search.
    """

    query_terms = Counter(tokenize_literal_query(query))
    if not query_terms:
        return []

    ranked: list[LexicalRankedItem[T, K]] = []
    max_tf = max(1, int(max_term_frequency or 1))

    for item in items:
        weighted_parts: list[str] = []
        if title is not None:
            item_title = str(title(item) or "")
            weighted_parts.extend([item_title] * max(0, int(title_weight)))
        if path is not None:
            item_path = str(path(item) or "")
            weighted_parts.extend([item_path] * max(0, int(path_weight)))
        if tags is not None:
            item_tags = " ".join(_coerce_tags(tags(item)))
            weighted_parts.extend([item_tags] * max(0, int(tag_weight)))
        weighted_parts.extend([str(text(item) or "")] * max(0, int(text_weight)))

        doc_terms = Counter(tokenize_literal_query(" ".join(weighted_parts)))
        score = 0.0
        for term, query_count in query_terms.items():
            count = doc_terms.get(term, 0)
            if count:
                score += min(count, max_tf) * query_count
        score /= max(1, sum(query_terms.values()))
        if score <= 0:
            continue
        ranked.append(
            LexicalRankedItem(
                item=item,
                score=score,
                key=key(item) if key is not None else None,
            )
        )

    def sort_key(result: LexicalRankedItem[T, K]) -> tuple[float, str]:
        return (-result.score, str(result.key) if result.key is not None else "")

    ranked.sort(key=sort_key)
    return ranked[: max(1, int(limit or 1))]
