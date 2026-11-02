"""Shared tokenization utilities for sparse vector generation.

This module centralizes text tokenization and fuzzy token matching used by
both SparseVectorizer and GlobalVocabulary to ensure consistent behavior and
avoid code duplication.
"""

import re
from collections.abc import Callable, Iterable

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

# Compiled regex patterns for tokenization (module-level for performance)
# Matches identifiers: start with letter/underscore, followed by alphanumeric/underscore
IDENTIFIER_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
# Matches camelCase parts: lowercase word optionally preceded by uppercase, or uppercase runs
CAMEL_CASE_PATTERN = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)")

# Fuzzy matching compares a query token only with candidates whose length is
# within this many characters of its own.
FUZZY_LENGTH_WINDOW = 2


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.

    Insertions, deletions and substitutions each cost 1; characters are
    compared by code point.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Minimum number of single-character edits to transform s1 into s2
    """
    return Levenshtein.distance(s1, s2)


def _similarity_from_distance(distance: int, max_len: int) -> float:
    """Similarity for an edit distance between strings whose longer length is max_len."""
    return 1.0 - (distance / max_len)


def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Compute similarity (0-1) based on Levenshtein distance.

    The similarity is ``1 - distance / max(len(s1), len(s2))``.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score between 0.0 (completely different) and 1.0 (identical);
        0.0 when either string is empty
    """
    if not s1 or not s2:
        return 0.0
    return _similarity_from_distance(Levenshtein.distance(s1, s2), max(len(s1), len(s2)))


def _max_qualifying_distance(max_len: int, threshold: float) -> int:
    """Largest distance whose similarity is positive and at least threshold, or -1.

    Similarity falls as distance grows, so the qualifying distances are exactly
    0 through the returned value. Testing each distance with the same float
    expression levenshtein_similarity uses keeps the boundary identical to
    comparing the similarity itself, e.g. distance 2 at length 8 scores exactly
    0.75 and qualifies for threshold 0.75. Distance max_len scores 0.0, which
    never counts as a match.
    """
    best = -1
    for distance in range(max_len):
        if _similarity_from_distance(distance, max_len) < threshold:
            break
        best = distance
    return best


class FuzzyCandidateIndex:
    """Candidate tokens grouped by length for fuzzy matching of query tokens.

    Build it once from an ordered collection of tokens and reuse it for many
    lookups. The order of the tokens decides ties (see best_match).
    """

    __slots__ = ("_buckets",)

    def __init__(self, tokens: Iterable[str]) -> None:
        """
        Group tokens by length, keeping each token's position in the input.

        Args:
            tokens: Candidate tokens in the order that decides ties
        """
        buckets: dict[int, tuple[list[str], list[int]]] = {}
        for position, token in enumerate(tokens):
            bucket = buckets.get(len(token))
            if bucket is None:
                bucket = buckets[len(token)] = ([], [])
            bucket[0].append(token)
            bucket[1].append(position)
        self._buckets = buckets

    def best_match(self, token: str, threshold: float) -> tuple[str | None, float]:
        """
        Find the candidate most similar to token.

        Every candidate whose length is within FUZZY_LENGTH_WINDOW of the
        token's length is scored with levenshtein_similarity. |len(a) - len(b)|
        is a lower bound on edit distance, so the window only skips tokens that
        cannot be close matches; the candidates inside it are not capped, since
        any cap could drop the closest match. A candidate qualifies when its
        similarity is above 0.0 and at least threshold. The highest similarity
        wins, and among equal similarities the candidate that came first in the
        tokens the index was built from wins.

        Args:
            token: Token to match
            threshold: Minimum similarity (0-1)

        Returns:
            Tuple of (best_match, similarity) or (None, 0.0) if nothing qualifies
        """
        target_len = len(token)
        best_match: str | None = None
        best_score = 0.0
        best_position = -1

        for length in range(
            max(0, target_len - FUZZY_LENGTH_WINDOW), target_len + FUZZY_LENGTH_WINDOW + 1
        ):
            bucket = self._buckets.get(length)
            if bucket is None:
                continue
            max_len = max(target_len, length)
            max_distance = _max_qualifying_distance(max_len, threshold)
            if max_distance < 0:
                continue
            candidates, positions = bucket
            # Every candidate in this bucket shares max_len, so the smallest
            # distance is the highest similarity. extractOne returns the first
            # candidate at the smallest distance not above max_distance.
            found = process.extractOne(
                token,
                candidates,
                scorer=Levenshtein.distance,
                processor=None,
                score_cutoff=max_distance,
            )
            if found is None:
                continue
            candidate, distance, offset = found
            score = _similarity_from_distance(distance, max_len)
            position = positions[offset]
            # Buckets are visited by length, not in token order, so equal
            # scores from different buckets are resolved by position.
            if (
                best_match is None
                or score > best_score
                or (score == best_score and position < best_position)
            ):
                best_match = candidate
                best_score = score
                best_position = position

        return best_match, best_score


# Default stop tokens (common English words with low signal for code/text search)
# fmt: off
DEFAULT_STOP_TOKENS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "and", "or", "not", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "only",
    "own", "same", "so", "than", "too", "very", "just", "also",
}
# fmt: on


def default_tokenize(
    text: str,
    min_token_length: int = 2,
    stop_tokens: set[str] | None = None,
    custom_tokenizer: Callable[[str], list[str]] | None = None,
) -> list[str]:
    """
    Tokenize text for sparse vectorization.

    Handles:
    - camelCase splitting (getUserData -> get, user, data)
    - snake_case splitting (get_user_data -> get, user, data)
    - Stop word filtering
    - Minimum length filtering

    Args:
        text: Text to tokenize
        min_token_length: Minimum token length to include (default 2)
        stop_tokens: Set of stop tokens to filter. Default: DEFAULT_STOP_TOKENS
        custom_tokenizer: Optional custom tokenizer function to use instead

    Returns:
        List of normalized lowercase tokens
    """
    if custom_tokenizer:
        return custom_tokenizer(text)

    if stop_tokens is None:
        stop_tokens = DEFAULT_STOP_TOKENS

    # Split on non-alphanumeric, keeping underscores
    raw_tokens = IDENTIFIER_PATTERN.findall(text)

    tokens = []
    for token in raw_tokens:
        # Split camelCase: getUserData -> get, User, Data
        parts = CAMEL_CASE_PATTERN.findall(token)
        if parts:
            tokens.extend(parts)
        else:
            # snake_case or single word
            tokens.extend(token.split("_"))

    # Normalize and filter
    result = []
    for token in tokens:
        normalized = token.lower()
        if (
            len(normalized) >= min_token_length
            and normalized not in stop_tokens
            and not normalized.isdigit()
        ):
            result.append(normalized)

    return result
