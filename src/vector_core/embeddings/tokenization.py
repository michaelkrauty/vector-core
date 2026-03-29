"""Shared tokenization utilities for sparse vector generation.

This module centralizes text tokenization logic used by both SparseVectorizer
and GlobalVocabulary to ensure consistent behavior and avoid code duplication.
"""

import re
from collections.abc import Callable

# Compiled regex patterns for tokenization (module-level for performance)
# Matches identifiers: start with letter/underscore, followed by alphanumeric/underscore
IDENTIFIER_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
# Matches camelCase parts: lowercase word optionally preceded by uppercase, or uppercase runs
CAMEL_CASE_PATTERN = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)")


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Minimum number of single-character edits to transform s1 into s2
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    previous_row: list[int] = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Compute similarity (0-1) based on Levenshtein distance.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score between 0.0 (completely different) and 1.0 (identical)
    """
    if not s1 or not s2:
        return 0.0
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len)


# Default stop tokens (common English words with low signal for code/text search)
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
