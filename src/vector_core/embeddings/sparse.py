"""Sparse vector generation for TF-IDF-style keyword matching."""

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from vector_core.embeddings.tokenization import (
    CAMEL_CASE_PATTERN,
    DEFAULT_STOP_TOKENS,
    IDENTIFIER_PATTERN,
    levenshtein_similarity,
)


@dataclass
class SparseVector:
    """Sparse vector representation for Qdrant."""

    indices: list[int]
    values: list[float]


class SparseVectorizer:
    """
    Generate sparse vectors using TF-IDF-like weighting.

    Features:
    - Pluggable tokenization
    - Vocabulary management (save/load)
    - Incremental vocabulary updates
    - Fuzzy matching for queries
    """

    def __init__(
        self,
        min_token_length: int = 2,
        stop_tokens: set[str] | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
    ):
        """
        Initialize sparse vectorizer.

        Args:
            min_token_length: Minimum token length to include
            stop_tokens: Set of stop tokens to filter out. Default: common English words.
            tokenizer: Custom tokenizer function. Default: splits on non-alphanumeric.
        """
        self.min_token_length = min_token_length
        self.stop_tokens = stop_tokens if stop_tokens is not None else DEFAULT_STOP_TOKENS
        self._custom_tokenizer = tokenizer

        # Vocabulary: token -> index mapping
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_freq: Counter[str] = Counter()
        self._doc_count = 0

        # Vocab consistency tracking
        self._vocab_version = 1
        self._initial_vocab_size = 0
        self._tokens_added_since_full = 0

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text.

        Uses custom tokenizer if provided, otherwise default tokenization:
        - Split on non-alphanumeric
        - Handle camelCase and snake_case
        - Lowercase and filter
        """
        if self._custom_tokenizer:
            return self._custom_tokenizer(text)

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
                len(normalized) >= self.min_token_length
                and normalized not in self.stop_tokens
                and not normalized.isdigit()
            ):
                result.append(normalized)

        return result

    def fit(self, documents: list[str]) -> None:
        """
        Build vocabulary and compute IDF from corpus.

        Call this once with all document contents before vectorizing.

        Args:
            documents: List of document contents
        """
        doc_freq: Counter[str] = Counter()
        self._doc_count = len(documents)

        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] += 1

        # Build vocab and IDF
        self._vocab = {}
        self._idf = {}
        self._doc_freq = doc_freq

        for idx, (token, freq) in enumerate(doc_freq.most_common()):
            self._vocab[token] = idx
            # IDF with smoothing
            self._idf[token] = math.log((self._doc_count + 1) / (freq + 1)) + 1

        # Reset consistency tracking on full fit
        self._initial_vocab_size = len(self._vocab)
        self._tokens_added_since_full = 0
        self._vocab_version = 1

    def extend_vocab(self, new_documents: list[str]) -> int:
        """
        Incrementally extend vocabulary with new documents.

        Adds new tokens and updates IDF scores for both new and existing tokens.
        Call this during incremental indexing to keep vocabulary current.

        Args:
            new_documents: New document contents to incorporate

        Returns:
            Number of new tokens added to vocabulary
        """
        if not new_documents:
            return 0

        # Count new document frequencies
        new_doc_freq: Counter[str] = Counter()
        for doc in new_documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                new_doc_freq[token] += 1

        # Update counts
        self._doc_count += len(new_documents)

        # Track new tokens
        new_tokens: list[str] = []
        next_idx = max(self._vocab.values()) + 1 if self._vocab else 0

        for token, freq in new_doc_freq.items():
            self._doc_freq[token] += freq

            if token not in self._vocab:
                # New token - add to vocabulary
                self._vocab[token] = next_idx
                next_idx += 1
                new_tokens.append(token)

        # Recompute IDF for the ENTIRE vocabulary, not just the tokens in the
        # new batch: _doc_count (N) grew, so every token's IDF changed, even
        # ones absent from new_documents. Recomputing only new_doc_freq left
        # existing tokens with an IDF computed against the smaller corpus,
        # under-weighting most of the vocabulary. (fit() recomputes the full
        # vocabulary for the same reason.)
        for token, freq in self._doc_freq.items():
            self._idf[token] = math.log((self._doc_count + 1) / (freq + 1)) + 1

        # Track cumulative vocab growth
        self._tokens_added_since_full += len(new_tokens)

        return len(new_tokens)

    def vocab_growth_ratio(self) -> float:
        """
        Calculate vocab growth ratio since last full index.

        Returns:
            Ratio of new tokens added to initial vocab size.
            0.0 if no initial vocab or no growth.
        """
        if self._initial_vocab_size == 0:
            return 0.0
        return self._tokens_added_since_full / self._initial_vocab_size

    def needs_reindex(self, threshold: float = 0.10) -> bool:
        """
        Check if a full reindex is recommended due to vocab drift.

        When vocabulary grows significantly through incremental indexing,
        old chunks may have stale sparse vectors (outdated IDF weights).

        Args:
            threshold: Growth ratio threshold (default 0.10 = 10%)

        Returns:
            True if vocab has grown beyond threshold since last full index
        """
        return self.vocab_growth_ratio() > threshold

    def vectorize(self, text: str) -> SparseVector:
        """
        Convert text to sparse vector using TF-IDF weights.

        Must call fit() first, or use vectorize_query() for queries.

        Args:
            text: Text to vectorize

        Returns:
            Sparse vector representation
        """
        tokens = self._tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])

        # Term frequency
        tf = Counter(tokens)

        indices = []
        values = []

        for token, count in tf.items():
            if token in self._vocab:
                idx = self._vocab[token]
                # TF-IDF: tf * idf (using log-normalized TF)
                tf_weight = 1 + math.log(count) if count > 0 else 0
                idf_weight = self._idf.get(token, 1.0)
                weight = tf_weight * idf_weight

                indices.append(idx)
                values.append(weight)

        # Sort by index for Qdrant
        if indices:
            pairs = sorted(zip(indices, values, strict=True))
            sorted_indices, sorted_values = zip(*pairs, strict=True)
            indices = list(sorted_indices)
            values = list(sorted_values)

        return SparseVector(indices=indices, values=values)

    def _find_fuzzy_match(
        self,
        token: str,
        threshold: float = 0.75,
        max_candidates: int = 500,
    ) -> tuple[str | None, float]:
        """
        Find closest vocabulary token using Levenshtein similarity.

        Args:
            token: Token to match
            threshold: Minimum similarity threshold (0-1)
            max_candidates: Max vocabulary terms to check

        Returns:
            Tuple of (best_match, similarity) or (None, 0.0) if no match
        """
        if not self._vocab:
            return None, 0.0

        target_len = len(token)
        best_match = None
        best_score = 0.0

        # Only check tokens of similar length for performance
        candidates = [
            t for t in self._vocab
            if abs(len(t) - target_len) <= 2
        ][:max_candidates]

        for candidate in candidates:
            score = levenshtein_similarity(token, candidate)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = candidate

        return best_match, best_score

    def vectorize_query(
        self,
        query: str,
        fuzzy: bool = True,
        fuzzy_threshold: float = 0.75,
    ) -> SparseVector:
        """
        Vectorize a search query.

        For queries, uses simpler weighting (just presence, no TF).
        Unknown tokens can be fuzzy-matched to closest vocabulary term.

        Args:
            query: Query string
            fuzzy: Whether to use fuzzy matching for unknown tokens
            fuzzy_threshold: Minimum similarity for fuzzy matches (0-1)

        Returns:
            Sparse vector representation
        """
        tokens = self._tokenize(query)
        if not tokens:
            return SparseVector(indices=[], values=[])

        # Deduplicate by both token AND index
        seen_tokens: set[str] = set()
        seen_indices: set[int] = set()
        indices: list[int] = []
        values: list[float] = []

        for token in tokens:
            if token in seen_tokens:
                continue
            seen_tokens.add(token)

            if token in self._vocab:
                # Exact match
                idx = self._vocab[token]
                if idx in seen_indices:
                    continue
                seen_indices.add(idx)
                weight = self._idf.get(token, 1.0)
                indices.append(idx)
                values.append(weight)
            elif fuzzy:
                # Try fuzzy match
                match, similarity = self._find_fuzzy_match(token, fuzzy_threshold)
                if match:
                    idx = self._vocab[match]
                    if idx in seen_indices:
                        continue
                    seen_indices.add(idx)
                    # Discount weight by similarity
                    weight = self._idf.get(match, 1.0) * similarity
                    indices.append(idx)
                    values.append(weight)

        # Sort by index (required by Qdrant)
        if indices:
            pairs = sorted(zip(indices, values, strict=True))
            sorted_indices, sorted_values = zip(*pairs, strict=True)
            indices = list(sorted_indices)
            values = list(sorted_values)

        return SparseVector(indices=indices, values=values)

    def save_vocab(self) -> dict:
        """Export vocabulary for persistence."""
        return {
            "vocab": self._vocab,
            "idf": self._idf,
            "doc_count": self._doc_count,
            "doc_freq": dict(self._doc_freq),
            "vocab_version": self._vocab_version,
            "initial_vocab_size": self._initial_vocab_size,
            "tokens_added_since_full": self._tokens_added_since_full,
        }

    def load_vocab(self, data: dict) -> None:
        """Load vocabulary from persistence."""
        self._vocab = data["vocab"]
        self._idf = data["idf"]
        self._doc_count = data["doc_count"]
        self._doc_freq = Counter(data.get("doc_freq", {}))
        self._vocab_version = data.get("vocab_version", 1)
        self._initial_vocab_size = data.get("initial_vocab_size", len(self._vocab))
        self._tokens_added_since_full = data.get("tokens_added_since_full", 0)
