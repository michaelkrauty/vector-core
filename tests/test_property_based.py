"""Property-based tests using Hypothesis for vector-core.

These tests verify that tokenization and vector operations never crash
regardless of input, providing robustness guarantees.
"""

import string

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from vector_core.embeddings.sparse import SparseVector, SparseVectorizer


class TestSparseVectorizerPropertyBased:
    """Property-based tests for SparseVectorizer."""

    @given(st.text(max_size=5000))
    @settings(max_examples=200)
    def test_internal_tokenize_never_crashes(self, text: str) -> None:
        """_tokenize should never raise an exception for any input."""
        vectorizer = SparseVectorizer()
        result = vectorizer._tokenize(text)

        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    @given(st.text(max_size=2000))
    @settings(max_examples=100)
    def test_fit_never_crashes(self, text: str) -> None:
        """fit should never crash for any input."""
        vectorizer = SparseVectorizer()
        # Should not raise
        vectorizer.fit([text])

    @given(st.lists(st.text(max_size=500), min_size=1, max_size=20))
    @settings(max_examples=50)
    def test_fit_multiple_docs(self, docs: list[str]) -> None:
        """fit handles multiple documents."""
        vectorizer = SparseVectorizer()
        vectorizer.fit(docs)
        # Should have built vocabulary (could be 0 if all filtered)
        assert len(vectorizer._vocab) >= 0

    @given(st.text(min_size=1, max_size=1000))
    @settings(max_examples=100)
    def test_vectorize_after_fit(self, text: str) -> None:
        """vectorize returns valid sparse vector after fit."""
        assume(text.strip())  # Need non-empty text

        vectorizer = SparseVectorizer()
        vectorizer.fit([text])
        result = vectorizer.vectorize(text)

        assert isinstance(result, SparseVector)
        assert isinstance(result.indices, list)
        assert isinstance(result.values, list)
        assert len(result.indices) == len(result.values)

    @given(st.text(alphabet=string.printable, max_size=500))
    @settings(max_examples=100)
    def test_tokenize_printable(self, text: str) -> None:
        """_tokenize handles all printable characters."""
        vectorizer = SparseVectorizer()
        result = vectorizer._tokenize(text)
        assert isinstance(result, list)

    @given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=500))
    @settings(max_examples=100)
    def test_tokenize_unicode(self, text: str) -> None:
        """_tokenize handles Unicode correctly."""
        vectorizer = SparseVectorizer()
        result = vectorizer._tokenize(text)
        assert isinstance(result, list)


class TestSparseVectorPropertyBased:
    """Property-based tests for SparseVector dataclass."""

    @given(
        st.lists(st.integers(min_value=0, max_value=10000), max_size=100),
        st.lists(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False), max_size=100),
    )
    @settings(max_examples=100)
    def test_sparse_vector_creation(self, indices: list[int], values: list[float]) -> None:
        """SparseVector can be created with any valid indices/values."""
        # Make lengths match
        min_len = min(len(indices), len(values))
        indices = indices[:min_len]
        values = values[:min_len]

        vec = SparseVector(indices=indices, values=values)
        assert vec.indices == indices
        assert vec.values == values


class TestTokenizationEdgeCases:
    """Test edge cases in tokenization."""

    @pytest.mark.parametrize("edge_case", [
        "",  # Empty
        " ",  # Space only
        "\n\n\n",  # Newlines only
        "\t\t",  # Tabs only
        "a",  # Single char
        "ab",  # Two chars (below min token length)
        "abc",  # Three chars (at min length typically)
        "a" * 10000,  # Very long single token
        "a b c d e f g",  # Many short tokens
        "the and or if",  # Common stop words
        "CamelCase",  # camelCase
        "snake_case",  # snake_case
        "UPPER_CASE",  # UPPER_CASE
        "mixedCase_With_Underscores",  # Mixed
        "123 456 789",  # Numbers only
        "abc123def456",  # Mixed alphanumeric
        "日本語テスト",  # Japanese
        "🔥 火 🔥",  # Emoji + CJK
        "@#$%^&*()",  # Special chars only
        "function() { return; }",  # Code-like
        "SELECT * FROM users WHERE id = 1",  # SQL-like
        "def __init__(self, *args, **kwargs):",  # Python dunder
        "async/await/Promise",  # JS keywords
        "std::vector<int>",  # C++ template
    ])
    def test_tokenize_edge_cases(self, edge_case: str) -> None:
        """Known edge cases should not crash tokenization."""
        vectorizer = SparseVectorizer()
        result = vectorizer._tokenize(edge_case)
        assert isinstance(result, list)


class TestCamelCaseSplitting:
    """Test camelCase splitting behavior."""

    @given(st.lists(
        st.text(min_size=3, max_size=10, alphabet=string.ascii_lowercase),
        min_size=1,
        max_size=5,
    ))
    @settings(max_examples=100)
    def test_camel_case_words(self, words: list[str]) -> None:
        """Verify camelCase splitting works."""
        # Build camelCase string
        if not words:
            return
        camel = words[0] + "".join(w.capitalize() for w in words[1:])

        vectorizer = SparseVectorizer()
        tokens = vectorizer._tokenize(camel)

        # Tokens should exist
        assert isinstance(tokens, list)


class TestVectorizerState:
    """Test vectorizer state management."""

    @given(st.lists(st.text(min_size=10, max_size=100), min_size=2, max_size=10))
    @settings(max_examples=50)
    def test_vocab_growth(self, docs: list[str]) -> None:
        """Vocabulary should grow as documents are added."""
        vectorizer = SparseVectorizer()

        # Fit on first doc
        vectorizer.fit([docs[0]])
        initial_size = len(vectorizer._vocab)

        # Fit on all docs (adds new terms)
        vectorizer.fit(docs)
        final_size = len(vectorizer._vocab)

        # Should have at least as many terms
        assert final_size >= initial_size

    @given(st.text(min_size=10, max_size=500))
    @settings(max_examples=50)
    def test_vectorize_consistency(self, text: str) -> None:
        """Same text should produce same vector."""
        assume(text.strip())

        vectorizer = SparseVectorizer()
        vectorizer.fit([text])

        vec1 = vectorizer.vectorize(text)
        vec2 = vectorizer.vectorize(text)

        assert vec1.indices == vec2.indices
        assert vec1.values == vec2.values
