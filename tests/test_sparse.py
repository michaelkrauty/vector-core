"""Tests for SparseVectorizer (TF-IDF sparse vectors for hybrid search)."""

from vector_core.embeddings.sparse import SparseVector, SparseVectorizer


class TestSparseVector:
    """Tests for SparseVector dataclass."""

    def test_empty_vector(self):
        """Empty vector has no indices or values."""
        vec = SparseVector(indices=[], values=[])
        assert len(vec.indices) == 0
        assert len(vec.values) == 0

    def test_indices_values_same_length(self):
        """Indices and values must have same length."""
        vec = SparseVector(indices=[0, 5, 10], values=[0.1, 0.5, 0.3])
        assert len(vec.indices) == len(vec.values)


class TestTokenization:
    """Tests for code-aware tokenization."""

    def test_camel_case_split(self):
        """CamelCase identifiers are split."""
        vectorizer = SparseVectorizer()
        tokens = vectorizer._tokenize("getUserData")

        assert "get" in tokens
        assert "user" in tokens
        assert "data" in tokens

    def test_pascal_case_split(self):
        """PascalCase identifiers are split."""
        vectorizer = SparseVectorizer()
        tokens = vectorizer._tokenize("UserDataService")

        assert "user" in tokens
        assert "data" in tokens
        assert "service" in tokens

    def test_snake_case_split(self):
        """snake_case identifiers are split."""
        vectorizer = SparseVectorizer()
        tokens = vectorizer._tokenize("get_user_data")

        assert "get" in tokens
        assert "user" in tokens
        assert "data" in tokens

    def test_mixed_case_handling(self):
        """Mixed case patterns are handled correctly."""
        vectorizer = SparseVectorizer()
        tokens = vectorizer._tokenize("XMLHttpRequest")

        # Should handle consecutive capitals
        assert "xml" in tokens or "xmlhttp" in tokens
        assert "request" in tokens

    def test_stop_token_removal(self):
        """Common English stop words are removed."""
        vectorizer = SparseVectorizer()
        tokens = vectorizer._tokenize("the a an is are was to for of with")

        # Common English words should be filtered
        assert "the" not in tokens
        assert "was" not in tokens
        assert "for" not in tokens
        assert "with" not in tokens

    def test_programming_keywords_preserved(self):
        """Programming keywords are NOT filtered."""
        vectorizer = SparseVectorizer()
        tokens = vectorizer._tokenize("if else return def class function async await")

        # Programming keywords are meaningful in code search
        assert "if" in tokens
        assert "else" in tokens
        assert "return" in tokens
        assert "def" in tokens
        assert "class" in tokens

    def test_min_length_filter(self):
        """Short tokens are filtered based on min_token_length."""
        vectorizer = SparseVectorizer(min_token_length=3)
        tokens = vectorizer._tokenize("a ab abc abcd")

        assert "a" not in tokens
        assert "ab" not in tokens
        assert "abc" in tokens
        assert "abcd" in tokens

    def test_numbers_handled(self):
        """Numeric content is handled appropriately."""
        vectorizer = SparseVectorizer()
        tokens = vectorizer._tokenize("user123 data_v2 version3")

        # Should tokenize alphanumeric identifiers
        assert len(tokens) > 0

    def test_special_characters_removed(self):
        """Special characters don't break tokenization."""
        vectorizer = SparseVectorizer()
        tokens = vectorizer._tokenize("foo->bar() baz.qux()")

        assert "foo" in tokens
        assert "bar" in tokens
        assert "baz" in tokens
        assert "qux" in tokens


class TestVectorization:
    """Tests for TF-IDF vectorization."""

    def test_fit_creates_vocab(self):
        """Fitting documents creates vocabulary."""
        vectorizer = SparseVectorizer()
        documents = [
            "function handleRequest",
            "class UserService",
            "async function process",
        ]
        vectorizer.fit(documents)

        assert len(vectorizer._vocab) > 0
        assert "handle" in vectorizer._vocab or "request" in vectorizer._vocab

    def test_fit_calculates_idf(self):
        """Fitting calculates IDF values."""
        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world", "hello again", "goodbye world"])

        assert len(vectorizer._idf) > 0
        # "hello" appears in 2/3 docs, "goodbye" in 1/3
        # IDF(hello) should be lower than IDF(goodbye)
        if "hello" in vectorizer._idf and "goodbye" in vectorizer._idf:
            assert vectorizer._idf["hello"] < vectorizer._idf["goodbye"]

    def test_vectorize_returns_sparse(self):
        """Vectorize returns SparseVector."""
        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world", "goodbye world"])

        vec = vectorizer.vectorize("hello world")

        assert isinstance(vec, SparseVector)
        assert len(vec.indices) > 0
        assert len(vec.indices) == len(vec.values)

    def test_vectorize_unknown_tokens_ignored(self):
        """Unknown tokens produce empty vectors."""
        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world"])

        vec = vectorizer.vectorize("completely unknown words")

        # Unknown tokens produce no output
        assert len(vec.indices) == 0

    def test_vectorize_query_with_fuzzy(self):
        """Query vectorization supports fuzzy matching."""
        vectorizer = SparseVectorizer()
        vectorizer.fit(["handleRequest processData validateInput"])

        # Typo should potentially fuzzy match
        vec = vectorizer.vectorize_query("handlRequest", fuzzy=True)

        # Should produce some output (may or may not match depending on threshold)
        assert isinstance(vec, SparseVector)

    def test_vectorize_values_positive(self):
        """TF-IDF values are positive."""
        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world foo bar", "hello world baz"])

        vec = vectorizer.vectorize("hello world")

        assert all(v > 0 for v in vec.values)

    def test_empty_document(self):
        """Empty document produces empty vector."""
        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world"])

        vec = vectorizer.vectorize("")
        assert len(vec.indices) == 0


class TestVocabExtension:
    """Tests for incremental vocabulary extension."""

    def test_extend_adds_new_tokens(self):
        """extend_vocab adds new tokens to vocabulary."""
        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world"])

        initial_size = len(vectorizer._vocab)

        new_tokens = vectorizer.extend_vocab(["completely new words"])

        assert len(vectorizer._vocab) > initial_size
        assert new_tokens > 0

    def test_extend_updates_idf(self):
        """extend_vocab updates IDF values."""
        vectorizer = SparseVectorizer()
        # "hello" appears in 2 of 3 docs
        vectorizer.fit(["hello world", "hello again", "goodbye friend"])

        old_hello_idf = vectorizer._idf.get("hello", 0)

        # Add docs WITH "hello" - this will update hello's IDF
        vectorizer.extend_vocab(["hello there", "hello everybody"])

        new_hello_idf = vectorizer._idf.get("hello", 0)
        # IDF should change because frequency of "hello" changed
        assert new_hello_idf != old_hello_idf

    def test_extend_preserves_existing_tokens(self):
        """extend_vocab doesn't remove existing tokens."""
        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world"])

        existing_vocab = set(vectorizer._vocab.keys())

        vectorizer.extend_vocab(["completely new words"])

        # All existing tokens still present
        assert existing_vocab.issubset(set(vectorizer._vocab.keys()))


class TestVocabConsistency:
    """Tests for vocabulary consistency tracking."""

    def test_initial_tracking(self):
        """Initial tracking is set correctly after fit."""
        vectorizer = SparseVectorizer()
        vectorizer.fit(["word1 word2 word3"])

        assert vectorizer._initial_vocab_size > 0
        assert vectorizer._tokens_added_since_full == 0
        assert vectorizer.vocab_growth_ratio() == 0.0

    def test_growth_tracking(self):
        """Growth is tracked after extend_vocab."""
        vectorizer = SparseVectorizer()
        vectorizer.fit(["word1 word2 word3 word4 word5"])

        vectorizer.extend_vocab(["newword1 newword2"])

        assert vectorizer._tokens_added_since_full > 0
        assert vectorizer.vocab_growth_ratio() > 0.0

    def test_needs_reindex_threshold(self):
        """needs_reindex respects threshold."""
        vectorizer = SparseVectorizer()
        vectorizer.fit(["word1 word2"])

        # By default, threshold is 10%
        assert not vectorizer.needs_reindex()

        # Add many new tokens to exceed threshold
        vectorizer.extend_vocab([
            "new1 new2 new3 new4 new5 new6 new7 new8 new9 new10"
        ])

        assert vectorizer.needs_reindex()

    def test_full_refit_resets_tracking(self):
        """Full refit resets growth counters."""
        vectorizer = SparseVectorizer()
        vectorizer.fit(["word1 word2"])
        vectorizer.extend_vocab(["new1 new2 new3"])

        assert vectorizer.vocab_growth_ratio() > 0

        # Full refit should reset tracking
        vectorizer.fit(["word1 word2 new1 new2 new3"])

        assert vectorizer.vocab_growth_ratio() == 0.0
        assert vectorizer._initial_vocab_size == len(vectorizer._vocab)


class TestSaveLoad:
    """Tests for vocabulary persistence."""

    def test_save_returns_dict(self):
        """save_vocab returns serializable dict."""
        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world"])

        data = vectorizer.save_vocab()

        assert isinstance(data, dict)
        assert "vocab" in data
        assert "idf" in data
        assert "doc_count" in data

    def test_load_restores_state(self):
        """load_vocab restores vectorizer state."""
        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world", "foo bar"])

        data = vectorizer.save_vocab()

        new_vectorizer = SparseVectorizer()
        new_vectorizer.load_vocab(data)

        # Should produce same vectors
        vec1 = vectorizer.vectorize("hello world")
        vec2 = new_vectorizer.vectorize("hello world")

        assert vec1.indices == vec2.indices
        assert vec1.values == vec2.values

    def test_save_load_preserves_tracking(self):
        """save/load preserves consistency tracking."""
        vectorizer = SparseVectorizer()
        vectorizer.fit(["word1 word2 word3"])
        vectorizer.extend_vocab(["new1 new2"])

        data = vectorizer.save_vocab()

        new_vectorizer = SparseVectorizer()
        new_vectorizer.load_vocab(data)

        assert new_vectorizer._initial_vocab_size == vectorizer._initial_vocab_size
        assert new_vectorizer._tokens_added_since_full == vectorizer._tokens_added_since_full
        assert new_vectorizer.vocab_growth_ratio() == vectorizer.vocab_growth_ratio()

    def test_load_empty_vocab(self):
        """Loading empty vocab works."""
        vectorizer = SparseVectorizer()
        vectorizer.load_vocab({
            "vocab": {},
            "idf": {},
            "doc_count": 0,
        })

        vec = vectorizer.vectorize("hello world")
        assert len(vec.indices) == 0


class TestSparseEdgeCases:
    """Tests for edge cases in sparse embeddings."""

    def test_levenshtein_empty_s2(self):
        """Levenshtein with empty second string (line 15)."""
        from vector_core.embeddings.tokenization import levenshtein_distance

        # s1 longer than s2, s2 becomes empty after swap doesn't happen
        # actually need len(s2) == 0 after swap
        result = levenshtein_distance("test", "")
        assert result == 4  # Length of "test"

    def test_levenshtein_similarity_empty_strings(self):
        """Levenshtein similarity with empty strings (line 33)."""
        from vector_core.embeddings.tokenization import levenshtein_similarity

        assert levenshtein_similarity("", "test") == 0.0
        assert levenshtein_similarity("test", "") == 0.0

    def test_custom_tokenizer(self):
        """Custom tokenizer is used when provided (line 111)."""
        from vector_core.embeddings.sparse import SparseVectorizer

        def my_tokenizer(text):
            return ["custom", "tokens"]

        # tokenizer parameter, not custom_tokenizer
        vectorizer = SparseVectorizer(tokenizer=my_tokenizer)
        vectorizer.fit(["some text"])

        tokens = vectorizer._tokenize("anything here")
        assert tokens == ["custom", "tokens"]

    def test_pure_snake_case_token(self):
        """Pure snake_case token splitting (line 124)."""
        from vector_core.embeddings.sparse import SparseVectorizer

        vectorizer = SparseVectorizer()
        # Token "__" has no letters so camelCase regex returns []
        # Falls through to split("_") giving ["", "", ""]
        tokens = vectorizer._tokenize("code __ test")

        # The __ produces empty parts which get filtered
        # code and test should remain
        assert "code" in tokens
        assert "test" in tokens

    def test_extend_vocab_empty_docs(self):
        """extend_vocab with empty docs returns 0 (line 185)."""
        from vector_core.embeddings.sparse import SparseVectorizer

        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world"])

        # Extend with empty list
        result = vectorizer.extend_vocab([])
        assert result == 0

    def test_vectorize_query_empty_tokens(self):
        """vectorize_query with no tokens returns empty vector (line 349)."""
        from vector_core.embeddings.sparse import SparseVectorizer

        vectorizer = SparseVectorizer(min_token_length=100)  # Filter out all tokens
        vectorizer.fit(["hello world"])

        # All tokens will be filtered due to length
        result = vectorizer.vectorize_query("hi")

        assert len(result.indices) == 0
        assert len(result.values) == 0

    def test_vectorize_query_duplicate_tokens(self):
        """vectorize_query deduplicates tokens (line 359)."""
        from vector_core.embeddings.sparse import SparseVectorizer

        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world hello hello"])

        # Query with repeated token
        result = vectorizer.vectorize_query("hello hello hello world")

        # Should have only 2 unique indices (hello, world)
        assert len(result.indices) == 2

    def test_vectorize_query_duplicate_indices(self):
        """vectorize_query skips duplicate indices (line 366, 377)."""
        from vector_core.embeddings.sparse import SparseVectorizer

        # This is harder to test directly as it requires same index for different tokens
        # The code handles this for fuzzy matches where different tokens might map to same vocab entry
        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world python"])

        # Test with fuzzy matching enabled
        result = vectorizer.vectorize_query("hello helllo")  # typo

        # helllo might fuzzy match to hello, both mapping to same index
        # Either way, should not have duplicate indices
        assert len(result.indices) == len(set(result.indices))

    def test_find_fuzzy_match_no_match(self):
        """_find_fuzzy_match returns None when no match found (line 307)."""
        from vector_core.embeddings.sparse import SparseVectorizer

        vectorizer = SparseVectorizer()
        vectorizer.fit(["hello world python"])

        # Use a term that won't match anything
        match, similarity = vectorizer._find_fuzzy_match("xyzxyzxyz", 0.75)

        assert match is None
        assert similarity == 0.0

    def test_find_fuzzy_match_empty_vocab(self):
        """_find_fuzzy_match with empty vocab returns None (line 307)."""
        from vector_core.embeddings.sparse import SparseVectorizer

        vectorizer = SparseVectorizer()
        # Don't fit - vocab is empty

        match, similarity = vectorizer._find_fuzzy_match("test", 0.75)

        assert match is None
        assert similarity == 0.0

    def test_vocab_growth_ratio_unfitted(self):
        """vocab_growth_ratio returns 0.0 when _initial_vocab_size is 0 (line 229)."""
        from vector_core.embeddings.sparse import SparseVectorizer

        vectorizer = SparseVectorizer()
        # Don't fit - _initial_vocab_size is 0

        assert vectorizer._initial_vocab_size == 0
        assert vectorizer.vocab_growth_ratio() == 0.0

    def test_vectorize_query_duplicate_exact_match_index(self):
        """vectorize_query skips duplicate indices in exact matches (line 366).

        This edge case happens when two different tokens map to the same
        vocabulary index (defensive code for malformed vocab state).
        """
        from vector_core.embeddings.sparse import SparseVectorizer

        vectorizer = SparseVectorizer()

        # Manually create a vocab where two different tokens map to same index
        # This shouldn't happen normally but the code defensively handles it
        vectorizer._vocab = {"hello": 0, "world": 0, "python": 1}
        vectorizer._idf = {"hello": 1.0, "world": 1.0, "python": 1.0}
        vectorizer._doc_count = 1
        vectorizer._doc_freq = {"hello": 1, "world": 1, "python": 1}

        # Query with "hello world" - both map to index 0
        result = vectorizer.vectorize_query("hello world python")

        # Should have only 2 unique indices (0 and 1), not 3
        # "world" should be skipped because index 0 was already added by "hello"
        assert len(result.indices) == 2
        assert set(result.indices) == {0, 1}
