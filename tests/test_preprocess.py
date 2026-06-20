"""Tests for query preprocessing."""

import pytest

from vector_core.search.preprocess import (
    DEFAULT_SYNONYMS,
    ProcessedQuery,
    QueryPreprocessor,
    create_default_preprocessor,
)


class TestProcessedQuery:
    """Tests for ProcessedQuery dataclass."""

    def test_default_values(self):
        """ProcessedQuery has sensible defaults."""
        query = ProcessedQuery(text="test", original="test")

        assert query.text == "test"
        assert query.original == "test"
        assert query.fields == {}
        assert query.expanded_terms == []

    def test_with_all_fields(self):
        """ProcessedQuery holds all components."""
        query = ProcessedQuery(
            text="expanded query",
            original="original query",
            fields={"tag": "important"},
            expanded_terms=["synonym1", "synonym2"],
        )

        assert query.text == "expanded query"
        assert query.original == "original query"
        assert query.fields == {"tag": "important"}
        assert query.expanded_terms == ["synonym1", "synonym2"]


class TestCamelCaseExpansion:
    """Tests for camelCase expansion."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor without synonyms."""
        return QueryPreprocessor()

    def test_camelcase_expansion(self, preprocessor):
        """Expands camelCase to words."""
        text, terms = preprocessor.expand_camelcase("getUserData")

        assert "get" in text.lower()
        assert "user" in text.lower()
        assert "data" in text.lower()
        assert len(terms) > 0

    def test_pascalcase_expansion(self, preprocessor):
        """Expands PascalCase to words."""
        text, terms = preprocessor.expand_camelcase("UserDataService")

        assert "user" in text.lower()
        assert "data" in text.lower()
        assert "service" in text.lower()

    def test_acronym_handling(self, preprocessor):
        """Handles acronyms like XMLParser."""
        text, terms = preprocessor.expand_camelcase("XMLParser")

        assert "xml" in text.lower()
        assert "parser" in text.lower()

    def test_no_expansion_needed(self, preprocessor):
        """Returns unchanged if no camelCase."""
        text, terms = preprocessor.expand_camelcase("simple words")

        assert text == "simple words"
        assert terms == []

    def test_single_word(self, preprocessor):
        """Single lowercase word unchanged."""
        text, terms = preprocessor.expand_camelcase("hello")

        assert text == "hello"
        assert terms == []


class TestSynonymExpansion:
    """Tests for synonym expansion."""

    def test_synonym_expansion(self):
        """Expands words with synonyms."""
        preprocessor = QueryPreprocessor(
            synonyms={"fn": ["function"], "cls": ["class"]}
        )

        text, terms = preprocessor.expand_synonyms("define fn in cls")

        assert "function" in text
        assert "class" in text
        assert "function" in terms
        assert "class" in terms

    def test_no_synonyms_configured(self):
        """Returns unchanged if no synonyms."""
        preprocessor = QueryPreprocessor(synonyms={})

        text, terms = preprocessor.expand_synonyms("fn cls")

        assert text == "fn cls"
        assert terms == []

    def test_no_matching_synonyms(self):
        """Returns unchanged if no matches."""
        preprocessor = QueryPreprocessor(
            synonyms={"fn": ["function"]}
        )

        text, terms = preprocessor.expand_synonyms("hello world")

        assert text == "hello world"
        assert terms == []

    def test_preserves_non_words(self):
        """Preserves punctuation and whitespace."""
        preprocessor = QueryPreprocessor(
            synonyms={"fn": ["function"]}
        )

        text, terms = preprocessor.expand_synonyms("call fn(x, y)")

        assert "function" in text
        assert "(" in text
        assert "," in text


class TestFieldExtraction:
    """Tests for field:value extraction."""

    def test_extract_single_field(self):
        """Extracts single field:value pair."""
        preprocessor = QueryPreprocessor(field_prefixes=["tag:"])

        remaining, fields = preprocessor.extract_fields("tag:important find error")

        assert remaining == "find error"
        assert fields == {"tag": "important"}

    def test_extract_multiple_fields(self):
        """Extracts multiple field:value pairs."""
        preprocessor = QueryPreprocessor(
            field_prefixes=["tag:", "category:"]
        )

        remaining, fields = preprocessor.extract_fields(
            "tag:urgent category:bugs search query"
        )

        assert remaining == "search query"
        assert fields["tag"] == "urgent"
        assert fields["category"] == "bugs"

    def test_no_fields(self):
        """Returns original if no fields."""
        preprocessor = QueryPreprocessor(field_prefixes=["tag:"])

        remaining, fields = preprocessor.extract_fields("plain search query")

        assert remaining == "plain search query"
        assert fields == {}

    def test_field_at_end(self):
        """Extracts field at end of query."""
        preprocessor = QueryPreprocessor(field_prefixes=["path:"])

        remaining, fields = preprocessor.extract_fields("find file path:src/main")

        assert "find file" in remaining
        assert fields["path"] == "src/main"

    def test_unrecognized_prefix(self):
        """Ignores unrecognized prefixes."""
        preprocessor = QueryPreprocessor(field_prefixes=["tag:"])

        remaining, fields = preprocessor.extract_fields("unknown:value search")

        assert "unknown:value" in remaining
        assert "unknown" not in fields


class TestFullPreprocess:
    """Tests for full preprocessing pipeline."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor with all features."""
        return QueryPreprocessor(
            synonyms={"fn": ["function"], "cfg": ["config"]},
            field_prefixes=["tag:", "path:"],
        )

    def test_full_pipeline(self, preprocessor):
        """Full pipeline extracts fields, expands camel and synonyms."""
        result = preprocessor.preprocess("tag:urgent findUserCfg")

        assert "tag" in result.fields
        assert result.fields["tag"] == "urgent"
        assert "function" not in result.text  # cfg should expand, not fn
        assert "config" in result.text
        assert "find" in result.text.lower()
        assert "user" in result.text.lower()

    def test_disable_camel_expansion(self, preprocessor):
        """Can disable camelCase expansion."""
        result = preprocessor.preprocess("getUserData", expand_camel=False)

        # Should not have "get user data" as separate words
        # No synonyms match in "getUserData", so just lowercased version
        assert "get" not in result.text.split() or "user" not in result.text.split()

    def test_disable_synonym_expansion(self, preprocessor):
        """Can disable synonym expansion."""
        result = preprocessor.preprocess("fn cfg", expand_syns=False)

        assert "function" not in result.text
        assert "config" not in result.text

    def test_disable_both_expansions(self, preprocessor):
        """Can disable all expansions."""
        result = preprocessor.preprocess(
            "getUserCfg",
            expand_camel=False,
            expand_syns=False,
        )

        assert result.text == "getUserCfg"
        assert result.expanded_terms == []

    def test_preserves_original(self, preprocessor):
        """Original query preserved in result."""
        original = "tag:test findConfig"
        result = preprocessor.preprocess(original)

        assert result.original == original


class TestDefaultSynonyms:
    """Tests for default synonyms."""

    def test_common_abbreviations(self):
        """Default synonyms include common abbreviations."""
        assert "fn" in DEFAULT_SYNONYMS
        assert "function" in DEFAULT_SYNONYMS["fn"]

        assert "cls" in DEFAULT_SYNONYMS
        assert "class" in DEFAULT_SYNONYMS["cls"]

        assert "cfg" in DEFAULT_SYNONYMS
        assert "config" in DEFAULT_SYNONYMS["cfg"]

    def test_programming_terms(self):
        """Default synonyms include programming terms."""
        assert "err" in DEFAULT_SYNONYMS
        assert "error" in DEFAULT_SYNONYMS["err"]

        assert "db" in DEFAULT_SYNONYMS
        assert "database" in DEFAULT_SYNONYMS["db"]


class TestCreateDefaultPreprocessor:
    """Tests for default preprocessor factory."""

    def test_creates_with_defaults(self):
        """Creates preprocessor with default settings."""
        preprocessor = create_default_preprocessor()

        assert preprocessor.synonyms is not None
        assert len(preprocessor.synonyms) > 0
        assert "path:" in preprocessor.field_prefixes

    def test_adds_extra_synonyms(self):
        """Adds extra synonyms to defaults."""
        preprocessor = create_default_preprocessor(
            extra_synonyms={"custom": ["synonym"]}
        )

        assert "custom" in preprocessor.synonyms
        assert "fn" in preprocessor.synonyms  # default still present

    def test_adds_extra_prefixes(self):
        """Adds extra prefixes to defaults."""
        preprocessor = create_default_preprocessor(
            extra_prefixes=["lang:"]
        )

        assert "lang:" in preprocessor.field_prefixes
        assert "path:" in preprocessor.field_prefixes  # default still present

    def test_extra_synonyms_override(self):
        """Extra synonyms can override defaults."""
        preprocessor = create_default_preprocessor(
            extra_synonyms={"fn": ["custom_function"]}
        )

        assert preprocessor.synonyms["fn"] == ["custom_function"]


class TestSynonymExpansionEdgeCases:
    """Edge case tests for synonym expansion."""

    def test_no_word_tokens_found(self):
        """expand_synonyms returns early when no tokens (line 86)."""
        preprocessor = QueryPreprocessor(synonyms={"fn": ["function"]})

        # Text with no word tokens (only punctuation, numbers, whitespace)
        text, terms = preprocessor.expand_synonyms("123 !@# $%^")

        # Should return unchanged since no word tokens found
        assert text == "123 !@# $%^"
        assert terms == []

    def test_empty_string(self):
        """expand_synonyms with empty string returns early (line 86)."""
        preprocessor = QueryPreprocessor(synonyms={"fn": ["function"]})

        text, terms = preprocessor.expand_synonyms("")

        assert text == ""
        assert terms == []

    def test_only_numbers(self):
        """expand_synonyms with only numbers returns early (line 86)."""
        preprocessor = QueryPreprocessor(synonyms={"fn": ["function"]})

        # Pure digits don't match the word pattern [a-zA-Z_][a-zA-Z0-9_]*
        text, terms = preprocessor.expand_synonyms("12345 67890")

        assert text == "12345 67890"
        assert terms == []


class TestNegatedFieldPrefix:
    """A prefix that starts with '-' (e.g. '-path:') must be extractable, and
    must not be mis-parsed as its positive counterpart with a stray '-'."""

    def test_negated_prefix_extracted_not_as_inclusion(self):
        preprocessor = QueryPreprocessor(field_prefixes=["path:", "-path:"])
        remaining, fields = preprocessor.extract_fields("-path:tests find error")
        assert fields.get("-path") == "tests"
        assert "path" not in fields  # not mis-captured as an inclusion
        assert "-" not in remaining  # no stray dash left behind
        assert remaining == "find error"

    def test_positive_prefix_still_extracted(self):
        preprocessor = QueryPreprocessor(field_prefixes=["path:", "-path:"])
        remaining, fields = preprocessor.extract_fields("path:src find error")
        assert fields.get("path") == "src"
        assert "-path" not in fields
        assert remaining == "find error"
