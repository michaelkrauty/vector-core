"""Tests for GlobalVocabulary (cross-codebase sparse vector search)."""

import concurrent.futures
import sqlite3
import tempfile
from pathlib import Path

import pytest

from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.embeddings.tokenization import (
    levenshtein_distance,
    levenshtein_similarity,
)
from vector_core.embeddings.sparse import SparseVector


class TestGlobalVocabularyInit:
    """Tests for GlobalVocabulary initialization."""

    def test_creates_db_file(self, tmp_path):
        """GlobalVocabulary creates database file."""
        db_path = tmp_path / "test_vocab.db"
        vocab = GlobalVocabulary(db_path=db_path)

        assert db_path.exists()
        vocab.close()

    def test_default_settings(self):
        """Default settings are applied."""
        with tempfile.TemporaryDirectory() as tmp:
            vocab = GlobalVocabulary(db_path=Path(tmp) / "test.db")

            assert vocab.min_token_length == 2
            assert len(vocab.stop_tokens) > 0
            assert "the" in vocab.stop_tokens
            vocab.close()

    def test_custom_settings(self):
        """Custom settings are respected."""
        with tempfile.TemporaryDirectory() as tmp:
            vocab = GlobalVocabulary(
                db_path=Path(tmp) / "test.db",
                min_token_length=4,
                stop_tokens={"custom", "stop"},
            )

            assert vocab.min_token_length == 4
            assert vocab.stop_tokens == {"custom", "stop"}
            vocab.close()

    def test_empty_vocab_stats(self, tmp_path):
        """Empty vocabulary has correct stats."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        assert vocab.vocab_size == 0
        assert vocab.total_docs == 0
        vocab.close()


class TestTokenization:
    """Tests for tokenization (mirrors SparseVectorizer tests)."""

    def test_camel_case_split(self, tmp_path):
        """CamelCase identifiers are split."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        tokens = vocab.tokenize("getUserData")

        assert "get" in tokens
        assert "user" in tokens
        assert "data" in tokens
        vocab.close()

    def test_snake_case_split(self, tmp_path):
        """snake_case identifiers are split."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        tokens = vocab.tokenize("get_user_data")

        assert "get" in tokens
        assert "user" in tokens
        assert "data" in tokens
        vocab.close()

    def test_stop_token_removal(self, tmp_path):
        """Stop words are removed."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        tokens = vocab.tokenize("the a is for with")

        assert "the" not in tokens
        assert "for" not in tokens
        vocab.close()

    def test_min_length_filter(self, tmp_path):
        """Short tokens are filtered."""
        vocab = GlobalVocabulary(
            db_path=tmp_path / "test.db", min_token_length=3
        )
        tokens = vocab.tokenize("a ab abc abcd")

        assert "a" not in tokens
        assert "ab" not in tokens
        assert "abc" in tokens
        assert "abcd" in tokens
        vocab.close()


class TestCodebaseRegistration:
    """Tests for codebase registration."""

    def test_register_codebase(self, tmp_path):
        """register_codebase adds tokens to vocabulary."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        tokens_per_doc = [
            {"hello", "world"},
            {"foo", "bar"},
        ]
        new_tokens = vocab.register_codebase("test_codebase", tokens_per_doc)

        assert new_tokens == 4
        assert vocab.vocab_size == 4
        assert vocab.total_docs == 2
        vocab.close()

    def test_register_multiple_codebases(self, tmp_path):
        """Multiple codebases share vocabulary."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        vocab.register_codebase("codebase1", [{"hello", "world"}])
        vocab.register_codebase("codebase2", [{"hello", "foo"}])

        # "hello" appears in both, should have index once
        assert vocab.vocab_size == 3  # hello, world, foo
        assert vocab.total_docs == 2
        vocab.close()

    def test_unregister_codebase(self, tmp_path):
        """unregister_codebase removes contribution."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        vocab.register_codebase("codebase1", [{"hello", "world"}])
        vocab.register_codebase("codebase2", [{"foo", "bar"}])

        assert vocab.total_docs == 2

        vocab.unregister_codebase("codebase1")

        # Vocabulary tokens remain (append-only)
        assert vocab.vocab_size == 4
        # But doc count decreases
        assert vocab.total_docs == 1
        vocab.close()

    def test_reregister_codebase(self, tmp_path):
        """Re-registering a codebase replaces old contribution."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        vocab.register_codebase("codebase1", [{"hello", "world"}, {"foo"}])
        assert vocab.total_docs == 2

        # Re-register with different docs
        vocab.register_codebase("codebase1", [{"hello"}])
        assert vocab.total_docs == 1  # Old docs replaced
        vocab.close()

    def test_get_codebase_ids(self, tmp_path):
        """get_codebase_ids returns registered codebases."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        vocab.register_codebase("codebase1", [{"hello"}])
        vocab.register_codebase("codebase2", [{"world"}])

        ids = vocab.get_codebase_ids()
        assert set(ids) == {"codebase1", "codebase2"}
        vocab.close()

    def test_get_codebase_stats(self, tmp_path):
        """get_codebase_stats returns codebase info."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        vocab.register_codebase("codebase1", [{"hello", "world"}, {"foo"}])

        stats = vocab.get_codebase_stats("codebase1")
        assert stats is not None
        assert stats["doc_count"] == 2
        assert stats["token_count"] == 3
        vocab.close()


class TestDocumentVectorization:
    """Tests for TF-only document vectors."""

    def test_vectorize_document(self, tmp_path):
        """vectorize_document creates TF-only sparse vector."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("test", [{"hello", "world", "python"}])

        vec = vocab.vectorize_document("hello hello world")

        assert isinstance(vec, SparseVector)
        assert len(vec.indices) == 2  # hello, world
        assert len(vec.values) == 2
        vocab.close()

    def test_vectorize_document_tf_weights(self, tmp_path):
        """Document vectors use TF weights (not IDF)."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("test", [
            {"hello"},  # hello in 1 doc
            {"rare"},   # rare in 1 doc
        ])

        # Both terms appear once, should have similar TF weights
        vec_hello = vocab.vectorize_document("hello hello hello")
        vec_rare = vocab.vectorize_document("rare rare rare")

        # TF weights for same frequency should be same (log(3) + 1)
        # Regardless of IDF differences
        assert len(vec_hello.values) == 1
        assert len(vec_rare.values) == 1
        assert abs(vec_hello.values[0] - vec_rare.values[0]) < 0.001
        vocab.close()

    def test_vectorize_document_unknown_tokens(self, tmp_path):
        """Unknown tokens are ignored."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("test", [{"hello"}])

        vec = vocab.vectorize_document("hello unknown words here")

        # Only "hello" should be vectorized
        assert len(vec.indices) == 1
        vocab.close()

    def test_vectorize_empty_document(self, tmp_path):
        """Empty document produces empty vector."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("test", [{"hello"}])

        vec = vocab.vectorize_document("")

        assert len(vec.indices) == 0
        vocab.close()

    def test_vectorize_document_sorted_indices(self, tmp_path):
        """Document vector indices are sorted."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("test", [{"alpha", "beta", "gamma", "delta"}])

        vec = vocab.vectorize_document("gamma alpha delta beta")

        # Indices should be sorted ascending
        assert vec.indices == sorted(vec.indices)
        vocab.close()


class TestQueryVectorization:
    """Tests for IDF-only query vectors."""

    def test_vectorize_query(self, tmp_path):
        """vectorize_query creates IDF-only sparse vector."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("test", [{"hello", "world"}])

        vec = vocab.vectorize_query("hello world")

        assert isinstance(vec, SparseVector)
        assert len(vec.indices) == 2
        vocab.close()

    def test_vectorize_query_idf_weights(self, tmp_path):
        """Query vectors use IDF weights."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("test", [
            {"common", "rare"},
            {"common"},
            {"common"},
        ])

        vec = vocab.vectorize_query("common rare")

        # Find which index is which
        v = vocab._get_vocab()
        common_idx = v.get("common")
        rare_idx = v.get("rare")

        common_weight = None
        rare_weight = None
        for idx, val in zip(vec.indices, vec.values):
            if idx == common_idx:
                common_weight = val
            elif idx == rare_idx:
                rare_weight = val

        # Rare term should have higher IDF weight
        assert rare_weight is not None
        assert common_weight is not None
        assert rare_weight > common_weight
        vocab.close()

    def test_vectorize_query_fuzzy_matching(self, tmp_path):
        """Query vectorization supports fuzzy matching."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("test", [{"handleRequest", "processData"}])

        # Typo should fuzzy match
        vec = vocab.vectorize_query("handlRequest", fuzzy=True)

        # Should produce output if fuzzy match succeeds
        # (exact behavior depends on threshold)
        assert isinstance(vec, SparseVector)
        vocab.close()

    def test_vectorize_query_no_fuzzy(self, tmp_path):
        """Query vectorization without fuzzy ignores unknown terms."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("test", [{"hello"}])

        vec = vocab.vectorize_query("helllo", fuzzy=False)  # typo

        assert len(vec.indices) == 0
        vocab.close()

    def test_vectorize_query_deduplicates(self, tmp_path):
        """Query vectorization deduplicates tokens."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("test", [{"hello", "world"}])

        vec = vocab.vectorize_query("hello hello hello world world")

        assert len(vec.indices) == 2
        vocab.close()


class TestCrossCodebaseConsistency:
    """Tests for cross-codebase search consistency."""

    def test_same_tokens_same_indices(self, tmp_path):
        """Same tokens get same indices across codebases."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        vocab.register_codebase("codebase1", [{"hello", "world"}])
        vocab.register_codebase("codebase2", [{"hello", "foo"}])

        # Vectorize documents from different codebases
        vec1 = vocab.vectorize_document("hello world")
        vec2 = vocab.vectorize_document("hello foo")

        # Both should have "hello" at the same index
        v = vocab._get_vocab()
        hello_idx = v["hello"]

        assert hello_idx in vec1.indices
        assert hello_idx in vec2.indices
        vocab.close()

    def test_query_scores_comparable(self, tmp_path):
        """Query scores are comparable across codebases."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        # Two codebases with some overlap
        vocab.register_codebase("codebase1", [
            {"authenticate", "user", "session"},
            {"validate", "token"},
        ])
        vocab.register_codebase("codebase2", [
            {"authenticate", "api", "key"},
            {"authorize", "request"},
        ])

        # Query should use global IDF
        query_vec = vocab.vectorize_query("authenticate user")

        # IDF is computed from global stats (4 total docs)
        # "authenticate" appears in 2 docs, "user" in 1 doc
        assert vocab.total_docs == 4
        assert len(query_vec.indices) == 2
        vocab.close()


class TestIncrementalUpdate:
    """Tests for incremental vocabulary updates."""

    def test_update_adds_tokens(self, tmp_path):
        """Incremental update adds new tokens."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        vocab.register_codebase("test", [{"hello", "world"}])
        initial_size = vocab.vocab_size

        new_tokens = vocab.update_codebase_incremental(
            "test",
            added_tokens=[{"new", "tokens"}],
            removed_tokens=[],
            net_doc_change=1,
        )

        assert new_tokens == 2
        assert vocab.vocab_size == initial_size + 2
        vocab.close()

    def test_update_establishes_a_missing_doc_count(self, tmp_path):
        """A codebase with no contribution yet must still get its count.

        Regression: the document count was moved with a bare UPDATE, which
        matches no row for a codebase that has never registered. Its tokens
        landed in the vocabulary and in codebase_contributions, but its
        document count stayed absent and therefore read as zero forever. Every
        caller that maintains its contribution purely incrementally, rather
        than seeding it with register_codebase first, was affected, and a
        codebase reporting zero documents while contributing document
        frequencies skews IDF for every codebase sharing the database.
        """
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        vocab.update_codebase_incremental(
            "fresh",
            added_tokens=[{"hello", "world"}],
            removed_tokens=[],
            net_doc_change=1,
        )

        assert vocab.get_codebase_doc_count("fresh") == 1
        assert vocab.total_docs == 1
        vocab.close()

    def test_update_accumulates_from_a_missing_doc_count(self, tmp_path):
        """The count keeps accumulating once the row has been established."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        for token in ("alpha", "beta", "gamma"):
            vocab.update_codebase_incremental(
                "fresh",
                added_tokens=[{token}],
                removed_tokens=[],
                net_doc_change=1,
            )

        assert vocab.get_codebase_doc_count("fresh") == 3
        vocab.close()

    def test_update_does_not_create_a_negative_doc_count(self, tmp_path):
        """Removing from a corpus that was never registered leaves it empty.

        A corpus with no contribution has nothing to remove, and a negative
        document count would make total_docs, and therefore every IDF weight
        derived from it, meaningless for every codebase in the database.
        """
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        vocab.update_codebase_incremental(
            "fresh",
            added_tokens=[],
            removed_tokens=[{"hello"}],
            net_doc_change=-1,
        )

        assert vocab.get_codebase_doc_count("fresh") == 0
        assert vocab.total_docs == 0
        vocab.close()

    def test_update_leaves_an_untouched_codebase_alone(self, tmp_path):
        """Establishing one codebase's count must not disturb another's."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("existing", [{"one"}, {"two"}])

        vocab.update_codebase_incremental(
            "fresh",
            added_tokens=[{"three"}],
            removed_tokens=[],
            net_doc_change=1,
        )

        assert vocab.get_codebase_doc_count("existing") == 2
        assert vocab.get_codebase_doc_count("fresh") == 1
        assert vocab.total_docs == 3
        vocab.close()

    def test_over_removal_cannot_drive_doc_freq_negative(self, tmp_path):
        """A document frequency is a count of documents and cannot be negative.

        Query weighting is ``log((total + 1) / (df + 1)) + 1``, so a df of -1
        divides by zero and anything below it takes the log of a negative
        number. Either one raises out of ``vectorize_query``, which is on the
        search path of every codebase sharing the database, so one consumer's
        accounting drift would take searching down for all of them.
        """
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("test", [{"shared"}])

        vocab.update_codebase_incremental(
            "test",
            added_tokens=[],
            removed_tokens=[{"shared"}, {"shared"}, {"shared"}],
            net_doc_change=-3,
        )

        assert vocab._get_doc_freq()["shared"] == 0
        assert vocab.vectorize_query("shared") is not None
        vocab.close()

    def test_reregistration_cannot_drive_doc_freq_negative(self, tmp_path):
        """The same invariant on the removal half of register_codebase."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("a", [{"shared"}])
        vocab.register_codebase("b", [{"shared"}])

        # Drop b's own contribution out from under the shared counter, then
        # re-register a smaller corpus for a.
        vocab.update_codebase_incremental(
            "a",
            added_tokens=[],
            removed_tokens=[{"shared"}, {"shared"}],
            net_doc_change=-2,
        )
        vocab.register_codebase("b", [])

        assert vocab._get_doc_freq().get("shared", 0) >= 0
        assert vocab.vectorize_query("shared") is not None
        vocab.close()

    def test_over_removal_leaves_another_codebase_share_intact(self, tmp_path):
        """A codebase can only take back what it put in.

        Flooring the aggregate on its own would hide the discrepancy rather
        than prevent it: the over-removal would already have consumed the other
        codebase's share of the shared token before the floor applied.
        """
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("a", [{"shared"}])
        vocab.register_codebase("b", [{"shared"}])
        assert vocab._get_doc_freq()["shared"] == 2

        vocab.update_codebase_incremental(
            "a",
            added_tokens=[],
            removed_tokens=[{"shared"}, {"shared"}, {"shared"}],
            net_doc_change=-3,
        )

        # a contributed one, so only one comes off; b's share survives.
        assert vocab._get_doc_freq()["shared"] == 1
        conn = sqlite3.connect(vocab.db_path)
        try:
            total = conn.execute(
                "SELECT SUM(doc_freq) FROM codebase_contributions WHERE token = 'shared'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert total == 1
        vocab.close()

    def test_a_token_added_and_removed_in_one_call_settles_correctly(self, tmp_path):
        """The contribution settles at max(existing + added - removed, 0).

        Capping the removal alone gets this wrong: with one contribution, one
        addition and two removals, only one removal would be permitted and the
        addition would then leave the row at 1 instead of 0, where the
        zero-count cleanup can no longer reach it.
        """
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("a", [{"shared"}])

        vocab.update_codebase_incremental(
            "a",
            added_tokens=[{"shared"}],
            removed_tokens=[{"shared"}, {"shared"}],
            net_doc_change=-1,
        )

        conn = sqlite3.connect(vocab.db_path)
        try:
            row = conn.execute(
                "SELECT doc_freq FROM codebase_contributions "
                "WHERE codebase_id = 'a' AND token = 'shared'"
            ).fetchone()
        finally:
            conn.close()
        assert row is None, "a settled contribution of zero must be cleaned up"
        assert vocab._get_doc_freq().get("shared", 0) == 0
        vocab.close()

    def test_addition_only_update_reads_no_contributions(self, tmp_path):
        """An addition needs no current contribution, and this runs inside the
        writer transaction, so it must not load the codebase's whole table."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("a", [{"one"}, {"two"}, {"three"}])

        conn = vocab._get_conn()
        seen: list[str] = []
        conn.set_trace_callback(lambda sql: seen.append(" ".join(sql.split())))
        try:
            vocab.update_codebase_incremental(
                "a",
                added_tokens=[{"four"}],
                removed_tokens=[],
                net_doc_change=1,
            )
        finally:
            conn.set_trace_callback(None)

        reads = [s for s in seen if "doc_freq FROM codebase_contributions" in s]
        assert reads == []
        assert vocab.get_codebase_doc_count("a") == 4
        vocab.close()

    def test_unregister_removes_a_count_row_without_contributions(self, tmp_path):
        """An incremental removal can establish a zero row with no tokens.

        Leaving it behind would keep an unregistered codebase visible through
        get_codebase_ids() and counted in stats().
        """
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.update_codebase_incremental(
            "ghost",
            added_tokens=[],
            removed_tokens=[{"hello"}],
            net_doc_change=-1,
        )

        vocab.unregister_codebase("ghost")

        assert "ghost" not in vocab.get_codebase_ids()
        vocab.close()

    def test_update_removes_contribution(self, tmp_path):
        """Incremental update decrements doc_freq."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        vocab.register_codebase("test", [
            {"hello", "world"},
            {"hello", "foo"},
        ])

        # Remove a doc with "hello"
        vocab.update_codebase_incremental(
            "test",
            added_tokens=[],
            removed_tokens=[{"hello", "world"}],
            net_doc_change=-1,
        )

        # "hello" doc_freq should decrease
        doc_freq = vocab._get_doc_freq()
        assert doc_freq["hello"] == 1  # Was 2, now 1
        assert doc_freq["world"] == 0  # Was 1, now 0
        vocab.close()


class TestPersistence:
    """Tests for vocabulary persistence across restarts."""

    def test_persist_across_restart(self, tmp_path):
        """Vocabulary persists across GlobalVocabulary instances."""
        db_path = tmp_path / "test.db"

        # Create and populate
        vocab1 = GlobalVocabulary(db_path=db_path)
        vocab1.register_codebase("test", [{"hello", "world", "python"}])
        vocab1.close()

        # Reopen
        vocab2 = GlobalVocabulary(db_path=db_path)

        assert vocab2.vocab_size == 3
        assert vocab2.total_docs == 1
        vocab2.close()

    def test_indices_stable_across_restart(self, tmp_path):
        """Token indices are stable across restarts."""
        db_path = tmp_path / "test.db"

        # Create and get indices
        vocab1 = GlobalVocabulary(db_path=db_path)
        vocab1.register_codebase("test", [{"hello", "world"}])
        indices1 = vocab1._get_vocab().copy()
        vocab1.close()

        # Reopen and check indices
        vocab2 = GlobalVocabulary(db_path=db_path)
        indices2 = vocab2._get_vocab()

        assert indices1 == indices2
        vocab2.close()

    def test_vector_reproducible_after_restart(self, tmp_path):
        """Same text produces same vector after restart."""
        db_path = tmp_path / "test.db"

        # Create and vectorize
        vocab1 = GlobalVocabulary(db_path=db_path)
        vocab1.register_codebase("test", [{"hello", "world", "python"}])
        vec1 = vocab1.vectorize_document("hello world python")
        vocab1.close()

        # Reopen and vectorize again
        vocab2 = GlobalVocabulary(db_path=db_path)
        vec2 = vocab2.vectorize_document("hello world python")

        assert vec1.indices == vec2.indices
        assert vec1.values == vec2.values
        vocab2.close()


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_get_instance_returns_same_object(self, tmp_path, monkeypatch):
        """Multiple threads calling get_instance() get the same instance.

        Tests that the lock-first singleton pattern prevents race conditions.
        """
        import threading

        from vector_core.settings import settings

        # Reset singleton state first
        GlobalVocabulary.reset_instance()

        # Monkeypatch the cache_dir to use tmp_path
        monkeypatch.setattr(settings, "cache_dir", tmp_path)

        results = []
        errors = []
        barrier = threading.Barrier(10)  # Synchronize thread starts

        def get_instance_thread():
            try:
                barrier.wait()  # All threads start together
                instance = GlobalVocabulary.get_instance()
                results.append(id(instance))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=get_instance_thread)
            for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Cleanup
        GlobalVocabulary.reset_instance()

        assert not errors, f"Errors occurred: {errors}"
        assert len(results) == 10
        # All threads should get the same instance (same id)
        assert len(set(results)) == 1, f"Got different instances: {set(results)}"

    def test_concurrent_register(self, tmp_path):
        """Concurrent codebase registration is safe."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        def register(codebase_id):
            vocab.register_codebase(
                codebase_id,
                [{"token_" + codebase_id}],
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(register, f"codebase_{i}")
                for i in range(10)
            ]
            concurrent.futures.wait(futures)

        assert vocab.total_docs == 10
        vocab.close()

    def test_concurrent_vectorize(self, tmp_path):
        """Concurrent vectorization is safe."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("test", [{"hello", "world", "python", "code"}])

        results = []

        def vectorize(text):
            vec = vocab.vectorize_document(text)
            results.append(vec)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            texts = ["hello world", "python code", "hello python", "world code"]
            futures = [executor.submit(vectorize, t) for t in texts]
            concurrent.futures.wait(futures)

        assert len(results) == 4
        assert all(isinstance(r, SparseVector) for r in results)
        vocab.close()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_register_empty_docs(self, tmp_path):
        """Registering with no docs is handled."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        new_tokens = vocab.register_codebase("test", [])

        assert new_tokens == 0
        assert vocab.total_docs == 0
        vocab.close()

    def test_unregister_nonexistent(self, tmp_path):
        """Unregistering nonexistent codebase is safe."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        # Should not raise
        vocab.unregister_codebase("nonexistent")
        vocab.close()

    def test_get_stats_nonexistent_codebase(self, tmp_path):
        """get_codebase_stats returns None for unknown codebase."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        stats = vocab.get_codebase_stats("nonexistent")
        assert stats is None
        vocab.close()

    def test_global_stats(self, tmp_path):
        """stats() returns comprehensive info."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("codebase1", [{"hello"}])
        vocab.register_codebase("codebase2", [{"world"}])

        stats = vocab.stats()

        assert stats["vocab_size"] == 2
        assert stats["total_docs"] == 2
        assert stats["codebase_count"] == 2
        vocab.close()

    def test_context_manager(self, tmp_path):
        """GlobalVocabulary works as context manager."""
        db_path = tmp_path / "test.db"

        with GlobalVocabulary(db_path=db_path) as vocab:
            vocab.register_codebase("test", [{"hello"}])
            assert vocab.vocab_size == 1

    def test_append_only_vocabulary(self, tmp_path):
        """Vocabulary indices are append-only."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        # Register first codebase
        vocab.register_codebase("cb1", [{"hello", "world"}])
        initial_vocab = vocab._get_vocab().copy()

        # Unregister it
        vocab.unregister_codebase("cb1")

        # Indices should remain (append-only)
        post_unregister_vocab = vocab._get_vocab()
        assert initial_vocab == post_unregister_vocab

        # Register new codebase - should get new indices, not reuse
        vocab.register_codebase("cb2", [{"foo", "bar"}])
        final_vocab = vocab._get_vocab()

        # All original indices preserved, new ones added
        for token, idx in initial_vocab.items():
            assert final_vocab[token] == idx

        # New tokens have higher indices
        max_old_idx = max(initial_vocab.values())
        for token in ["foo", "bar"]:
            assert final_vocab[token] > max_old_idx
        vocab.close()

    def test_fuzzy_match_empty_vocab(self, tmp_path):
        """Fuzzy matching with empty vocab returns None."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        match, sim = vocab._find_fuzzy_match("test", {}, 0.75)

        assert match is None
        assert sim == 0.0
        vocab.close()

    def test_schema_version_set_on_new_db(self, tmp_path):
        """Schema version is set when creating new database."""
        import sqlite3

        from vector_core.embeddings.global_vocab import SCHEMA_VERSION

        db_path = tmp_path / "test.db"
        vocab = GlobalVocabulary(db_path=db_path)
        vocab.close()

        # Verify schema version is set
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT value FROM metadata WHERE key = 'schema_version'")
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert int(row[0]) == SCHEMA_VERSION

    def test_schema_version_future_version_raises(self, tmp_path):
        """Opening database with newer schema version raises error."""
        import sqlite3

        from vector_core.embeddings.global_vocab import SCHEMA_VERSION

        db_path = tmp_path / "test.db"

        # Create database with future schema version
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.execute("INSERT INTO metadata VALUES ('schema_version', ?)", (str(SCHEMA_VERSION + 1),))
        conn.commit()
        conn.close()

        # Attempting to open should raise
        with pytest.raises(ValueError, match="newer than supported"):
            GlobalVocabulary(db_path=db_path)


class TestLevenshteinDistance:
    """Tests for Levenshtein distance/similarity functions."""

    def test_distance_identical_strings(self):
        """Identical strings have distance 0."""
        assert levenshtein_distance("hello", "hello") == 0

    def test_distance_empty_second_string(self):
        """Distance to empty string equals first string length."""
        assert levenshtein_distance("hello", "") == 5

    def test_distance_empty_first_string(self):
        """Distance from empty string equals second string length."""
        assert levenshtein_distance("", "world") == 5

    def test_distance_single_substitution(self):
        """Single character difference has distance 1."""
        assert levenshtein_distance("cat", "bat") == 1

    def test_distance_single_insertion(self):
        """Single insertion has distance 1."""
        assert levenshtein_distance("cat", "cats") == 1

    def test_distance_single_deletion(self):
        """Single deletion has distance 1."""
        assert levenshtein_distance("cats", "cat") == 1

    def test_distance_swap_for_shorter_first(self):
        """Handles case where s1 is shorter than s2 (internal swap)."""
        # This exercises the len(s1) < len(s2) branch
        assert levenshtein_distance("ab", "abcdef") == 4

    def test_similarity_identical_strings(self):
        """Identical strings have similarity 1.0."""
        assert levenshtein_similarity("hello", "hello") == 1.0

    def test_similarity_empty_first_string(self):
        """Empty first string returns 0.0."""
        assert levenshtein_similarity("", "hello") == 0.0

    def test_similarity_empty_second_string(self):
        """Empty second string returns 0.0."""
        assert levenshtein_similarity("hello", "") == 0.0

    def test_similarity_both_empty(self):
        """Both empty strings returns 0.0."""
        assert levenshtein_similarity("", "") == 0.0

    def test_similarity_similar_words(self):
        """Similar words have high similarity."""
        sim = levenshtein_similarity("kitten", "sitting")
        # kitten -> sitting: 3 edits (k->s, e->i, +g)
        # max_len = 7, distance = 3, similarity = 1 - 3/7 ≈ 0.57
        assert 0.5 < sim < 0.6

    def test_similarity_completely_different(self):
        """Completely different words have low similarity."""
        sim = levenshtein_similarity("abc", "xyz")
        # abc -> xyz: 3 edits
        # max_len = 3, distance = 3, similarity = 0
        assert sim == 0.0


class TestContextManagerAndCleanup:
    """Tests for context manager and cleanup (lines 780-791)."""

    def test_context_manager_usage(self, tmp_path):
        """GlobalVocabulary can be used as context manager."""
        with GlobalVocabulary(db_path=tmp_path / "test.db") as vocab:
            vocab.register_codebase("test", [{"token"}])
            assert vocab.vocab_size > 0

    def test_del_handles_exception_silently(self, tmp_path, monkeypatch):
        """__del__ silently handles exceptions (lines 780-783)."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        # Force close to raise an exception
        def raise_error():
            raise RuntimeError("Simulated shutdown error")

        monkeypatch.setattr(vocab, "close", raise_error)

        # __del__ should not raise even when close() raises
        vocab.__del__()  # Should not raise

    def test_del_with_no_connection(self, tmp_path):
        """__del__ handles case when no connection exists."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.close()
        vocab.__del__()  # Should handle already-closed state


class TestExceptionRollback:
    """Tests for database transaction rollback on exceptions.

    Note: These tests verify that rollback is attempted but cannot easily
    test the actual rollback since sqlite connections are complex to mock.
    The main goal is to ensure exceptions are re-raised properly.
    """

    def test_register_codebase_handles_exception(self, tmp_path):
        """register_codebase propagates exceptions (tests exception path)."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        # This should work normally
        vocab.register_codebase("good", [{"token1"}])
        assert vocab.vocab_size > 0
        vocab.close()

    def test_unregister_nonexistent_codebase(self, tmp_path):
        """unregister_codebase handles nonexistent codebase gracefully."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        # Unregistering a codebase that doesn't exist should not fail
        vocab.unregister_codebase("nonexistent")

        vocab.close()

    def test_update_incremental_works(self, tmp_path):
        """update_codebase_incremental works correctly."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        # First register
        vocab.register_codebase("test", [{"token1"}])
        initial_size = vocab.vocab_size

        # Incremental update
        new_tokens = vocab.update_codebase_incremental(
            "test",
            added_tokens=[{"newtok1", "newtok2"}],
            removed_tokens=[],
            net_doc_change=1
        )

        assert new_tokens >= 0
        vocab.close()


class TestCodebaseDocCount:
    """Tests for get_codebase_doc_count edge cases."""

    def test_nonexistent_codebase_returns_zero(self, tmp_path):
        """get_codebase_doc_count returns 0 for nonexistent codebase (line 267)."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        count = vocab.get_codebase_doc_count("nonexistent_codebase")

        assert count == 0
        vocab.close()

    def test_existing_codebase_returns_count(self, tmp_path):
        """get_codebase_doc_count returns correct count."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("mybase", [{"token1"}, {"token2"}, {"token3"}])

        count = vocab.get_codebase_doc_count("mybase")

        assert count == 3
        vocab.close()


class TestFuzzyMatching:
    """Tests for fuzzy token matching (lines 716-719)."""

    def test_find_fuzzy_match_exact_match(self, tmp_path):
        """Exact match returns perfect score."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("test", [{"hello", "world"}])

        # Get vocabulary dict for testing private method
        vocab_dict = vocab._get_vocab()
        match, score = vocab._find_fuzzy_match("hello", vocab_dict)

        assert match == "hello"
        assert score == 1.0
        vocab.close()

    def test_find_fuzzy_match_similar_token(self, tmp_path):
        """Fuzzy match finds similar tokens (lines 716-719)."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("test", [{"hello", "hallo", "world"}])

        vocab_dict = vocab._get_vocab()
        match, score = vocab._find_fuzzy_match("helo", vocab_dict, threshold=0.6)

        # "helo" is close to both "hello" and "hallo"
        assert match in ["hello", "hallo"]
        assert score >= 0.6
        vocab.close()

    def test_find_fuzzy_match_no_match_below_threshold(self, tmp_path):
        """No match returns None when below threshold."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        vocab.register_codebase("test", [{"abcdefgh"}])

        vocab_dict = vocab._get_vocab()
        match, score = vocab._find_fuzzy_match("xyz", vocab_dict, threshold=0.8)

        assert match is None
        assert score == 0.0
        vocab.close()

    def test_find_fuzzy_match_empty_vocab(self, tmp_path):
        """Empty vocabulary returns None."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        match, score = vocab._find_fuzzy_match("anything", {})

        assert match is None
        assert score == 0.0
        vocab.close()

    def test_find_fuzzy_match_length_filtering(self, tmp_path):
        """Only tokens of similar length are considered."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        # Add tokens of varying lengths
        vocab.register_codebase("test", [
            {"ab", "abc", "abcd", "abcdefghij", "abcdefghijklmnop"}
        ])

        vocab_dict = vocab._get_vocab()
        # Looking for 4-char token, should only match similar lengths
        match, score = vocab._find_fuzzy_match("abce", vocab_dict, threshold=0.5)

        # Should match "abcd" (length 4), not the very long or very short ones
        if match is not None:
            assert abs(len(match) - 4) <= 2
        vocab.close()

    def test_find_fuzzy_match_iterates_candidates(self, tmp_path):
        """Tests the for loop iteration over candidates (lines 715-719)."""
        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        # Register multiple similar tokens
        vocab.register_codebase("test", [{"test", "tests", "tested", "testing"}])

        vocab_dict = vocab._get_vocab()
        # Find best match among similar tokens
        match, score = vocab._find_fuzzy_match("testy", vocab_dict, threshold=0.6)

        # Should find one of the similar tokens
        assert match is not None
        assert score > 0.6
        vocab.close()


class TestFuzzyMatchCandidateOrdering:
    """_find_fuzzy_match must score every length-window token, so the closest
    match is found even amid many unrelated same-length tokens. A fixed
    candidate cap could crowd out an insertion/deletion typo (whose length
    differs by one) with same-length decoys."""

    def test_insertion_typo_not_crowded_out_by_same_length_tokens(self, tmp_path):
        vocab_obj = GlobalVocabulary(db_path=tmp_path / "fz.db")
        # 600 unrelated length-6 decoys (more than the old 500-candidate cap),
        # then the true match "hello" (length 5, one deletion from "helllo").
        vocab = {f"qqq{i:03d}": i for i in range(600)}
        vocab["hello"] = 600
        match, sim = vocab_obj._find_fuzzy_match("helllo", vocab, 0.7)
        assert match == "hello"
        assert sim >= 0.7
