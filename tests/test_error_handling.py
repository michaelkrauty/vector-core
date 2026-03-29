"""Comprehensive tests for error handling across vector-core.

Tests exception classes, specific error conditions, and recovery scenarios.
"""

import sqlite3
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from vector_core.errors import (
    CollectionError,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseIntegrityError,
    DatabaseLockError,
    ErrorCategory,
    ErrorCollector,
    ErrorSeverity,
    PointOperationError,
    StorageError,
    error_response,
    ErrorCode,
    is_error_response,
)


# =============================================================================
# Exception Class Tests
# =============================================================================


class TestDatabaseExceptions:
    """Tests for database exception hierarchy."""

    def test_database_error_base(self):
        """DatabaseError is base for database exceptions."""
        err = DatabaseError("generic db error")
        assert str(err) == "generic db error"
        assert isinstance(err, Exception)

    def test_database_connection_error(self):
        """DatabaseConnectionError captures connection failure details."""
        original = OSError("disk full")
        err = DatabaseConnectionError(
            db_path="/tmp/test.db",
            operation="write",
            original=original,
        )

        assert err.db_path == "/tmp/test.db"
        assert err.operation == "write"
        assert err.original is original
        assert "connection failed during write" in str(err).lower()
        assert "/tmp/test.db" in str(err)
        assert isinstance(err, DatabaseError)

    def test_database_connection_error_defaults(self):
        """DatabaseConnectionError has sensible defaults."""
        err = DatabaseConnectionError("/path/to/db.sqlite")

        assert err.db_path == "/path/to/db.sqlite"
        assert err.operation == "connect"
        assert err.original is None

    def test_database_integrity_error(self):
        """DatabaseIntegrityError captures constraint violations."""
        err = DatabaseIntegrityError(
            "Duplicate key 'foo'",
            constraint="unique_term",
        )

        assert "Duplicate key 'foo'" in str(err)
        assert err.constraint == "unique_term"
        assert isinstance(err, DatabaseError)

    def test_database_integrity_error_no_constraint(self):
        """DatabaseIntegrityError works without constraint name."""
        err = DatabaseIntegrityError("Foreign key violation")

        assert err.constraint is None
        assert "Foreign key" in str(err)

    def test_database_lock_error(self):
        """DatabaseLockError captures lock timeout details."""
        err = DatabaseLockError("/tmp/locked.db", timeout=5.0)

        assert err.db_path == "/tmp/locked.db"
        assert err.timeout == 5.0
        assert "locked" in str(err).lower()
        assert "5.0" in str(err)
        assert isinstance(err, DatabaseError)


class TestStorageExceptions:
    """Tests for vector storage exception hierarchy."""

    def test_storage_error_base(self):
        """StorageError is base for storage exceptions."""
        err = StorageError("generic storage error")
        assert str(err) == "generic storage error"
        assert isinstance(err, Exception)

    def test_collection_error(self):
        """CollectionError captures collection operation failures."""
        err = CollectionError(
            collection="my_collection",
            operation="create",
        )

        assert err.collection == "my_collection"
        assert err.operation == "create"
        assert "my_collection" in str(err)
        assert "create" in str(err).lower()
        assert isinstance(err, StorageError)

    def test_collection_error_custom_message(self):
        """CollectionError accepts custom message."""
        err = CollectionError(
            collection="test_col",
            operation="delete",
            message="Collection is protected",
        )

        assert str(err) == "Collection is protected"
        assert err.collection == "test_col"

    def test_point_operation_error(self):
        """PointOperationError captures point operation failures."""
        err = PointOperationError(
            collection="vectors",
            operation="upsert",
            point_count=100,
        )

        assert err.collection == "vectors"
        assert err.operation == "upsert"
        assert err.point_count == 100
        assert "upsert" in str(err).lower()
        assert "100" in str(err)
        assert isinstance(err, StorageError)

    def test_point_operation_error_defaults(self):
        """PointOperationError has sensible defaults."""
        err = PointOperationError("col", "delete")

        assert err.point_count == 0


# =============================================================================
# ErrorCollector Tests
# =============================================================================


class TestErrorCollector:
    """Tests for ErrorCollector batch error handling."""

    def test_empty_collector(self):
        """Empty collector has correct initial state."""
        collector = ErrorCollector()

        assert not collector.has_errors
        assert collector.error_count == 0
        assert collector.warning_count == 0
        assert collector.critical_count == 0
        assert collector.truncated_count == 0
        assert collector.format_summary() == ""

    def test_add_error(self):
        """Can add errors with full details."""
        collector = ErrorCollector()
        exc = ValueError("invalid value")

        collector.add(
            path="/path/to/file.py",
            message="Processing failed",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.PARSE_ERROR,
            exception=exc,
        )

        assert collector.has_errors
        assert collector.error_count == 1
        err = collector.errors[0]
        assert err.path == "/path/to/file.py"
        assert err.message == "Processing failed"
        assert err.severity == ErrorSeverity.ERROR
        assert err.category == ErrorCategory.PARSE_ERROR
        assert err.exception_type == "ValueError"
        assert "invalid value" in err.exception_message

    def test_convenience_methods(self):
        """Convenience methods add correct categories."""
        collector = ErrorCollector()

        collector.add_file_access_error("/a.txt")
        collector.add_encoding_error("/b.txt")
        collector.add_parse_error("/c.txt")
        collector.add_embedding_error("/d.txt")
        collector.add_storage_error("/e.txt")

        by_cat = collector.by_category()
        assert ErrorCategory.FILE_ACCESS in by_cat
        assert ErrorCategory.ENCODING in by_cat
        assert ErrorCategory.PARSE_ERROR in by_cat
        assert ErrorCategory.EMBEDDING in by_cat
        assert ErrorCategory.STORAGE in by_cat

    def test_severity_counts(self):
        """Severity counts are accurate."""
        collector = ErrorCollector()

        collector.add("/a", "warn", ErrorSeverity.WARNING, ErrorCategory.UNKNOWN)
        collector.add("/b", "warn", ErrorSeverity.WARNING, ErrorCategory.UNKNOWN)
        collector.add("/c", "err", ErrorSeverity.ERROR, ErrorCategory.UNKNOWN)
        collector.add("/d", "crit", ErrorSeverity.CRITICAL, ErrorCategory.UNKNOWN)

        assert collector.warning_count == 2
        assert collector.error_count == 4  # Total, not just ERROR severity
        assert collector.critical_count == 1

    def test_max_errors_truncation(self):
        """Collector truncates after max_errors."""
        collector = ErrorCollector(max_errors=5)

        for i in range(10):
            collector.add(f"/file{i}.txt", f"Error {i}")

        assert len(collector.errors) == 5
        assert collector.truncated_count == 5

    def test_format_summary(self):
        """Summary formatting includes key information."""
        collector = ErrorCollector()

        collector.add_file_access_error("/missing.txt", FileNotFoundError("not found"))
        collector.add_parse_error("/bad.json", ValueError("invalid json"))

        summary = collector.format_summary(include_exceptions=True)

        assert "issues during indexing" in summary.lower()
        assert "/missing.txt" in summary
        assert "/bad.json" in summary
        assert "FileNotFoundError" in summary

    def test_clear_errors(self):
        """Clear resets collector state."""
        collector = ErrorCollector(max_errors=5)

        for i in range(10):
            collector.add(f"/file{i}.txt", f"Error {i}")

        collector.clear()

        assert not collector.has_errors
        assert collector.truncated_count == 0

    def test_traceback_capture(self):
        """Tracebacks are captured when enabled."""
        collector = ErrorCollector(capture_tracebacks=True)

        try:
            raise ValueError("test error")
        except ValueError as e:
            collector.add("/test.py", "Test error", exception=e)

        assert collector.errors[0].traceback_str is not None
        assert "ValueError" in collector.errors[0].traceback_str

    def test_traceback_capture_disabled(self):
        """Tracebacks not captured when disabled."""
        collector = ErrorCollector(capture_tracebacks=False)

        try:
            raise ValueError("test error")
        except ValueError as e:
            collector.add("/test.py", "Test error", exception=e)

        assert collector.errors[0].traceback_str is None
        assert collector.errors[0].exception_type == "ValueError"


# =============================================================================
# Error Response Tests
# =============================================================================


class TestErrorResponse:
    """Tests for API error response helpers."""

    def test_error_response_basic(self):
        """error_response creates correct structure."""
        resp = error_response(ErrorCode.NOT_FOUND, "Item not found")

        assert resp["error_code"] == "not_found"
        assert resp["message"] == "Item not found"
        assert "details" not in resp

    def test_error_response_with_details(self):
        """error_response includes details when provided."""
        resp = error_response(
            ErrorCode.VALIDATION_FAILED,
            "Invalid input",
            details={"field": "email", "reason": "Invalid format"},
        )

        assert resp["error_code"] == "validation_failed"
        assert resp["details"]["field"] == "email"

    def test_is_error_response(self):
        """is_error_response correctly identifies errors."""
        err = error_response(ErrorCode.INTERNAL_ERROR, "Oops")
        success = {"status": "ok", "data": [1, 2, 3]}

        assert is_error_response(err)
        assert not is_error_response(success)


# =============================================================================
# GlobalVocabulary Error Tests
# =============================================================================


class TestGlobalVocabularyErrors:
    """Tests for GlobalVocabulary error handling."""

    def test_cache_refresh_uses_configurable_ttl(self, tmp_path):
        """GlobalVocabulary uses configurable cache TTL."""
        from vector_core.embeddings.global_vocab import GlobalVocabulary

        vocab = GlobalVocabulary(db_path=tmp_path / "test.db", cache_ttl=0.1)
        assert vocab._cache_ttl == 0.1
        vocab.close()

    def test_cache_refresh_default_from_settings(self, tmp_path):
        """GlobalVocabulary uses settings default when not specified."""
        from vector_core.embeddings.global_vocab import GlobalVocabulary
        from vector_core.settings import settings

        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")
        assert vocab._cache_ttl == settings.global_vocab_cache_ttl
        vocab.close()

    def test_database_created_on_init(self, tmp_path):
        """GlobalVocabulary creates database file on initialization."""
        from vector_core.embeddings.global_vocab import GlobalVocabulary

        db_path = tmp_path / "vocab.db"
        vocab = GlobalVocabulary(db_path=db_path)

        assert db_path.exists()
        vocab.close()

    def test_register_codebase_updates_vocab(self, tmp_path):
        """register_codebase updates vocabulary with tokens."""
        from vector_core.embeddings.global_vocab import GlobalVocabulary

        vocab = GlobalVocabulary(db_path=tmp_path / "test.db")

        # Register with some document tokens
        tokens_per_doc = [{"hello", "world"}, {"hello", "python"}]
        unique_token_count = vocab.register_codebase("test_codebase", tokens_per_doc)

        # Returns count of unique tokens (hello, world, python = 3)
        assert unique_token_count > 0
        assert vocab.vocab_size > 0
        assert vocab.total_docs == 2
        vocab.close()


# =============================================================================
# FactStore Error Tests
# =============================================================================


class TestFactStoreErrors:
    """Tests for FactStore error handling."""

    @pytest.fixture
    def fact_store(self, tmp_path):
        """Create a FactStore for testing."""
        from vector_core.facts.database import FactStore

        store = FactStore(db_path=tmp_path / "test_facts.db")
        yield store
        store.close()

    def test_create_duplicate_raises_error(self, fact_store):
        """Creating duplicate fact raises DuplicateFactError."""
        from vector_core.facts.models import DuplicateFactError

        # Create first fact
        fact_store.create(
            subject="Alice",
            predicate="works_at",
            object_value="Acme Corp",
        )

        # Try to create duplicate
        with pytest.raises(DuplicateFactError):
            fact_store.create(
                subject="Alice",
                predicate="works_at",
                object_value="Acme Corp",
            )

    def test_read_nonexistent_raises_error(self, fact_store):
        """Reading nonexistent fact raises FactNotFoundError."""
        from vector_core.facts.models import FactNotFoundError

        with pytest.raises(FactNotFoundError):
            fact_store.read(uuid4())

    def test_delete_nonexistent_returns_false(self, fact_store):
        """Deleting nonexistent fact returns False."""
        result = fact_store.delete(uuid4())
        assert result is False


# =============================================================================
# GlossaryStore Error Tests
# =============================================================================


class TestGlossaryStoreErrors:
    """Tests for GlossaryStore error handling."""

    @pytest.fixture
    def glossary_store(self, tmp_path):
        """Create a GlossaryStore for testing."""
        from vector_core.glossary.store import GlossaryStore

        store = GlossaryStore(db_path=tmp_path / "test_glossary.db")
        yield store
        store.close()

    def test_create_duplicate_term_raises_error(self, glossary_store):
        """Creating duplicate term raises TermExistsError."""
        from vector_core.glossary.models import TermExistsError

        glossary_store.create(
            term="API",
            expansion="Application Programming Interface",
            definition="A set of protocols.",
        )

        with pytest.raises(TermExistsError) as exc:
            glossary_store.create(
                term="API",
                expansion="Different expansion",
                definition="Different definition",
            )

        assert exc.value.term == "API"

    def test_read_nonexistent_raises_error(self, glossary_store):
        """Reading nonexistent entry raises GlossaryNotFoundError."""
        from vector_core.glossary.models import GlossaryNotFoundError

        with pytest.raises(GlossaryNotFoundError):
            glossary_store.read(uuid4())

    def test_delete_nonexistent_returns_false(self, glossary_store):
        """Deleting nonexistent entry returns False."""
        result = glossary_store.delete(uuid4())
        assert result is False


# =============================================================================
# Concurrent Error Tests
# =============================================================================


class TestConcurrentErrors:
    """Tests for error handling under concurrent access."""

    def test_concurrent_glossary_access(self, tmp_path):
        """GlossaryStore handles concurrent access safely."""
        from vector_core.glossary.store import GlossaryStore

        db_path = tmp_path / "concurrent.db"
        errors = []

        def create_entries(start, count):
            store = GlossaryStore(db_path=db_path)
            try:
                for i in range(start, start + count):
                    try:
                        store.create(
                            term=f"TERM{i}",
                            expansion=f"Expansion {i}",
                            definition=f"Definition {i}",
                        )
                    except Exception as e:
                        errors.append(e)
            finally:
                store.close()

        threads = [
            threading.Thread(target=create_entries, args=(i * 10, 10))
            for i in range(3)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without deadlock
        # Some duplicates are expected if terms overlap
        store = GlossaryStore(db_path=db_path)
        entries = store.list_all()
        store.close()

        assert len(entries) > 0  # At least some succeeded

    def test_concurrent_fact_access(self, tmp_path):
        """FactStore handles concurrent access safely."""
        from vector_core.facts.database import FactStore

        db_path = tmp_path / "concurrent_facts.db"

        def create_facts(thread_id, count):
            store = FactStore(db_path=db_path)
            try:
                for i in range(count):
                    try:
                        store.create(
                            subject=f"Thread{thread_id}",
                            predicate="created",
                            object_value=f"Fact{i}",
                        )
                    except Exception:
                        pass  # Expect some conflicts
            finally:
                store.close()

        threads = [
            threading.Thread(target=create_facts, args=(i, 10))
            for i in range(3)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without deadlock
        store = FactStore(db_path=db_path)
        facts = list(store.query())
        store.close()

        assert len(facts) > 0


# =============================================================================
# Settings Validation Tests
# =============================================================================


class TestSettingsValidation:
    """Tests for settings validation edge cases."""

    def test_global_vocab_cache_ttl_zero_allowed(self):
        """Zero cache TTL is allowed (disables caching)."""
        from vector_core.settings import VectorCoreSettings

        # Should not raise
        settings = VectorCoreSettings(global_vocab_cache_ttl=0.0)
        assert settings.global_vocab_cache_ttl == 0.0

    def test_global_vocab_cache_ttl_negative_rejected(self):
        """Negative cache TTL is rejected."""
        from pydantic import ValidationError
        from vector_core.settings import VectorCoreSettings

        with pytest.raises(ValidationError):
            VectorCoreSettings(global_vocab_cache_ttl=-1.0)

    def test_content_hash_display_length_positive(self):
        """Content hash display length must be positive."""
        from pydantic import ValidationError
        from vector_core.settings import VectorCoreSettings

        with pytest.raises(ValidationError):
            VectorCoreSettings(content_hash_display_length=0)

        with pytest.raises(ValidationError):
            VectorCoreSettings(content_hash_display_length=-1)
