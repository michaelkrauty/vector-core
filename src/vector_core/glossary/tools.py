"""Shared helper for glossary MCP tools.

Servers wrap these methods with thin MCP decorators.
All methods are async to match indexing/embedding operations.
"""


from vector_core.errors import ErrorCode, error_response
from vector_core.utils.sentinel import UNSET, UnsetType
from vector_core.glossary.indexer import GlossaryIndexer
from vector_core.glossary.models import (
    GlossaryEntry,
    TermExistsError,
)
from vector_core.glossary.store import GlossaryStore


class GlossaryToolHelper:
    """
    Shared async-first logic for glossary MCP tools.

    Servers wrap with thin MCP decorators:

    ```python
    @mcp.tool()
    async def add_glossary_entry(...) -> dict:
        helper = get_glossary_helper()
        return await helper.add_entry(...)
    ```
    """

    def __init__(
        self,
        store: GlossaryStore,
        indexer: GlossaryIndexer | None = None,
    ):
        """
        Initialize helper.

        Args:
            store: GlossaryStore for CRUD operations
            indexer: Optional GlossaryIndexer for search and indexing.
                     If None, search operations will return errors and
                     add/update/delete won't update the vector index.
        """
        self.store = store
        self.indexer = indexer

    def _entry_to_dict(self, entry: GlossaryEntry) -> dict:
        """Convert entry to dict for MCP response."""
        return entry.to_dict()

    async def add_entry(
        self,
        term: str,
        expansion: str,
        definition: str,
        domain: str | None = None,
        aliases: list[str] | None = None,
    ) -> dict:
        """
        Add a new glossary entry.

        Args:
            term: Canonical term (e.g., "USAF")
            expansion: Full expansion (e.g., "United States Air Force")
            definition: Detailed definition
            domain: Optional category (e.g., "military", "tech")
            aliases: Optional alternative terms

        Returns:
            Created entry as dict
        """
        try:
            entry = self.store.create(term, expansion, definition, domain, aliases)
            if self.indexer is not None:
                await self.indexer.index_entry(entry.id)
            return self._entry_to_dict(entry)
        except TermExistsError as e:
            return error_response(ErrorCode.DUPLICATE, f"Term already exists: {e.term}")

    async def lookup(self, term: str) -> dict:
        """
        Exact lookup by term or alias (case-insensitive).

        Args:
            term: Term to look up

        Returns:
            Glossary entry if found, or error dict
        """
        entry = self.store.lookup(term)
        if entry is None:
            return error_response(ErrorCode.GLOSSARY_NOT_FOUND, f"Term not found: {term}")
        return self._entry_to_dict(entry)

    async def search(
        self,
        query: str,
        domain: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Semantic search for glossary entries.

        Args:
            query: Natural language search query
            domain: Optional domain filter
            limit: Max results (default 10, max 100)

        Returns:
            List of matching entries with relevance scores, or error dict
        """
        if self.indexer is None:
            return [error_response(ErrorCode.SERVICE_UNAVAILABLE, "Search not available: indexer not configured")]
        limit = min(limit, 100)
        return await self.indexer.search(query, domain, limit)

    async def list_entries(
        self,
        domain: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        List all glossary entries.

        Args:
            domain: Optional domain filter
            limit: Max results (default 50, max 100)

        Returns:
            List of glossary entry summaries
        """
        limit = min(limit, 100)
        entries = self.store.list_all(domain, limit)
        return [e.to_dict() for e in entries]

    async def update_entry(
        self,
        term_or_id: str,
        term: str | None = None,
        expansion: str | None = None,
        definition: str | None = None,
        domain: str | None | UnsetType = UNSET,
        aliases: list[str] | None | UnsetType = UNSET,
    ) -> dict:
        """
        Update an existing glossary entry.

        Args:
            term_or_id: Term (case-insensitive) or UUID to identify the entry
            term: New canonical term (optional)
            expansion: New expansion (optional)
            definition: New definition (optional)
            domain: New domain (optional, pass None to clear)
            aliases: New aliases (optional, replaces existing)

        Returns:
            Updated entry as dict, or error dict
        """
        # Find the entry
        entry = self.store.find_by_term_or_id(term_or_id)
        if entry is None:
            return error_response(ErrorCode.GLOSSARY_NOT_FOUND, f"Entry not found: {term_or_id}")

        try:
            # Build update kwargs, only passing explicitly provided values
            kwargs: dict = {}
            if term is not None:
                kwargs["term"] = term
            if expansion is not None:
                kwargs["expansion"] = expansion
            if definition is not None:
                kwargs["definition"] = definition
            if domain is not UNSET:
                kwargs["domain"] = domain
            if aliases is not UNSET:
                kwargs["aliases"] = aliases

            updated = self.store.update(entry.id, **kwargs)
            if self.indexer is not None:
                await self.indexer.index_entry(updated.id)
            return self._entry_to_dict(updated)
        except TermExistsError as e:
            return error_response(ErrorCode.DUPLICATE, f"Term already exists: {e.term}")

    async def delete_entry(self, term_or_id: str) -> dict:
        """
        Delete a glossary entry.

        Args:
            term_or_id: Term (case-insensitive) or UUID string

        Returns:
            Success status or error dict
        """
        # Find the entry
        entry = self.store.find_by_term_or_id(term_or_id)
        if entry is None:
            return error_response(ErrorCode.GLOSSARY_NOT_FOUND, f"Entry not found: {term_or_id}")

        entry_id = entry.id
        if self.indexer is not None:
            await self.indexer.delete_entry_index(entry_id)
        self.store.delete(entry_id)
        return {"success": True, "deleted_id": str(entry_id)}

    async def get_domains(self) -> list[str]:
        """
        Get list of all unique domains.

        Returns:
            List of domain names
        """
        return self.store.get_domains()

    async def get_stats(self) -> dict:
        """
        Get glossary statistics.

        Returns:
            Stats dict with count, domains
        """
        return {
            "count": self.store.count(),
            "domains": self.store.get_domains(),
        }
