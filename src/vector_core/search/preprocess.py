"""Query preprocessing: parsing, expansion, normalization."""

import re
from dataclasses import dataclass, field

# Compiled patterns for camelCase expansion (module-level for performance)
# Matches lowercase -> uppercase boundary: "getUser" -> "get User"
CAMEL_LOWER_UPPER = re.compile(r"([a-z])([A-Z])")
# Matches uppercase run followed by uppercase+lowercase: "XMLParser" -> "XML Parser"
CAMEL_UPPER_LOWER = re.compile(r"([A-Z]+)([A-Z][a-z])")


@dataclass
class ProcessedQuery:
    """Processed query with extracted components."""

    text: str  # Processed text for embedding
    original: str  # Original query
    fields: dict[str, str] = field(default_factory=dict)  # Extracted field:value pairs
    expanded_terms: list[str] = field(default_factory=list)  # Added terms from expansion


class QueryPreprocessor:
    """
    Preprocesses search queries.

    Features:
    - Field prefix extraction (e.g., "tag:important")
    - CamelCase expansion
    - Synonym expansion (configurable)
    - Text normalization
    """

    def __init__(
        self,
        synonyms: dict[str, list[str]] | None = None,
        field_prefixes: list[str] | None = None,
    ):
        """
        Initialize query preprocessor.

        Args:
            synonyms: Map of term -> list of synonyms to expand
            field_prefixes: List of recognized field prefixes (e.g., ["tag:", "category:"])
        """
        self.synonyms = synonyms or {}
        self.field_prefixes = field_prefixes or []

    def expand_camelcase(self, text: str) -> tuple[str, list[str]]:
        """
        Expand camelCase and PascalCase to space-separated words.

        Example: "getUserData" -> ("getUserData get user data", ["get", "user", "data"])

        Args:
            text: Text to expand

        Returns:
            Tuple of (expanded text, list of added terms)
        """
        # Split on lowercase->uppercase boundary
        expanded = CAMEL_LOWER_UPPER.sub(r"\1 \2", text)
        # Split on uppercase->uppercase+lowercase
        expanded = CAMEL_UPPER_LOWER.sub(r"\1 \2", expanded)

        if expanded != text:
            # Extract the new terms
            original_words = set(text.lower().split())
            expanded_words = expanded.lower().split()
            new_terms = [w for w in expanded_words if w not in original_words]
            return f"{text} {expanded.lower()}", new_terms

        return text, []

    def expand_synonyms(self, text: str) -> tuple[str, list[str]]:
        """
        Expand query with synonyms.

        Args:
            text: Text to expand

        Returns:
            Tuple of (expanded text, list of added terms)
        """
        if not self.synonyms:
            return text, []

        # Find all word tokens
        tokens = list(re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', text.lower()))

        if not tokens:
            return text, []

        # Build result by processing each token
        result_parts = []
        added_terms = []
        last_end = 0

        for match in tokens:
            # Add characters between tokens
            if match.start() > last_end:
                result_parts.append(text[last_end:match.start()])

            word = match.group(1)
            result_parts.append(word)

            # Add synonyms if any
            if word in self.synonyms:
                syns = self.synonyms[word]
                result_parts.append(" " + " ".join(syns))
                added_terms.extend(syns)

            last_end = match.end()

        # Add trailing characters
        if last_end < len(text):
            result_parts.append(text[last_end:])

        return "".join(result_parts), added_terms

    def extract_fields(self, query: str) -> tuple[str, dict[str, str]]:
        """
        Extract field:value pairs from query.

        Example: "tag:urgent find error" -> ("find error", {"tag": "urgent"})

        Args:
            query: Query string

        Returns:
            Tuple of (remaining text, extracted fields dict)
        """
        fields: dict[str, str] = {}
        remaining = query

        for prefix in self.field_prefixes:
            # Match prefix:value (value is non-whitespace)
            pattern = rf'\b{re.escape(prefix)}(\S+)'
            for match in re.finditer(pattern, remaining, re.I):
                field_name = prefix.rstrip(":")
                fields[field_name] = match.group(1)

            # Remove matched patterns
            remaining = re.sub(pattern, '', remaining, flags=re.I)

        return remaining.strip(), fields

    def preprocess(
        self,
        query: str,
        expand_camel: bool = True,
        expand_syns: bool = True,
    ) -> ProcessedQuery:
        """
        Full query preprocessing pipeline.

        Args:
            query: Raw query string
            expand_camel: Whether to expand camelCase
            expand_syns: Whether to expand synonyms

        Returns:
            ProcessedQuery with all extracted information
        """
        # Extract fields first
        remaining, fields = self.extract_fields(query)

        # Build text for embedding
        text = remaining
        all_expanded = []

        # Expand camelCase
        if expand_camel:
            text, camel_terms = self.expand_camelcase(text)
            all_expanded.extend(camel_terms)

        # Expand synonyms
        if expand_syns:
            text, syn_terms = self.expand_synonyms(text)
            all_expanded.extend(syn_terms)

        return ProcessedQuery(
            text=text.strip(),
            original=query,
            fields=fields,
            expanded_terms=all_expanded,
        )


# Default generic synonyms (non-domain-specific)
# Used by all MCP servers for basic abbreviation expansion
DEFAULT_SYNONYMS: dict[str, list[str]] = {
    # Common abbreviations
    "fn": ["function"],
    "func": ["function"],
    "cls": ["class"],
    "err": ["error"],
    "msg": ["message"],
    "req": ["request"],
    "res": ["response"],
    "resp": ["response"],
    "cfg": ["config", "configuration"],
    "conf": ["config", "configuration"],
    "db": ["database"],
    "auth": ["authentication"],
    "ctx": ["context"],
    "env": ["environment"],
    "pkg": ["package"],
    "dep": ["dependency"],
    "deps": ["dependencies"],
    "util": ["utility"],
    "utils": ["utilities"],
    # Types
    "str": ["string"],
    "int": ["integer"],
    "bool": ["boolean"],
    "dict": ["dictionary"],
    "arr": ["array"],
    "obj": ["object"],
    "idx": ["index"],
    "len": ["length"],
    "num": ["number"],
    "max": ["maximum"],
    "min": ["minimum"],
    "avg": ["average"],
    # I/O
    "src": ["source"],
    "dst": ["destination"],
    "tmp": ["temporary"],
    "buf": ["buffer"],
    "io": ["input output"],
    # Patterns
    "init": ["initialize"],
    "impl": ["implementation"],
    "val": ["value"],
    "var": ["variable"],
    "params": ["parameters"],
    "args": ["arguments"],
    "ret": ["return"],
}


# Extended synonyms for code search (import and merge with DEFAULT_SYNONYMS)
# These are more specialized for programming domain
CODE_SEARCH_SYNONYMS: dict[str, list[str]] = {
    # Concurrency
    "async": ["asynchronous"],
    "sync": ["synchronous"],
    "mutex": ["lock", "synchronization"],
    "thread": ["concurrent", "parallel"],
    "callback": ["handler", "listener"],
    "promise": ["future", "async"],
    "future": ["promise", "async"],
    # Network
    "ws": ["websocket"],
    "http": ["protocol", "request"],
    "api": ["interface", "endpoint"],
    "url": ["address", "link"],
    "rpc": ["remote procedure call"],
    # Testing
    "test": ["verify", "check"],
    "mock": ["fake", "stub"],
    "stub": ["fake", "mock"],
    "assert": ["verify", "check"],
    "spec": ["specification", "test"],
    # Error handling
    "throw": ["raise", "error"],
    "catch": ["handle", "trap"],
    "except": ["catch", "handle"],
    # Memory
    "alloc": ["allocate", "memory"],
    "gc": ["garbage collection"],
    "cache": ["memory", "store"],
    "mem": ["memory"],
    # Data structures
    "map": ["dictionary", "key value"],
    "vec": ["vector", "array"],
    "ptr": ["pointer", "reference"],
    "ref": ["reference"],
    "queue": ["list", "fifo"],
    "stack": ["list", "lifo"],
    # Patterns
    "ctor": ["constructor"],
    "dtor": ["destructor"],
    "singleton": ["single instance", "pattern"],
    "factory": ["create", "pattern"],
    "handler": ["process", "callback"],
    "middleware": ["interceptor", "handler"],
    # Misc
    "regex": ["regular expression", "pattern"],
    "json": ["javascript object notation"],
    "sql": ["structured query language", "database"],
    "orm": ["object relational mapping"],
    "crud": ["create read update delete"],
}


def get_all_code_synonyms() -> dict[str, list[str]]:
    """
    Get combined DEFAULT_SYNONYMS + CODE_SEARCH_SYNONYMS.

    Use this for code search applications that want comprehensive
    abbreviation expansion.

    Returns:
        Combined synonym dictionary
    """
    combined = dict(DEFAULT_SYNONYMS)
    combined.update(CODE_SEARCH_SYNONYMS)
    return combined


def create_default_preprocessor(
    extra_synonyms: dict[str, list[str]] | None = None,
    extra_prefixes: list[str] | None = None,
    include_code_synonyms: bool = False,
) -> QueryPreprocessor:
    """
    Create a preprocessor with default settings.

    Args:
        extra_synonyms: Additional synonyms to include
        extra_prefixes: Additional field prefixes to recognize
        include_code_synonyms: Include CODE_SEARCH_SYNONYMS (for code search apps)

    Returns:
        Configured QueryPreprocessor
    """
    synonyms = dict(DEFAULT_SYNONYMS)
    if include_code_synonyms:
        synonyms.update(CODE_SEARCH_SYNONYMS)
    if extra_synonyms:
        synonyms.update(extra_synonyms)

    prefixes = ["path:", "-path:"]
    if extra_prefixes:
        prefixes.extend(extra_prefixes)

    return QueryPreprocessor(synonyms=synonyms, field_prefixes=prefixes)
