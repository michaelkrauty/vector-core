"""Embedding generation and caching."""

from vector_core.embeddings.cache import EmbeddingCache
from vector_core.embeddings.client import (
    EmbeddingClient,
    EmbeddingServiceError,
    SyncEmbeddingClient,
)
from vector_core.embeddings.global_vocab import GlobalVocabulary
from vector_core.embeddings.limiter import GlobalRequestLimiter
from vector_core.embeddings.sparse import SparseVector, SparseVectorizer
from vector_core.embeddings.tokenization import (
    DEFAULT_STOP_TOKENS,
    default_tokenize,
    levenshtein_distance,
    levenshtein_similarity,
)

__all__ = [
    "EmbeddingClient",
    "SyncEmbeddingClient",
    "EmbeddingServiceError",
    "EmbeddingCache",
    "GlobalRequestLimiter",
    "SparseVectorizer",
    "SparseVector",
    "GlobalVocabulary",
    # Tokenization utilities
    "DEFAULT_STOP_TOKENS",
    "default_tokenize",
    "levenshtein_distance",
    "levenshtein_similarity",
]
