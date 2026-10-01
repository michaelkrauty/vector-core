"""Explicit identity of one dense embedding space."""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from vector_core.embeddings.cache import EMBEDDING_PREPROCESSING_VERSION


@dataclass(frozen=True)
class EmbeddingIdentity:
    """A deployment assertion, not an inferred fingerprint of server weights.

    Change ``namespace`` whenever a model alias points to different weights or
    serving behavior. The embeddings protocol cannot attest that revision.
    """

    model: str
    namespace: str | None
    endpoint: str
    dimension: int
    preprocessing: str = EMBEDDING_PREPROCESSING_VERSION
    max_text_chars: int = 8000
    profile: str = "raw"
    query_prefix: str = ""
    document_prefix: str = ""
    max_input_bytes: int = 0
    max_input_tokens: int | None = None
    tokenizer_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not isinstance(self.endpoint, str):
            raise ValueError("Embedding identity model and endpoint must be strings")
        if self.namespace is not None and not isinstance(self.namespace, str):
            raise ValueError("Embedding identity namespace must be a string or None")
        if type(self.dimension) is not int or self.dimension <= 0:
            raise ValueError("Embedding identity requires a resolved positive dimension")
        if type(self.max_text_chars) is not int or self.max_text_chars <= 0:
            raise ValueError("Embedding identity requires a positive text limit")
        if not isinstance(self.preprocessing, str) or not self.preprocessing:
            raise ValueError("Embedding identity requires a preprocessing version")
        if self.profile not in {"raw", "qwen3", "nemotron3"}:
            raise ValueError("Unknown embedding formatting profile")
        if not isinstance(self.query_prefix, str) or not isinstance(self.document_prefix, str):
            raise ValueError("Embedding identity prefixes must be strings")
        if type(self.max_input_bytes) is not int or self.max_input_bytes < 0:
            raise ValueError("Embedding identity requires a non-negative byte limit")
        if self.max_input_tokens is not None and (
            type(self.max_input_tokens) is not int or self.max_input_tokens <= 0
        ):
            raise ValueError("Embedding identity requires a positive token limit")
        object.__setattr__(self, "endpoint", self.endpoint.rstrip("/"))
        object.__setattr__(self, "namespace", self.namespace or None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EmbeddingIdentity":
        return cls(**value)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()
