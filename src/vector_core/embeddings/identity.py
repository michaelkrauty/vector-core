"""Explicit identity of one dense embedding space."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

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
    max_text_chars: int | None = None
    profile: str = "raw"
    query_prefix: str = ""
    document_prefix: str = ""
    max_input_bytes: int = 0
    max_input_tokens: int | None = None
    tokenizer_fingerprint: str | None = None
    tokenizer_add_special_tokens: bool = True
    reserved_tokens: int = 0
    endpoint_auth_fingerprint: str | None = None
    input_encoding: str = "text"
    tokenizer_implementation: str | None = None
    tokenizer_version: str | None = None

    def __post_init__(self) -> None:  # noqa: PLR0912 - validate persisted identity at the boundary
        if not isinstance(self.model, str) or not isinstance(self.endpoint, str):
            raise ValueError("Embedding identity model and endpoint must be strings")
        if self.namespace is not None and not isinstance(self.namespace, str):
            raise ValueError("Embedding identity namespace must be a string or None")
        if type(self.dimension) is not int or self.dimension <= 0:
            raise ValueError("Embedding identity requires a resolved positive dimension")
        if self.max_text_chars is not None and (
            type(self.max_text_chars) is not int or self.max_text_chars <= 0
        ):
            raise ValueError("Embedding identity requires a positive text limit")
        if type(self.tokenizer_add_special_tokens) is not bool:
            raise ValueError("Embedding tokenizer special-token policy must be a bool")
        if type(self.reserved_tokens) is not int or self.reserved_tokens < 0:
            raise ValueError("Embedding reserved token count must be non-negative")
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
        if self.endpoint_auth_fingerprint is not None and (
            not isinstance(self.endpoint_auth_fingerprint, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.endpoint_auth_fingerprint) is None
        ):
            raise ValueError("Embedding endpoint auth fingerprint must be a SHA-256 digest")
        if not isinstance(self.input_encoding, str) or self.input_encoding not in {
            "text",
            "token_ids",
        }:
            raise ValueError("Unknown embedding input encoding")
        if self.input_encoding == "token_ids":
            if (
                not isinstance(self.tokenizer_fingerprint, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.tokenizer_fingerprint) is None
            ):
                raise ValueError("Token-ID embedding identity requires a tokenizer SHA-256 digest")
            if not all(
                isinstance(value, str) and value.strip()
                for value in (self.tokenizer_implementation, self.tokenizer_version)
            ):
                raise ValueError(
                    "Token-ID embedding identity requires tokenizer implementation/version"
                )
        elif self.tokenizer_implementation is not None or self.tokenizer_version is not None:
            raise ValueError("Tokenizer implementation/version identity is only used for token IDs")
        endpoint = self.endpoint.rstrip("/")
        parsed = urlsplit(endpoint)
        if "@" in parsed.netloc:
            userinfo, _, host = parsed.netloc.rpartition("@")
            # Persist only an opaque discriminator; the HTTP client retains its original URL.
            object.__setattr__(
                self, "endpoint_auth_fingerprint", hashlib.sha256(userinfo.encode()).hexdigest()
            )
            endpoint = urlunsplit(parsed._replace(netloc=host))
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "namespace", self.namespace or None)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # Keep unauthenticated identities compatible with existing manifests and cache keys.
        if self.endpoint_auth_fingerprint is None:
            value.pop("endpoint_auth_fingerprint")
        # Existing text-mode generations and cache entries keep their exact identity.
        if self.input_encoding == "text":
            for field in ("input_encoding", "tokenizer_implementation", "tokenizer_version"):
                value.pop(field)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EmbeddingIdentity":
        return cls(**value)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()
