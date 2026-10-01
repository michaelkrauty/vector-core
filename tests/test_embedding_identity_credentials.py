"""Embedding identities must not persist HTTP Basic credentials."""

import asyncio
import base64
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx
import pytest
from qdrant_client import AsyncQdrantClient

from vector_core.embeddings.client import EmbeddingClient
from vector_core.embeddings.identity import EmbeddingIdentity
from vector_core.settings import settings
from vector_core.storage.embedding_migration import ensure_embedding_collection
from vector_core.storage.qdrant import QdrantStorage


@pytest.mark.parametrize(
    ("userinfo", "host"),
    [
        ("private-user:private-password", "example.invalid:8080"),
        ("private%40user:private%2Fpassword", "example.invalid:8080"),
        ("private-user:private-password", "[::1]:8080"),
        ("private-user", "example.invalid:8080"),
    ],
)
def test_identity_strips_userinfo_and_roundtrips(userinfo: str, host: str) -> None:
    identity = EmbeddingIdentity("model", "revision", f"http://{userinfo}@{host}/api/", 3)
    serialized = json.dumps(identity.to_dict())
    assert identity.endpoint == f"http://{host}/api"
    for credential in userinfo.split(":"):
        assert credential not in serialized
        assert unquote(credential) not in serialized
    assert identity.endpoint_auth_fingerprint is not None
    restored = EmbeddingIdentity.from_dict(identity.to_dict())
    assert restored == identity
    assert restored.fingerprint == identity.fingerprint


def test_credentials_remain_distinct_embedding_spaces() -> None:
    identities = [
        EmbeddingIdentity("model", "revision", f"http://{userinfo}example.invalid", 3)
        for userinfo in (
            "",
            "private-user:secret-one@",
            "private-user:secret-two@",
            "other:secret-one@",
        )
    ]
    assert len({identity.endpoint for identity in identities}) == 1
    assert len({identity.fingerprint for identity in identities}) == len(identities)
    assert "endpoint_auth_fingerprint" not in identities[0].to_dict()
    assert EmbeddingIdentity.from_dict(identities[0].to_dict()) == identities[0]


@pytest.mark.parametrize("fingerprint", ["private-user:private-password", "g" * 64, "abc"])
def test_identity_rejects_non_digest_auth_fingerprint(fingerprint: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        EmbeddingIdentity(
            "model", None, "http://example.invalid", 3, endpoint_auth_fingerprint=fingerprint
        )


@pytest.mark.parametrize(
    "userinfo", ["private-user:private-password", "private%40user:private%2Fpassword"]
)
async def test_manifest_is_sanitized_without_changing_http_auth(
    userinfo: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    base_url = f"http://{userinfo}@example.invalid/api"
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]})

    async def text_resolver(payload: dict[str, Any]) -> str | None:
        return None

    storage = QdrantStorage(url="http://isolated.invalid", embedding_dim=3)
    storage._client = AsyncQdrantClient(location=":memory:")
    embedder = EmbeddingClient(
        base_url=base_url, model="model", dim=3, cache_namespace="revision", global_concurrency=0
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        embedder._client = http_client
        embedder._client_loop = asyncio.get_running_loop()
        try:
            generation = await ensure_embedding_collection(
                storage, "corpus", embedder, text_resolver
            )
            metadata = await storage.get_metadata(generation.physical_name)
            assert metadata is not None
            serialized = json.dumps(metadata)
            for credential in userinfo.split(":"):
                assert credential not in serialized
                assert unquote(credential) not in serialized
            manifest_identity = metadata["embedding_generation"]["identity"]
            assert EmbeddingIdentity.from_dict(manifest_identity) == generation.identity
            assert embedder.base_url == base_url
            assert len(requests) == 1
            assert requests[0].url.path == "/api/v1/embeddings"
            expected_auth = base64.b64encode(unquote(userinfo).encode()).decode()
            assert requests[0].headers["Authorization"] == f"Basic {expected_auth}"
            assert (
                await ensure_embedding_collection(storage, "corpus", embedder, text_resolver)
                == generation
            )
        finally:
            await embedder.close()
            await storage.close()
