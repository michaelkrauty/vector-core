"""Qdrant storage layer with hybrid search support."""

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    ScoredPoint,
    SparseVectorParams,
    VectorParams,
)
from qdrant_client.models import (
    SparseVector as QdrantSparseVector,
)

from vector_core.embeddings.sparse import SparseVector
from vector_core.settings import settings

logger = logging.getLogger(__name__)

# Type alias for point IDs - Qdrant accepts both int and str
PointId = int | str


class QdrantConnectionError(Exception):
    """Raised when Qdrant is unavailable or connection fails."""
    pass


def generate_collection_name(identifier: str, prefix: str = "vc") -> str:
    """
    Generate deterministic collection name from identifier.

    If settings.collection_name is set, returns that value directly
    (for unified servers like mcp-notes/mcp-docs sharing a collection).

    Args:
        identifier: Unique identifier (e.g., absolute path)
        prefix: Collection name prefix

    Returns:
        Collection name in format "{prefix}_{hash[:12]}" or settings override
    """
    # Check for explicit override first
    if settings.collection_name:
        return settings.collection_name

    # Normalize: remove trailing slashes for consistent hashing
    normalized = identifier.rstrip("/")
    hash_val = hashlib.sha256(normalized.encode()).hexdigest()[:12]
    return f"{prefix}_{hash_val}"


def generate_point_id(key: str) -> str:
    """
    Generate deterministic UUID point ID from key.

    Creates a deterministic version-4 UUID from the first 128 bits of a
    SHA256 hash of the key. Qdrant requires point IDs to be either unsigned
    integers or UUIDs.

    Args:
        key: Unique key string (e.g., "file:/path/to/file.py")

    Returns:
        UUID string (e.g., "a5fecfb3-6489-4eec-a090-03234f11ae4f")

    Example:
        >>> generate_point_id("chunk:src/main.py:42")
        'a5fecfb3-6489-4eec-a090-03234f11ae4f'
    """
    hash_bytes = hashlib.sha256(key.encode()).digest()
    # Take first 16 bytes (128 bits) and format as UUID
    # Set version (4) and variant (RFC 4122) bits
    uuid_bytes = bytearray(hash_bytes[:16])
    uuid_bytes[6] = (uuid_bytes[6] & 0x0F) | 0x40  # Version 4
    uuid_bytes[8] = (uuid_bytes[8] & 0x3F) | 0x80  # RFC 4122 variant
    hex_str = uuid_bytes.hex()
    return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:]}"


class QdrantStorage:
    """
    Qdrant storage with hybrid dense+sparse vectors.

    Provides:
    - Collection management (create, delete, list)
    - Point operations (upsert, delete, scroll)
    - Metadata storage (ID=0 pattern)
    - Connection management with retry

    Point IDs:
        This class accepts both integer and string point IDs for flexibility.
        Use `generate_point_id()` for collision-resistant deterministic IDs.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        embedding_dim: int | None = None,
        health_check_interval: float = 30.0,
    ):
        """
        Initialize Qdrant storage.

        Args:
            url: Qdrant server URL. Default from settings.
            api_key: Optional API key. Default from settings.
            embedding_dim: Embedding dimension. Default from settings.
            health_check_interval: Seconds between health checks. Default 30s.
        """
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.embedding_dim = embedding_dim or settings.embedding_dim
        self._client: AsyncQdrantClient | None = None
        self._health_check_interval = health_check_interval
        self._last_health_check: float = 0.0
        self._healthy = False

    async def _get_client(self) -> AsyncQdrantClient:
        """Get or create Qdrant client with periodic health checking."""
        if self._client is None:
            self._client = AsyncQdrantClient(
                url=self.url,
                api_key=self.api_key,
            )
            self._healthy = False
            self._last_health_check = 0.0

        # Check health periodically
        now = time.monotonic()
        if now - self._last_health_check >= self._health_check_interval:
            if not await self.check_health():
                # Connection unhealthy - force reconnect
                logger.warning(
                    f"Qdrant health check failed, reconnecting to {self.url}"
                )
                await self._reconnect()

        return self._client

    async def _reconnect(self) -> None:
        """Close and recreate the client connection."""
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.debug(f"Error closing stale Qdrant connection: {e}")
            self._client = None
        self._healthy = False
        self._last_health_check = 0.0

        # Create new client
        self._client = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key,
        )

    async def check_health(self, timeout: float = 5.0) -> bool:
        """
        Check Qdrant server health.

        Performs a lightweight collections list request to verify connectivity.
        Updates internal health state and timestamp.

        Args:
            timeout: Health check timeout in seconds

        Returns:
            True if healthy, False otherwise
        """
        if self._client is None:
            return False

        try:
            async with asyncio.timeout(timeout):
                await self._client.get_collections()
            self._healthy = True
            self._last_health_check = time.monotonic()
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Qdrant health check timed out after {timeout}s")
            self._healthy = False
            return False
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            self._healthy = False
            return False

    @property
    def is_healthy(self) -> bool:
        """Check if Qdrant connection is known to be healthy (without triggering check)."""
        return self._healthy

    async def close(self) -> None:
        """Close the Qdrant client."""
        if self._client:
            await self._client.close()
            self._client = None
        self._healthy = False
        self._last_health_check = 0.0

    async def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        client = await self._get_client()
        try:
            collections = await client.get_collections()
            return any(c.name == name for c in collections.collections)
        except httpx.ConnectError as e:
            raise QdrantConnectionError(
                f"Cannot connect to Qdrant at {self.url}. "
                f"Ensure Qdrant is running. Error: {e}"
            ) from e
        except Exception as e:
            if "connect" in str(e).lower() or "refused" in str(e).lower():
                raise QdrantConnectionError(
                    f"Qdrant connection failed at {self.url}: {e}"
                ) from e
            raise

    async def create_collection(
        self,
        name: str,
        dense_dim: int | None = None,
    ) -> None:
        """
        Create collection with hybrid vector config.

        Args:
            name: Collection name
            dense_dim: Dense vector dimension. Default from settings.
        """
        client = await self._get_client()
        dim = dense_dim or self.embedding_dim

        if dim == 0:
            raise RuntimeError(
                "embedding_dim not yet initialized (still 0). Either call embed_batch() first "
                "to auto-detect the dimension, or set VECTOR_EMBEDDING_DIM explicitly."
            )

        await client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )

    async def delete_collection(self, name: str) -> None:
        """Delete collection."""
        client = await self._get_client()
        await client.delete_collection(name)

    async def list_collections(self, prefix: str | None = None) -> list[str]:
        """
        List collections, optionally filtered by prefix.

        Args:
            prefix: Optional prefix filter

        Returns:
            List of collection names
        """
        client = await self._get_client()
        collections = await client.get_collections()
        names = [c.name for c in collections.collections]
        if prefix:
            names = [n for n in names if n.startswith(prefix)]
        return names

    async def upsert_point(
        self,
        collection: str,
        point_id: PointId,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        payload: dict[str, Any],
    ) -> None:
        """
        Upsert a single point.

        Args:
            collection: Collection name
            point_id: Point ID (int or str). Use generate_point_id() for safe IDs.
            dense_vector: Dense embedding vector
            sparse_vector: Sparse TF-IDF vector
            payload: Point payload/metadata
        """
        client = await self._get_client()

        point = PointStruct(
            id=point_id,
            vector={
                "dense": dense_vector,
                "sparse": QdrantSparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values,
                ),
            },
            payload=payload,
        )

        await client.upsert(collection, [point])

    async def upsert_batch(
        self,
        collection: str,
        points: list[PointStruct],
        batch_size: int = 100,
        concurrency: int = 1,
        max_retries: int = 3,
    ) -> None:
        """
        Batch upsert points with concurrent batch processing and retry logic.

        Args:
            collection: Collection name
            points: Points to upsert
            batch_size: Size of each batch
            concurrency: Max concurrent upserts
            max_retries: Max retry attempts per batch on transient failures
        """
        if not points:
            return

        client = await self._get_client()

        # Split into batches
        batches = [
            points[i : i + batch_size]
            for i in range(0, len(points), batch_size)
        ]

        # Use semaphore to limit concurrent upserts
        semaphore = asyncio.Semaphore(concurrency)

        async def upsert_with_retry(batch: list[PointStruct]) -> None:
            """Upsert a batch with exponential backoff retry (capped at 8 seconds)."""
            async with semaphore:
                last_error = None
                max_delay = 8.0  # Cap maximum delay to prevent excessive waits
                for attempt in range(max_retries):
                    try:
                        await client.upsert(collection, batch)
                        return
                    except Exception as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            # Exponential backoff: 1s, 2s, 4s... capped at max_delay
                            delay = min(2 ** attempt, max_delay)
                            await asyncio.sleep(delay)
                if last_error:
                    raise last_error

        # Run all batches concurrently (limited by semaphore)
        try:
            async with asyncio.timeout(settings.qdrant_operation_timeout):
                await asyncio.gather(
                    *(upsert_with_retry(batch) for batch in batches)
                )
        except TimeoutError as e:
            # Bare TimeoutError has empty __str__, so the MCP tool layer
            # surfaces a useless "Error executing tool X: " at the client.
            # Raise with an operation-specific message and preserve the chain.
            msg = (
                f"Qdrant upsert_batch timed out after "
                f"{settings.qdrant_operation_timeout}s "
                f"(collection={collection!r}, {len(batches)} batches, "
                f"{len(points)} points)"
            )
            logger.error(msg)
            raise TimeoutError(msg) from e

    async def delete_by_filter(
        self,
        collection: str,
        field: str,
        value: Any,
    ) -> None:
        """
        Delete points matching a filter condition.

        Args:
            collection: Collection name
            field: Payload field to filter on
            value: Value to match
        """
        client = await self._get_client()

        await client.delete(
            collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key=field, match=MatchValue(value=value)),
                ],
            ),
        )

    async def delete_points(
        self,
        collection: str,
        point_ids: Sequence[str | int],
    ) -> None:
        """
        Delete points by their IDs.

        Args:
            collection: Collection name
            point_ids: IDs of the points to delete. If empty, this is a no-op
                and no request is sent to Qdrant.
        """
        if not point_ids:
            return

        client = await self._get_client()

        await client.delete(
            collection,
            points_selector=PointIdsList(points=list(point_ids)),
        )

    async def update_payload(
        self,
        collection: str,
        filter_conditions: Sequence[FieldCondition],
        payload: dict[str, Any],
    ) -> None:
        """
        Update payload fields for points matching a filter condition.

        Args:
            collection: Collection name
            filter_conditions: Filter conditions to match points
            payload: Payload fields to update
        """
        client = await self._get_client()

        await client.set_payload(
            collection_name=collection,
            payload=payload,
            points=Filter(must=list(filter_conditions)),
        )

    async def scroll_points(
        self,
        collection: str,
        filter_conditions: Sequence[FieldCondition] | None = None,
        payload_fields: list[str] | None = None,
        limit: int = 5000,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Scroll through points matching filter with bounded memory usage.

        Args:
            collection: Collection name
            filter_conditions: Optional filter conditions
            payload_fields: Specific payload fields to retrieve (None = all)
            limit: Max points per scroll request
            max_results: Maximum total results to return (None = use settings.scroll_max_results).
                         Set to 0 for truly unlimited. Prevents memory exhaustion on large collections.

        Returns:
            List of point payloads (may be truncated if max_results reached)
        """
        if max_results is None:
            max_results = settings.scroll_max_results
        client = await self._get_client()

        results: list[dict[str, Any]] = []
        offset = None
        query_filter = Filter(must=list(filter_conditions)) if filter_conditions else None

        while True:
            points, offset = await client.scroll(
                collection,
                scroll_filter=query_filter,
                limit=limit,
                offset=offset,
                with_payload=payload_fields or True,
            )

            for point in points:
                if point.payload:
                    results.append(dict(point.payload))
                    # Check if we've reached the limit (0 = unlimited)
                    if max_results and len(results) >= max_results:
                        return results

            if offset is None:
                break

        return results

    async def store_metadata(
        self,
        collection: str,
        metadata: dict[str, Any],
    ) -> None:
        """
        Store collection metadata at reserved point ID=0.

        Args:
            collection: Collection name
            metadata: Metadata dict to store
        """
        client = await self._get_client()

        # Serialize complex values to JSON
        payload = {"type": "__metadata__"}
        for key, value in metadata.items():
            if isinstance(value, dict):
                payload[key] = json.dumps(value)
            else:
                payload[key] = value
        payload["updated_at"] = datetime.now(UTC).isoformat()

        if self.embedding_dim == 0:
            raise RuntimeError(
                "embedding_dim not yet initialized. Call embed_batch() first or set "
                "VECTOR_EMBEDDING_DIM before calling store_metadata()."
            )

        point = PointStruct(
            id=0,  # Reserved ID for metadata
            vector={
                "dense": [0.0] * self.embedding_dim,  # Dummy vector
                "sparse": QdrantSparseVector(indices=[], values=[]),
            },
            payload=payload,
        )

        await client.upsert(collection, [point])

    async def get_metadata(self, collection: str) -> dict[str, Any] | None:
        """
        Get collection metadata from reserved point ID=0.

        Args:
            collection: Collection name

        Returns:
            Metadata dict or None if not found.
            String values round-trip unchanged: only dict values are
            JSON-serialized by store_metadata, so only strings that parse
            back to a dict are deserialized here. A string value that merely
            looks like other JSON (e.g. "123", "true") stays a string.
        """
        client = await self._get_client()

        points = await client.retrieve(
            collection,
            ids=[0],
            with_payload=True,
        )

        if not points:
            return None

        payload = dict(points[0].payload) if points[0].payload else {}

        # Deserialize values store_metadata serialized (dicts only)
        for key in list(payload.keys()):
            if isinstance(payload[key], str):
                try:
                    parsed = json.loads(payload[key])
                except (json.JSONDecodeError, TypeError):
                    continue  # Keep as string
                if isinstance(parsed, dict):
                    payload[key] = parsed

        return payload

    async def query_dense(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        filter_conditions: Sequence[FieldCondition] | None = None,
    ) -> list[ScoredPoint]:
        """
        Query using dense vector only.

        Args:
            collection: Collection name
            query_vector: Dense query vector
            limit: Max results
            filter_conditions: Optional filter conditions

        Returns:
            List of scored points
        """
        client = await self._get_client()
        query_filter = Filter(must=list(filter_conditions)) if filter_conditions else None

        response = await client.query_points(
            collection,
            query=query_vector,
            using="dense",
            limit=limit,
            query_filter=query_filter,
        )

        return response.points

    async def query_sparse(
        self,
        collection: str,
        query_vector: SparseVector,
        limit: int = 10,
        filter_conditions: Sequence[FieldCondition] | None = None,
    ) -> list[ScoredPoint]:
        """
        Query using sparse vector only.

        Args:
            collection: Collection name
            query_vector: Sparse query vector
            limit: Max results
            filter_conditions: Optional filter conditions

        Returns:
            List of scored points
        """
        client = await self._get_client()
        query_filter = Filter(must=list(filter_conditions)) if filter_conditions else None

        response = await client.query_points(
            collection,
            query=QdrantSparseVector(
                indices=query_vector.indices,
                values=query_vector.values,
            ),
            using="sparse",
            limit=limit,
            query_filter=query_filter,
        )

        return response.points

    async def get_client(self) -> AsyncQdrantClient:
        """
        Get the underlying Qdrant client for advanced operations.

        Use this when you need direct access to Qdrant API methods
        not exposed through QdrantStorage (e.g., prefetch queries,
        retrieve with vectors).

        Returns:
            AsyncQdrantClient instance
        """
        return await self._get_client()

    async def retrieve_points(
        self,
        collection: str,
        point_ids: list[PointId],
        with_vectors: bool = False,
    ) -> list[Any]:
        """
        Retrieve specific points by ID.

        Args:
            collection: Collection name
            point_ids: List of point IDs to retrieve (int or str)
            with_vectors: Whether to include vectors in response

        Returns:
            List of retrieved points
        """
        client = await self._get_client()
        return await client.retrieve(
            collection,
            ids=point_ids,
            with_payload=True,
            with_vectors=with_vectors,
        )

    def create_point(
        self,
        point_id: PointId,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        payload: dict[str, Any],
    ) -> PointStruct:
        """
        Create a PointStruct for batch operations.

        Args:
            point_id: Point ID (int or str). Use generate_point_id() for safe IDs.
            dense_vector: Dense embedding vector
            sparse_vector: Sparse TF-IDF vector
            payload: Point payload/metadata

        Returns:
            PointStruct ready for upsert
        """
        return PointStruct(
            id=point_id,
            vector={
                "dense": dense_vector,
                "sparse": QdrantSparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values,
                ),
            },
            payload=payload,
        )

    async def ensure_payload_indexes(
        self,
        collection: str,
        indexes: list[tuple[str, PayloadSchemaType]],
    ) -> None:
        """
        Create payload indexes (idempotent).

        Qdrant silently ignores requests to create indexes that already exist,
        so this is safe to call on every startup.

        Args:
            collection: Collection name
            indexes: List of (field_name, schema_type) tuples

        Example:
            await storage.ensure_payload_indexes("my_collection", [
                ("type", PayloadSchemaType.KEYWORD),
                ("confidence", PayloadSchemaType.FLOAT),
            ])
        """
        client = await self._get_client()
        for field_name, schema_type in indexes:
            await client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=schema_type,
                wait=True,
            )

    async def ensure_collection_with_indexes(
        self,
        collection_name: str,
        payload_indexes: list[tuple[str, PayloadSchemaType]] | None = None,
        dense_dim: int | None = None,
    ) -> bool:
        """
        Ensure collection exists with payload indexes.

        Idempotent - safe to call multiple times from multiple servers.
        First caller creates collection and indexes, subsequent calls skip.

        Args:
            collection_name: Collection name
            payload_indexes: Optional list of (field_name, schema_type) tuples
            dense_dim: Dense vector dimension (default from settings)

        Returns:
            True if collection was created, False if it already existed
        """
        created = False
        if not await self.collection_exists(collection_name):
            await self.create_collection(collection_name, dense_dim)
            created = True

        if payload_indexes:
            await self.ensure_payload_indexes(collection_name, payload_indexes)

        return created

    async def __aenter__(self) -> "QdrantStorage":
        """Support async context manager protocol."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Close connection on context exit."""
        await self.close()
