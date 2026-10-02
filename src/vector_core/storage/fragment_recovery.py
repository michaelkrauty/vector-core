"""Compensate failed fragment replacements; retain recovery evidence if compensation fails."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from qdrant_client.models import PointIdsList, PointStruct, WriteOrdering

from vector_core.settings import settings

if TYPE_CHECKING:
    from vector_core.storage.qdrant import QdrantStorage

logger = logging.getLogger(__name__)


class FragmentGroupRecoveryError(RuntimeError):
    """Replacement and rollback both failed; the prior group remains in a local journal."""

    def __init__(self, path: Path, original: BaseException, rollback: BaseException):
        self.recovery_path = path
        self.original_error = original
        self.rollback_error = rollback
        super().__init__(
            f"Fragment replacement failed ({type(original).__name__}: {original}); "
            f"rollback did not complete ({type(rollback).__name__}: {rollback}). "
            f"Prior source payloads and vectors are retained at {path}. "
            "Inspect current collection state before recovery; do not replay against later edits."
        )


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_journal(
    scope: str,
    collection: str,
    parent_id: Any,
    previous: list[PointStruct],
    introduced: list[Any],
) -> Path:
    directory = settings.cache_dir.resolve() / "embedding-fragment-recovery"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    _sync_directory(directory.parent)
    path = directory / f"{uuid4().hex}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema": 1,
                    "storage_scope": scope,
                    "collection": collection,
                    "parent_id": str(parent_id) if isinstance(parent_id, UUID) else parent_id,
                    "introduced_ids": [
                        str(value) if isinstance(value, UUID) else value for value in introduced
                    ],
                    "previous_points": [point.model_dump(mode="json") for point in previous],
                },
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(directory)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _remove_journal(path: Path) -> None:
    try:
        path.unlink()
        _sync_directory(path.parent)
    except OSError:
        # The Qdrant operation is already complete. A stale journal must never
        # be replayed automatically, including after a cleanup-only failure.
        logger.warning(
            "Completed fragment operation left an obsolete recovery journal: %s",
            path,
            exc_info=True,
        )


async def _settle(task: asyncio.Task[Any]) -> bool:
    """Finish in-flight workers or mutations before releasing the caller's lock."""
    cancelled = False
    while True:
        try:
            await asyncio.shield(task)
            return cancelled
        except asyncio.CancelledError:
            cancelled = True
            if task.done():
                task.result()
                return cancelled


async def _delete_ids(client: Any, collection: str, point_ids: list[Any]) -> None:
    from vector_core.storage.embedding_migration import _MAX_UPSERT_BYTES  # noqa: PLC0415

    start = 0
    while start < len(point_ids):
        end = min(start + 128, len(point_ids))
        while True:
            selector = PointIdsList(points=point_ids[start:end])
            size = len(selector.model_dump_json(exclude_none=True, exclude_unset=True).encode())
            if size <= _MAX_UPSERT_BYTES:
                break
            if end == start + 1:
                raise ValueError("Fragment point ID exceeds the request byte budget")
            end = start + (end - start) // 2
        await client.delete(
            collection, points_selector=selector, wait=True, ordering=WriteOrdering.STRONG
        )
        start = end


async def replace_fragment_group(
    storage: QdrantStorage,
    collection: str,
    points: list[PointStruct],
    previous: list[PointStruct],
) -> None:
    """Compensate failures under the caller's writer lock, not a reader transaction.

    The journal survives rollback failure or process death. Recovery is manual:
    another successful edit can make an old snapshot unsafe to restore.
    """
    from vector_core.storage.embedding_migration import (  # noqa: PLC0415
        _storage_scope,
        _upsert_copy_points,
    )

    client = await storage.get_client()
    previous_ids = {point.id for point in previous}
    current_ids = {point.id for point in points}
    introduced = [point.id for point in points if point.id not in previous_ids]
    stale = [point.id for point in previous if point.id not in current_ids]
    journal_task = asyncio.create_task(
        asyncio.to_thread(
            _save_journal,
            _storage_scope(storage.url),
            collection,
            points[0].id,
            previous,
            introduced,
        )
    )
    cancelled = await _settle(journal_task)
    journal = journal_task.result()

    async def remove_journal() -> bool:
        return await _settle(asyncio.create_task(asyncio.to_thread(_remove_journal, journal)))

    if cancelled:
        await remove_journal()
        raise asyncio.CancelledError

    async def replace() -> None:
        await _upsert_copy_points(client, collection, points)
        await _delete_ids(client, collection, stale)

    async def rollback() -> None:
        await _upsert_copy_points(client, collection, previous)
        await _delete_ids(client, collection, introduced)

    try:
        if await _settle(asyncio.create_task(replace())):
            raise asyncio.CancelledError
    except BaseException as original:
        try:
            await _settle(asyncio.create_task(rollback()))
        except BaseException as failure:
            raise FragmentGroupRecoveryError(journal, original, failure) from failure
        await remove_journal()
        raise
    # Replacement has committed. Finish cleanup even if cancellation arrives;
    # never leave a filesystem worker running after releasing the writer lock.
    if await remove_journal():
        raise asyncio.CancelledError
