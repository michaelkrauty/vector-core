"""Regression tests for lock pathname inode stability."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

from vector_core.utils import locking


def test_sync_waiter_cannot_overlap_new_arrival(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "resource"
    waiter_opened = threading.Event()
    allow_waiter = threading.Event()
    waiter_entered = threading.Event()
    arrival_entered = threading.Event()
    release_waiter = threading.Event()
    release_arrival = threading.Event()
    real_open = locking.os.open

    def controlled_open(path: str, flags: int, *args: int) -> int:
        fd = real_open(path, flags, *args)
        if threading.current_thread().name == "inode-waiter":
            waiter_opened.set()
            allow_waiter.wait()
        return fd

    monkeypatch.setattr(locking.os, "open", controlled_open)

    first = locking.file_lock(target, timeout=2.0)
    first.__enter__()

    def wait_on_old_inode() -> None:
        with locking.file_lock(target, timeout=2.0):
            waiter_entered.set()
            release_waiter.wait()

    def new_arrival() -> None:
        with locking.file_lock(target, timeout=2.0):
            arrival_entered.set()
            release_arrival.wait()

    waiter = threading.Thread(target=wait_on_old_inode, name="inode-waiter")
    arrival = threading.Thread(target=new_arrival, name="inode-arrival")
    waiter.start()
    assert waiter_opened.wait(1.0)

    first.__exit__(None, None, None)
    arrival.start()
    assert arrival_entered.wait(1.0)

    try:
        allow_waiter.set()
        time.sleep(0.15)
        assert not waiter_entered.is_set()
    finally:
        release_arrival.set()
        assert waiter_entered.wait(1.0)
        release_waiter.set()
        arrival.join(timeout=1.0)
        waiter.join(timeout=1.0)

    assert not arrival.is_alive()
    assert not waiter.is_alive()
    assert target.with_suffix(".lock").exists()


async def test_async_waiter_cannot_overlap_new_arrival(monkeypatch, tmp_path: Path) -> None:
    lock_dir = tmp_path / "locks"
    waiter_opened = threading.Event()
    allow_waiter = threading.Event()
    waiter_entered = asyncio.Event()
    arrival_entered = asyncio.Event()
    release_waiter = asyncio.Event()
    release_arrival = asyncio.Event()
    open_count = 0
    count_lock = threading.Lock()
    real_open = locking.os.open

    def controlled_open(path: str, flags: int, *args: int) -> int:
        nonlocal open_count
        fd = real_open(path, flags, *args)
        with count_lock:
            open_count += 1
            current = open_count
        if current == 2:
            waiter_opened.set()
            allow_waiter.wait()
        return fd

    monkeypatch.setattr(locking.os, "open", controlled_open)

    first = locking.async_file_lock("resource", timeout=2.0, lock_dir=lock_dir)
    await first.__aenter__()

    async def wait_on_old_inode() -> None:
        async with locking.async_file_lock("resource", timeout=2.0, lock_dir=lock_dir):
            waiter_entered.set()
            await release_waiter.wait()

    async def new_arrival() -> None:
        async with locking.async_file_lock("resource", timeout=2.0, lock_dir=lock_dir):
            arrival_entered.set()
            await release_arrival.wait()

    waiter = asyncio.create_task(wait_on_old_inode())
    assert await asyncio.to_thread(waiter_opened.wait, 1.0)

    await first.__aexit__(None, None, None)
    arrival = asyncio.create_task(new_arrival())
    await asyncio.wait_for(arrival_entered.wait(), timeout=1.0)

    try:
        allow_waiter.set()
        await asyncio.sleep(0.15)
        assert not waiter_entered.is_set()
    finally:
        release_arrival.set()
        await asyncio.wait_for(waiter_entered.wait(), timeout=1.0)
        release_waiter.set()
        await asyncio.gather(arrival, waiter)

    assert (lock_dir / "resource.lock").exists()
