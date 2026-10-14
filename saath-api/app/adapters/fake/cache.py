"""
FakeCacheAdapter — in-memory dict-based cache.

Rule 7: Fully synchronous in-memory implementation for hermetic tests.
"""

from __future__ import annotations

import time


class FakeCacheAdapter:
    """
    In-memory key-value cache backed by a Python dict.

    Supports optional per-key TTL enforcement (based on time.monotonic).
    """

    def __init__(self, max_items: int = 1000) -> None:
        self._store: dict[str, tuple[str, float | None]] = {}
        self._max_items = max_items

    async def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    async def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        if len(self._store) >= self._max_items:
            # Evict the oldest entry (first key in insertion-order dict)
            evict_key = next(iter(self._store))
            del self._store[evict_key]
        expires_at = time.monotonic() + ttl_s if ttl_s is not None else None
        self._store[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        """Test helper — wipe all entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
