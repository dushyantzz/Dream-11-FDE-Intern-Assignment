"""
Adapter base interfaces — Rule 3 enforcement.

Defined as typing.Protocol so that any implementation can be used without
rigid inheritance hierarchies.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from app.domain.models import Ball, CostRecord, MatchState, Roster, UserContext


@runtime_checkable
class MatchFeed(Protocol):
    """
    Feed providing an async stream of Ball deliveries,
    along with team rosters, match metadata, and running match state.
    """

    def __aiter__(self) -> AsyncIterator[Ball]:
        """Iterate through deliveries asynchronously."""
        ...

    @property
    def roster(self) -> dict[str, Roster]:
        """Team rosters keyed by team name."""
        ...

    @property
    def metadata(self) -> dict[str, Any]:
        """Match metadata (teams, format, date, venue, event, etc.)."""
        ...

    @property
    def current_state(self) -> MatchState:
        """Running MatchState snapshot."""
        ...


@runtime_checkable
class UserContextProvider(Protocol):
    """Source of user preferences, fantasy picks, and tone/language choices."""

    async def get_user_context(self, user_id: str) -> UserContext:
        """Fetch user context with fantasy picks and preferred persona/language."""
        ...


@runtime_checkable
class Cache(Protocol):
    """Generic key-value cache supporting time-to-live expiration."""

    async def get(self, key: str) -> str | None:
        """Fetch cached string value or None if expired or missing."""
        ...

    async def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        """Store value with an optional TTL in seconds."""
        ...


@runtime_checkable
class TraceStore(Protocol):
    """Observability and billing store for LLM cost/latency audit records."""

    async def record_cost(self, record: CostRecord) -> None:
        """Record an LLM cost and latency record."""
        ...
