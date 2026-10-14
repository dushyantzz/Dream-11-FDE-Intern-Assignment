"""
Port interfaces — Rule 3 enforcement.

All external system interactions are defined here as Protocols.
Engine code depends only on these interfaces, never on concrete implementations.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.domain.models import MatchState, MomentEvent, UserContext


@runtime_checkable
class MatchFeedPort(Protocol):
    """Source of live match data."""

    async def get_match_state(self, match_id: str) -> MatchState:
        """Fetch current match state by match_id."""
        ...

    async def get_recent_events(self, match_id: str, limit: int) -> list[MomentEvent]:
        """Fetch the most recent match moment events."""
        ...


@runtime_checkable
class UserContextPort(Protocol):
    """Source of user-specific context for personalisation."""

    async def get_user_context(self, user_id: str) -> UserContext:
        """Fetch user preferences and fantasy team state."""
        ...


@runtime_checkable
class CachePort(Protocol):
    """Generic key-value cache port."""

    async def get(self, key: str) -> str | None:
        """Return cached value or None if not found / expired."""
        ...

    async def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        """Store value; optionally override the default TTL."""
        ...

    async def delete(self, key: str) -> None:
        """Evict a key."""
        ...


@runtime_checkable
class LLMPort(Protocol):
    """
    Low-level LLM provider interface — consumed only by app/llm/gateway.py.

    Rule 2: No other module may reference this protocol directly.
    """

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Send a prompt and return the raw completion string.

        The returned string MUST only contain slot tokens for data fields,
        never raw numeric facts (enforced upstream by the gateway verifier).
        """
        ...

    def estimate_cost_usd(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> float:
        """Compute estimated USD cost for a given token pair."""
        ...


@runtime_checkable
class FeatureStorePort(Protocol):
    """Source of pre-computed ML features (e.g. player form vectors)."""

    async def get_features(self, entity_id: str, feature_names: list[str]) -> dict[str, Any]:
        """Fetch named features for an entity (player/team)."""
        ...
