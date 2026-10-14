"""
Dependency injection container — Rule 3 enforcement.

This is the ONLY place where concrete adapter implementations are instantiated.
Engine code receives ports (interfaces), never concrete classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.config import Settings

if TYPE_CHECKING:
    from app.adapters.ports import CachePort, MatchFeedPort, UserContextPort
    from app.llm.gateway import LLMGateway


class AppContainer:
    """
    Wires together all ports and their concrete adapter implementations.

    Each property is lazy-initialized and cached on first access so that
    the container can be constructed cheaply (e.g., in tests).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache_adapter: CachePort | None = None
        self._match_feed_adapter: MatchFeedPort | None = None
        self._user_context_adapter: UserContextPort | None = None
        self._llm_gateway: LLMGateway | None = None

    # ------------------------------------------------------------------
    # Cache port
    # ------------------------------------------------------------------
    @property
    def cache(self) -> CachePort:
        if self._cache_adapter is None:
            backend = self._settings.cache.backend
            if backend == "memory":
                from app.adapters.fake.cache import FakeCacheAdapter

                self._cache_adapter = FakeCacheAdapter(
                    max_items=self._settings.cache.max_memory_items
                )
            else:
                raise ValueError(f"Unknown cache backend: {backend!r}")
        return self._cache_adapter

    # ------------------------------------------------------------------
    # Match feed port
    # ------------------------------------------------------------------
    @property
    def match_feed(self) -> MatchFeedPort:
        if self._match_feed_adapter is None:
            backend = self._settings.match_feed.backend
            if backend == "fake":
                from app.adapters.fake.match_feed import FakeMatchFeedAdapter

                self._match_feed_adapter = FakeMatchFeedAdapter()
            else:
                raise ValueError(f"Unknown match_feed backend: {backend!r}")
        return self._match_feed_adapter

    # ------------------------------------------------------------------
    # User context port
    # ------------------------------------------------------------------
    @property
    def user_context(self) -> UserContextPort:
        if self._user_context_adapter is None:
            backend = self._settings.user_context.backend
            if backend == "fake":
                from app.adapters.fake.user_context import FakeUserContextAdapter

                self._user_context_adapter = FakeUserContextAdapter()
            else:
                raise ValueError(f"Unknown user_context backend: {backend!r}")
        return self._user_context_adapter

    # ------------------------------------------------------------------
    # LLM gateway (always one; provider selected inside gateway)
    # ------------------------------------------------------------------
    @property
    def llm_gateway(self) -> LLMGateway:
        if self._llm_gateway is None:
            from app.llm.gateway import LLMGateway

            self._llm_gateway = LLMGateway(settings=self._settings)
        return self._llm_gateway
