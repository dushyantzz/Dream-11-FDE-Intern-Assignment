"""
Pytest configuration and shared fixtures — Rule 7 enforcement.

All tests default to using fake adapters.
No real LLM / network calls happen in `make test`.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.adapters.fake.cache import FakeCacheAdapter
from app.adapters.fake.llm import BrokenLLMAdapter, FakeLLMAdapter
from app.adapters.fake.match_feed import FakeMatchFeedAdapter
from app.adapters.fake.user_context import FakeUserContextAdapter
from app.core.config import Settings
from app.llm.gateway import LLMGateway


@pytest.fixture
def fake_settings() -> Settings:
    """Minimal valid Settings using only config.yaml defaults + fake backends."""
    return Settings(
        llm={"provider": "fake", "model": "fake-model", "budget_usd": 1.0},
        cache={"backend": "memory"},
        match_feed={"backend": "fake"},
        user_context={"backend": "fake"},
        logging={"format": "console", "level": "DEBUG"},
    )


@pytest.fixture
def fake_llm() -> FakeLLMAdapter:
    return FakeLLMAdapter()


@pytest.fixture
def broken_llm() -> BrokenLLMAdapter:
    return BrokenLLMAdapter()


@pytest.fixture
def fake_cache() -> FakeCacheAdapter:
    return FakeCacheAdapter()


@pytest.fixture
def fake_match_feed() -> FakeMatchFeedAdapter:
    return FakeMatchFeedAdapter()


@pytest.fixture
def fake_user_context() -> FakeUserContextAdapter:
    return FakeUserContextAdapter()


@pytest.fixture
def fake_gateway(fake_settings: Settings, fake_llm: FakeLLMAdapter) -> LLMGateway:
    return LLMGateway(settings=fake_settings, provider=fake_llm)


@pytest.fixture
def broken_gateway(fake_settings: Settings, broken_llm: BrokenLLMAdapter) -> LLMGateway:
    return LLMGateway(settings=fake_settings, provider=broken_llm)


@pytest.fixture
def test_client(fake_settings: Settings) -> TestClient:
    """ASGI test client with fake container wired in."""
    from unittest.mock import MagicMock

    from app.adapters.fake.cache import FakeCacheAdapter
    from app.adapters.fake.match_feed import FakeMatchFeedAdapter
    from app.adapters.fake.user_context import FakeUserContextAdapter
    from app.core.container import AppContainer
    from app.main import create_app

    application = create_app()

    # Override the container with all-fake adapters
    container = MagicMock(spec=AppContainer)
    container.cache = FakeCacheAdapter()
    container.match_feed = FakeMatchFeedAdapter()
    container.user_context = FakeUserContextAdapter()
    container.llm_gateway = LLMGateway(
        settings=fake_settings,
        provider=FakeLLMAdapter(),
    )
    application.state.container = container

    return TestClient(application, raise_server_exceptions=False)
