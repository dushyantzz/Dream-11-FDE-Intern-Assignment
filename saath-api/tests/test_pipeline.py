"""
Tests for CommentaryPipeline — Rule 4 graceful degradation.
"""

from __future__ import annotations

import pytest

from app.adapters.fake.cache import FakeCacheAdapter
from app.adapters.fake.llm import BrokenLLMAdapter, FakeLLMAdapter
from app.adapters.fake.match_feed import FakeMatchFeedAdapter
from app.adapters.fake.user_context import FakeUserContextAdapter
from app.core.config import Settings
from app.domain.models import MomentEvent, MomentType
from app.engine.pipeline import CommentaryPipeline
from app.llm.gateway import LLMGateway


def _make_pipeline(settings: Settings, llm_adapter: object) -> CommentaryPipeline:
    gateway = LLMGateway(settings=settings, provider=llm_adapter)  # type: ignore[arg-type]
    return CommentaryPipeline(
        settings=settings,
        llm_gateway=gateway,
        match_feed=FakeMatchFeedAdapter(),  # type: ignore[arg-type]
        user_context=FakeUserContextAdapter(),  # type: ignore[arg-type]
        cache=FakeCacheAdapter(),
    )


def _sample_event(moment_type: MomentType = MomentType.BOUNDARY) -> MomentEvent:
    return MomentEvent(
        moment_id="m-test-001",
        match_id="match-001",
        moment_type=moment_type,
        primary_player_id="p001",
        metadata={"runs": 4},
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_pipeline_happy_path(fake_settings: Settings) -> None:
    """Pipeline returns filled commentary on the happy path."""
    fake_llm = FakeLLMAdapter(response="{player_name} hits a glorious shot for {event_runs}!")
    pipeline = _make_pipeline(fake_settings, fake_llm)
    result = await pipeline.generate(event=_sample_event(), user_id="user-001")

    assert result.commentary_text != ""
    assert result.is_fallback is False
    assert result.moment_id == "m-test-001"
    assert result.match_id == "match-001"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_pipeline_falls_back_on_broken_llm(fake_settings: Settings) -> None:
    """Rule 4: BrokenLLM must still produce a non-empty 200 response via fallback."""
    pipeline = _make_pipeline(fake_settings, BrokenLLMAdapter())
    result = await pipeline.generate(event=_sample_event(), user_id="user-001")

    assert result.commentary_text != ""
    assert result.is_fallback is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_pipeline_cache_hit(fake_settings: Settings) -> None:
    """Second call for same match+moment_type uses cached template."""
    fake_llm = FakeLLMAdapter(response="{player_name} hits for {event_runs}!")
    pipeline = _make_pipeline(fake_settings, fake_llm)
    event = _sample_event()

    # First call — populates cache
    result1 = await pipeline.generate(event=event, user_id="user-001")
    assert result1.is_fallback is False

    # Second call — should hit cache; fake_llm call count stays at 1
    result2 = await pipeline.generate(event=event, user_id="user-001")
    assert result2.commentary_text != ""
    assert fake_llm.call_count == 1  # no second LLM call


@pytest.mark.asyncio
@pytest.mark.unit
async def test_pipeline_fallback_for_wicket(fake_settings: Settings) -> None:
    """Fallback works for all moment types, including WICKET."""
    pipeline = _make_pipeline(fake_settings, BrokenLLMAdapter())
    event = _sample_event(MomentType.WICKET)
    result = await pipeline.generate(event=event, user_id="user-001")
    assert result.is_fallback is True
    assert result.commentary_text != ""
