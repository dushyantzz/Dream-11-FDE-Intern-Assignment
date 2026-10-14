"""
Commentary pipeline — Rule 4 enforcement.

Orchestrates the full commentary generation flow with a three-tier
degradation ladder:
  1. LLM path  → slot-verified, slot-filled LLM template
  2. Cache hit → previously generated template for same moment_type
  3. Fallback  → deterministic FallbackRenderer (always succeeds)

No business logic yet — this is the scaffold that all future logic builds on.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from app.adapters.ports import CachePort, MatchFeedPort, UserContextPort
from app.core.config import Settings
from app.domain.models import (
    CommentaryResponse,
    MomentEvent,
    SlotMap,
)
from app.engine.fallback import FallbackRenderer
from app.engine.verifier import SlotVerificationError, verify_and_render
from app.llm.gateway import BudgetExceededError, LLMGateway, LLMTimeoutError

logger = structlog.get_logger(__name__)


def _build_slot_map(event: MomentEvent, match_state: Any, user_ctx: Any) -> SlotMap:
    """
    Build a SlotMap from verified domain objects.

    TODO: Populate with real fields as the domain model grows.
    """
    innings = match_state.innings[0] if match_state.innings else None

    slots: dict[str, str | int | float] = {
        "match_id": event.match_id,
        "moment_type": str(event.moment_type),
        "home_team": match_state.home_team,
        "away_team": match_state.away_team,
        "venue": match_state.venue,
    }

    if innings:
        slots.update(
            {
                "batting_team": innings.batting_team,
                "bowling_team": innings.bowling_team,
                "runs": innings.total_runs,
                "wickets": innings.wickets_fallen,
                "overs": innings.overs_completed,
            }
        )

    if event.primary_player_id and innings:
        player = innings.player_stats.get(event.primary_player_id)
        if player:
            slots["player_name"] = player.player_name
            slots["player_runs"] = player.runs_scored
            slots["player_balls"] = player.balls_faced
            slots["player_wickets"] = player.wickets_taken
            slots["fantasy_points"] = player.fantasy_points

    # Event-level metadata
    for k, v in event.metadata.items():
        if isinstance(v, (str, int, float)):
            slots[f"event_{k}"] = v

    return SlotMap(slots=slots)


def _build_prompt(event: MomentEvent, slot_map: SlotMap, user_ctx: Any) -> str:
    """
    Build the LLM prompt for a commentary moment.

    Slots in the prompt are passed as their token names (e.g. {runs}),
    instructing the LLM to preserve them verbatim in its output.
    Rule 1: The prompt explicitly forbids raw numeric output.
    """
    slot_hint = ", ".join(f"{{{k}}}" for k in slot_map.keys())
    return (
        f"You are a cricket commentary AI. Generate exactly ONE sentence of vivid commentary "
        f"for a '{event.moment_type}' moment.\n\n"
        f"RULES:\n"
        f"- Do NOT emit any numeric values, player names, or team names directly.\n"
        f"- Use ONLY these slot tokens where data is needed: {slot_hint}\n"
        f"- Return ONLY the commentary sentence — no explanations, no extra text.\n\n"
        f"Moment type: {event.moment_type}\n"
        f"User preference: {getattr(user_ctx, 'tone_preference', 'enthusiastic')}\n"
    )


class CommentaryPipeline:
    """
    Orchestrates commentary generation with graceful degradation (Rule 4).

    Inject adapters via the constructor so the pipeline is testable offline.
    """

    def __init__(
        self,
        settings: Settings,
        llm_gateway: LLMGateway,
        match_feed: MatchFeedPort,
        user_context: UserContextPort,
        cache: CachePort,
    ) -> None:
        self._settings = settings
        self._gateway = llm_gateway
        self._match_feed = match_feed
        self._user_context = user_context
        self._cache = cache
        self._fallback = FallbackRenderer(template_path=settings.engine.fallback_template_path)

    async def generate(
        self,
        event: MomentEvent,
        user_id: str,
    ) -> CommentaryResponse:
        """
        Generate commentary for ``event``, personalised for ``user_id``.

        Guaranteed to return a valid CommentaryResponse (Rule 4).
        """
        start_ms = time.monotonic() * 1000
        is_fallback = False
        model_used: str | None = None
        cost_usd = 0.0

        try:
            # 1. Fetch domain data
            match_state = await self._match_feed.get_match_state(event.match_id)
            user_ctx = await self._user_context.get_user_context(user_id)
            slot_map = _build_slot_map(event, match_state, user_ctx)

            # 2. Check cache
            cache_key = f"commentary:{event.match_id}:{event.moment_type}"
            cached = await self._cache.get(cache_key)
            if cached:
                logger.info("commentary_cache_hit", cache_key=cache_key)
                commentary_text = verify_and_render(
                    cached, slot_map, strict=self._settings.verifier.strict
                )
                latency_ms = time.monotonic() * 1000 - start_ms
                return CommentaryResponse(
                    moment_id=event.moment_id,
                    match_id=event.match_id,
                    commentary_text=commentary_text,
                    moment_type=event.moment_type,
                    is_fallback=False,
                    latency_ms=round(latency_ms, 1),
                )

            # 3. LLM path
            prompt = _build_prompt(event, slot_map, user_ctx)
            raw_template, meta = await self._gateway.complete(prompt)
            model_used = meta.get("model")
            cost_usd = meta.get("cost_usd", 0.0)

            # 4. Verify and render
            commentary_text = verify_and_render(
                raw_template, slot_map, strict=self._settings.verifier.strict
            )

            # 5. Cache the raw template (not the rendered string)
            await self._cache.set(cache_key, raw_template, ttl_s=self._settings.cache.ttl_s)

        except (
            BudgetExceededError,
            LLMTimeoutError,
            SlotVerificationError,
            Exception,
        ) as exc:
            logger.error(
                "commentary_pipeline_error",
                error_type=type(exc).__name__,
                error=str(exc),
                moment_id=event.moment_id,
                match_id=event.match_id,
            )
            is_fallback = True
            # Build a minimal slot_map for fallback (may already be set above)
            try:
                slot_map  # noqa: B018
            except UnboundLocalError:
                slot_map = SlotMap(
                    slots={
                        "match_id": event.match_id,
                        "moment_type": str(event.moment_type),
                    }
                )
            commentary_text = self._fallback.render(event.moment_type, slot_map)

        latency_ms = time.monotonic() * 1000 - start_ms
        logger.info(
            "commentary_generated",
            moment_id=event.moment_id,
            match_id=event.match_id,
            is_fallback=is_fallback,
            latency_ms=round(latency_ms, 1),
            cost_usd=cost_usd,
        )

        return CommentaryResponse(
            moment_id=event.moment_id,
            match_id=event.match_id,
            commentary_text=commentary_text,
            moment_type=event.moment_type,
            is_fallback=is_fallback,
            model_used=model_used,
            latency_ms=round(latency_ms, 1),
            cost_usd=cost_usd,
        )
