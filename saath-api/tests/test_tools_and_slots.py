"""
Tests for app/engine/tools.py and app/engine/slots.py.

Verifies:
1. Pure functions in tools.py return typed Pydantic models or None.
2. build_slots builds a typed slot map with provenance.
3. None tool result removes that slot from the allowlist rather than inventing a value.
4. render with an unknown token raises SlotRenderError.
5. Definition of done: build_slots succeeds for every moment in a full match replay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from app.adapters.cricsheet import CricsheetReplayFeed
from app.domain.models import Ball, MatchFormat, MatchState, MomentEvent, MomentType
from app.engine.moments import MomentEngine
from app.engine.slots import SlotRenderError, build_slots, get_slot_allowlist, render
from app.engine.tools import (
    HeadToHeadSummary,
    LastNOversSummary,
    PartnershipSummary,
    PlayerSummary,
    RequiredRateSummary,
    WinProbSummary,
    batter_vs_bowler,
    last_n_overs,
    partnership,
    player_summary,
    required_rate,
    win_prob_now,
)

SAMPLE_MATCH_PATH = Path("../odis_json/919615.json")


# ---------------------------------------------------------------------------
# Unit tests for tools.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_player_summary_batting_and_bowling() -> None:
    """player_summary aggregates runs, balls, strike rate, wickets, and economy."""
    balls = [
        Ball(
            ball_id="b1",
            match_id="m1",
            innings=1,
            over=0,
            ball_in_over=1,
            bowler="Bowler A",
            batter="Batter A",
            non_striker="Batter B",
            runs_batter=4,
            runs_extras=0,
            runs_total=4,
            is_legal=True,
        ),
        Ball(
            ball_id="b2",
            match_id="m1",
            innings=1,
            over=0,
            ball_in_over=2,
            bowler="Bowler A",
            batter="Batter A",
            non_striker="Batter B",
            runs_batter=6,
            runs_extras=0,
            runs_total=6,
            is_legal=True,
        ),
    ]

    summary_bat = player_summary(balls, "Batter A")
    assert isinstance(summary_bat, PlayerSummary)
    assert summary_bat.runs == 10
    assert summary_bat.balls_faced == 2
    assert summary_bat.strike_rate == 500.0
    assert summary_bat.fours == 1
    assert summary_bat.sixes == 1

    summary_bowl = player_summary(balls, "Bowler A")
    assert isinstance(summary_bowl, PlayerSummary)
    assert summary_bowl.runs_conceded == 10
    assert summary_bowl.overs_bowled == 0.2

    # Uninvolved player returns None
    assert player_summary(balls, "Ghost Player") is None


@pytest.mark.unit
def test_batter_vs_bowler_head_to_head() -> None:
    """batter_vs_bowler extracts exact match-up data or None if unplayed."""
    balls = [
        Ball(
            ball_id="b1",
            match_id="m1",
            innings=1,
            over=1,
            ball_in_over=1,
            bowler="Bumrah",
            batter="Warner",
            non_striker="Smith",
            runs_batter=0,
            runs_extras=0,
            runs_total=0,
            is_legal=True,
        ),
        Ball(
            ball_id="b2",
            match_id="m1",
            innings=1,
            over=1,
            ball_in_over=2,
            bowler="Bumrah",
            batter="Warner",
            non_striker="Smith",
            runs_batter=0,
            runs_extras=0,
            runs_total=0,
            is_legal=True,
            is_wicket=True,
            wicket_kind="bowled",
            player_dismissed="Warner",
        ),
    ]

    h2h = batter_vs_bowler(balls, "Warner", "Bumrah")
    assert isinstance(h2h, HeadToHeadSummary)
    assert h2h.balls_faced == 2
    assert h2h.runs_scored == 0
    assert h2h.dismissals == 1
    assert h2h.dots == 2

    # Unplayed pair returns None
    assert batter_vs_bowler(balls, "Smith", "Bumrah") is None


@pytest.mark.unit
def test_partnership_and_wicket_reset() -> None:
    """partnership tracks active pair and resets on a wicket."""
    st = MatchState(match_id="m1", innings_idx=1)
    balls = [
        Ball(
            ball_id="b1",
            match_id="m1",
            innings=1,
            over=0,
            ball_in_over=1,
            bowler="Bowler",
            batter="Player 1",
            non_striker="Player 2",
            runs_batter=4,
            runs_extras=0,
            runs_total=4,
            is_legal=True,
        ),
        Ball(
            ball_id="b2",
            match_id="m1",
            innings=1,
            over=0,
            ball_in_over=2,
            bowler="Bowler",
            batter="Player 1",
            non_striker="Player 2",
            runs_batter=0,
            runs_extras=0,
            runs_total=0,
            is_legal=True,
            is_wicket=True,
            player_dismissed="Player 1",
        ),
        Ball(
            ball_id="b3",
            match_id="m1",
            innings=1,
            over=0,
            ball_in_over=3,
            bowler="Bowler",
            batter="Player 3",
            non_striker="Player 2",
            runs_batter=2,
            runs_extras=0,
            runs_total=2,
            is_legal=True,
        ),
    ]

    part = partnership(balls, st)
    assert isinstance(part, PartnershipSummary)
    # Partnership is only since the wicket on ball 2
    assert part.runs == 2
    assert part.balls == 1


@pytest.mark.unit
def test_required_rate_tool() -> None:
    """required_rate returns None for innings 1 and typed summary for innings 2."""
    st_inn1 = MatchState(match_id="m1", innings_idx=1)
    assert required_rate(st_inn1) is None

    st_inn2 = MatchState(
        match_id="m1",
        innings_idx=2,
        score=200,
        target=250,
        balls_left=60,
        current_run_rate=5.0,
    )
    req = required_rate(st_inn2)
    assert isinstance(req, RequiredRateSummary)
    assert req.target == 250
    assert req.runs_needed == 50
    assert req.balls_left == 60
    assert req.overs_left == 10.0
    assert req.required_rate == 5.0
    assert req.rate_difference == 0.0


@pytest.mark.unit
def test_win_prob_now_and_last_n_overs() -> None:
    """win_prob_now and last_n_overs return valid typed summaries."""
    st = MatchState(
        match_id="m1",
        innings_idx=1,
        batting_team="India",
        bowling_team="Australia",
        score=100,
        balls_left=150,
    )
    wp = win_prob_now(st, swing=0.08)
    assert isinstance(wp, WinProbSummary)
    assert 0.0 <= wp.win_prob_batting <= 1.0
    assert wp.swing == 0.08

    balls = [
        Ball(
            ball_id=f"b{i}",
            match_id="m1",
            innings=1,
            over=i // 6,
            ball_in_over=(i % 6) + 1,
            bowler="Bowler",
            batter="Batter",
            non_striker="NonStriker",
            runs_batter=1,
            runs_extras=0,
            runs_total=1,
            is_legal=True,
        )
        for i in range(12)  # 2 overs
    ]
    last_overs = last_n_overs(balls, st, n_overs=5)
    assert isinstance(last_overs, LastNOversSummary)
    assert last_overs.runs == 12
    assert last_overs.legal_balls == 12


# ---------------------------------------------------------------------------
# Unit tests for slots.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_render_with_unknown_token_raises() -> None:
    """render must raise SlotRenderError on unknown tokens or missing values."""
    ev = MomentEvent(
        moment_id="ev1",
        match_id="m1",
        moment_type=MomentType.BOUNDARY,
        metadata={"verified_facts": {"batter": "Rohit Sharma", "score": 150}},
    )
    slots = build_slots(ev, {})

    # Happy path
    template_ok = "{batter} has pushed the team to {score}."
    rendered = render(template_ok, slots)
    assert rendered == "Rohit Sharma has pushed the team to 150."

    # Unknown token raises
    template_bad = "{batter} smashed {unknown_shot_type} for four!"
    with pytest.raises(SlotRenderError) as exc_info:
        render(template_bad, slots)
    assert "unknown_shot_type" in exc_info.value.missing_token


@pytest.mark.unit
def test_none_tool_result_removes_slot_from_allowlist() -> None:
    """A tool returning None MUST omit those slots from allowlist, never inventing values."""
    ev = MomentEvent(
        moment_id="ev1",
        match_id="m1",
        moment_type=MomentType.GENERIC,
        metadata={"verified_facts": {"match_id": "m1", "score": 120}},
    )

    # Tool required_rate returned None (innings 1)
    tools_results: dict[str, BaseModel | None] = {
        "required_rate": None,
        "player_summary": PlayerSummary(player_name="Kohli", runs=45),
    }

    slots = build_slots(ev, tools_results)
    allowlist = get_slot_allowlist(slots)

    # Required rate slots must NOT exist
    assert "required_rate" not in allowlist
    assert "required_rate_runs_needed" not in allowlist
    assert "runs_needed" not in allowlist

    # Available tool slots must exist
    assert "player_summary_runs" in allowlist
    assert slots["player_summary_runs"].value == 45
    assert slots["player_summary_runs"].source == "tool:player_summary.runs"

    # Attempting to render a missing tool slot raises
    with pytest.raises(SlotRenderError):
        render("Chasing {runs_needed} to win", slots)


# ---------------------------------------------------------------------------
# Definition of Done Test: Full match replay
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_definition_of_done_build_slots_for_every_moment() -> None:
    """
    Definition of Done:
    For every single delivery in the full match replay, tools and build_slots
    must succeed without raising any exceptions.
    """
    assert SAMPLE_MATCH_PATH.exists()

    feed = CricsheetReplayFeed(SAMPLE_MATCH_PATH, speed_multiplier=1_000_000.0)
    engine = MomentEngine()

    balls_history: list[Ball] = []
    state_prev = feed.current_state

    moment_count = 0

    async for ball in feed:
        balls_history.append(ball)
        state_curr = feed.current_state

        event, _ = engine.process_ball(ball, state_prev, state_curr)

        # Run all 6 pure tools
        tool_results: dict[str, BaseModel | None] = {
            "player_summary": player_summary(balls_history, ball.batter),
            "batter_vs_bowler": batter_vs_bowler(balls_history, ball.batter, ball.bowler),
            "partnership": partnership(balls_history, state_curr),
            "required_rate": required_rate(state_curr),
            "win_prob_now": win_prob_now(state_curr, swing=event.metadata.get("swing", 0.0)),
            "last_n_overs": last_n_overs(balls_history, state_curr, n_overs=5),
        }

        # Build slots and allowlist
        slots = build_slots(event, tool_results)
        allowlist = get_slot_allowlist(slots)

        assert len(slots) > 0, f"Slots empty on ball {ball.ball_id}"
        assert len(allowlist) == len(slots)
        assert "score" in slots
        assert "batter" in slots
        assert "bowler" in slots

        # Verify safe rendering with verified slots
        test_template = "{batter} on {score}/{wickets}"
        rendered = render(test_template, slots)
        assert str(state_curr.score) in rendered

        moment_count += 1
        state_prev = state_curr

    assert moment_count == 606
