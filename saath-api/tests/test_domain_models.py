"""
Tests for app/domain/models.py — frozen Pydantic v2 models.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.models import (
    Ball,
    CostRecord,
    MatchState,
    MomentEvent,
    MomentType,
    Reaction,
    Roster,
    Slot,
    UserContext,
)


@pytest.mark.unit
def test_all_models_are_frozen() -> None:
    """All domain models must be immutable (frozen=True)."""
    ball = Ball(
        ball_id="b1",
        match_id="m1",
        innings=1,
        over=0,
        ball_in_over=1,
        bowler="Bowler",
        batter="Batter",
        non_striker="NonStriker",
        runs_batter=4,
        runs_extras=0,
        runs_total=4,
        is_legal=True,
    )
    with pytest.raises(ValidationError):
        ball.runs_total = 6  # type: ignore[misc]

    roster = Roster(team_name="India", players=["Rohit", "Virat"])
    with pytest.raises(ValidationError):
        roster.team_name = "Bharat"  # type: ignore[misc]

    user = UserContext(
        user_id="u1",
        fantasy_picks=["Rohit"],
        preferred_persona="analyst",
        preferred_language="en",
    )
    with pytest.raises(ValidationError):
        user.preferred_persona = "roaster"  # type: ignore[misc]

    cost = CostRecord(
        request_id="r1",
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=50,
        cost_usd=0.002,
        latency_ms=150.0,
    )
    with pytest.raises(ValidationError):
        cost.cost_usd = 0.005  # type: ignore[misc]

    state = MatchState(match_id="m1", score=100, wickets=2)
    with pytest.raises(ValidationError):
        state.score = 101  # type: ignore[misc]

    slot = Slot(name="runs", value=100)
    with pytest.raises(ValidationError):
        slot.value = 105  # type: ignore[misc]

    reaction = Reaction(
        user_id="u1",
        moment_id="ev1",
        commentary_text="What a shot!",
    )
    with pytest.raises(ValidationError):
        reaction.persona = "roaster"  # type: ignore[misc]

    event = MomentEvent(
        moment_id="ev1",
        match_id="m1",
        moment_type=MomentType.BOUNDARY,
    )
    with pytest.raises(ValidationError):
        event.moment_id = "ev2"  # type: ignore[misc]


@pytest.mark.unit
def test_match_state_running_fields() -> None:
    """MatchState contains all live scoring, target, and rate fields."""
    state = MatchState(
        match_id="m1",
        score=150,
        wickets=3,
        balls_left=30,
        target=180,
        current_run_rate=6.0,
        required_run_rate=6.0,
        is_super_over=False,
    )
    assert state.score == 150
    assert state.wickets == 3
    assert state.balls_left == 30
    assert state.target == 180
    assert state.current_run_rate == 6.0
    assert state.required_run_rate == 6.0
    assert state.is_super_over is False
