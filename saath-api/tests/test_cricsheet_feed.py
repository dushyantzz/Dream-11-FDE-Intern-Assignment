"""
Tests for CricsheetReplayFeed and MockUserContextProvider.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.adapters.cricsheet import CricsheetReplayFeed
from app.adapters.mock_user_context import MockUserContextProvider

SAMPLE_MATCH_PATH = Path("../odis_json/919615.json")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replay_final_state_equals_official_result() -> None:
    """The final replayed match state must equal the file's official result."""
    assert SAMPLE_MATCH_PATH.exists(), f"Sample match file missing at {SAMPLE_MATCH_PATH}"

    feed = CricsheetReplayFeed(SAMPLE_MATCH_PATH, speed_multiplier=1_000_000.0)
    async for _ in feed:
        pass

    final = feed.current_state
    assert final.is_match_complete is True
    assert final.winner == "Zimbabwe"
    assert final.win_margin == "5 wickets"
    assert len(final.innings) == 2

    # Innings 1 (Ireland)
    inn1 = final.innings[0]
    assert inn1.batting_team == "Ireland"
    assert inn1.total_runs == 268
    assert inn1.wickets_fallen == 7
    assert inn1.overs_completed == 50.0

    # Innings 2 (Zimbabwe)
    inn2 = final.innings[1]
    assert inn2.batting_team == "Zimbabwe"
    assert inn2.total_runs == 270
    assert inn2.wickets_fallen == 5
    assert inn2.overs_completed == 48.3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replay_at_1000x_finishes_under_two_seconds() -> None:
    """Replaying at 1000x speed must complete in under 2 seconds."""
    assert SAMPLE_MATCH_PATH.exists()

    start = time.perf_counter()
    feed = CricsheetReplayFeed(
        SAMPLE_MATCH_PATH,
        speed_multiplier=1000.0,
        nominal_interval_s=1.0,
    )
    ball_count = 0
    async for _ in feed:
        ball_count += 1

    elapsed = time.perf_counter() - start
    assert ball_count == 606
    assert elapsed < 2.0, f"Replay took {elapsed:.2f}s, expected < 2.0s"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_injected_clock_instant_replay() -> None:
    """An injected clock allows tests to intercept delays and run instantaneously."""
    clock_delays: list[float] = []

    async def mock_clock(delay: float) -> None:
        clock_delays.append(delay)

    feed = CricsheetReplayFeed(
        SAMPLE_MATCH_PATH,
        speed_multiplier=100.0,
        nominal_interval_s=1.0,
        sleep_fn=mock_clock,
    )
    async for _ in feed:
        pass

    assert len(clock_delays) == 606
    assert all(d == pytest.approx(0.01) for d in clock_delays)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_edge_case_wides_and_noballs_not_legal() -> None:
    """Wides and no-balls must not increment legal balls count or completed overs."""
    sample_data = {
        "info": {
            "teams": ["Team A", "Team B"],
            "overs": 20,
            "match_type": "T20",
        },
        "innings": [
            {
                "team": "Team A",
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "Batter 1",
                                "bowler": "Bowler 1",
                                "runs": {"batter": 0, "extras": 1, "total": 1},
                                "extras": {"wides": 1},
                            },
                            {
                                "batter": "Batter 1",
                                "bowler": "Bowler 1",
                                "runs": {"batter": 1, "extras": 1, "total": 2},
                                "extras": {"noballs": 1},
                            },
                            {
                                "batter": "Batter 1",
                                "bowler": "Bowler 1",
                                "runs": {"batter": 4, "extras": 0, "total": 4},
                            },
                        ],
                    }
                ],
            }
        ],
    }

    feed = CricsheetReplayFeed(sample_data, speed_multiplier=1_000_000.0)
    balls = [b async for b in feed]

    assert len(balls) == 3
    # Wide
    assert balls[0].is_legal is False
    assert balls[0].runs_total == 1
    # No-ball
    assert balls[1].is_legal is False
    assert balls[1].runs_total == 2
    # Legal ball
    assert balls[2].is_legal is True
    assert balls[2].runs_total == 4

    # Running state: only 1 legal ball bowled, total score is 7
    st = feed.current_state
    assert st.score == 7
    assert st.legal_balls_bowled == 1
    assert st.overs_completed == 0.1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_edge_case_rain_reduced_target() -> None:
    """Rain-reduced matches with explicit target overs and runs are respected."""
    sample_data = {
        "info": {
            "teams": ["England", "Australia"],
            "overs": 50,
            "match_type": "ODI",
        },
        "innings": [
            {
                "team": "England",
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "Root",
                                "bowler": "Starc",
                                "runs": {"batter": 1, "extras": 0, "total": 1},
                            }
                        ],
                    }
                ],
            },
            {
                "team": "Australia",
                "target": {"overs": 20, "runs": 154},
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "Warner",
                                "bowler": "Woakes",
                                "runs": {"batter": 4, "extras": 0, "total": 4},
                            }
                        ],
                    }
                ],
            },
        ],
    }

    feed = CricsheetReplayFeed(sample_data, speed_multiplier=1_000_000.0)
    async for _ in feed:
        pass

    final = feed.current_state
    assert final.target == 154
    # 20 overs = 120 scheduled balls. 1 legal ball bowled -> 119 balls left
    assert final.balls_left == 119
    # RRR = (154 - 4) / (119 / 6.0)
    assert final.required_run_rate is not None
    assert final.required_run_rate > 7.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_edge_case_super_over() -> None:
    """Super overs flag is_super_over on both Ball and MatchState."""
    sample_data = {
        "info": {"teams": ["India", "New Zealand"], "overs": 20},
        "innings": [
            {
                "team": "India",
                "super_over": True,
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "Sharma",
                                "bowler": "Southee",
                                "runs": {"batter": 6, "extras": 0, "total": 6},
                            }
                        ],
                    }
                ],
            }
        ],
    }

    feed = CricsheetReplayFeed(sample_data, speed_multiplier=1_000_000.0)
    balls = [b async for b in feed]
    assert len(balls) == 1
    assert balls[0].is_super_over is True
    assert feed.current_state.is_super_over is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_edge_case_missing_fields() -> None:
    """Deliveries with missing or empty fields are handled safely."""
    sample_data = {
        "info": {},
        "innings": [{"overs": [{"deliveries": [{}]}]}],
    }

    feed = CricsheetReplayFeed(sample_data, speed_multiplier=1_000_000.0)
    balls = [b async for b in feed]
    assert len(balls) == 1
    assert balls[0].runs_total == 0
    assert balls[0].batter == ""
    assert feed.current_state.score == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mock_user_context_provider() -> None:
    """MockUserContextProvider provides 3 sample users and a valid fallback."""
    provider = MockUserContextProvider()

    # User 1: Analyst
    u1 = await provider.get_user_context("usr_analyst")
    assert u1.preferred_persona == "analyst"
    assert u1.preferred_language == "en"
    assert len(u1.fantasy_picks) == 3
    assert "Virat Kohli" in u1.fantasy_picks

    # User 2: Roaster
    u2 = await provider.get_user_context("usr_roaster")
    assert u2.preferred_persona == "roaster"
    assert u2.preferred_language == "hi"
    assert "Babar Azam" in u2.fantasy_picks

    # User 3: Fanatic
    u3 = await provider.get_user_context("usr_fanatic")
    assert u3.preferred_persona == "enthusiastic"
    assert u3.preferred_language == "hinglish"
    assert "Hardik Pandya" in u3.fantasy_picks

    # Unknown user fallback
    unknown = await provider.get_user_context("usr_stranger_99")
    assert unknown.user_id == "usr_stranger_99"
    assert len(unknown.fantasy_picks) > 0
    assert unknown.preferred_persona == "analyst"
