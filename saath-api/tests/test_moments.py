"""
Tests for app/engine/moments.py and app/engine/winprob.py.

Verifies:
1. 5 hand-picked real balls with expected flags.
2. moment_id is stable and deterministic across runs.
3. moment_score is monotonic in win probability swing.
4. Full match replay generates a moment score distribution table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.adapters.cricsheet import CricsheetReplayFeed
from app.domain.models import Ball, MatchFormat, MatchState, MomentType
from app.engine.moments import MomentEngine, compute_deterministic_moment_id
from app.engine.winprob import WinProbabilityEstimator

SAMPLE_MATCH_PATH = Path("../odis_json/919615.json")


def _make_dummy_state(
    score: int = 150,
    wickets: int = 3,
    balls_left: int = 60,
    target: int = 200,
    innings_idx: int = 2,
) -> MatchState:
    return MatchState(
        match_id="test_match_001",
        match_format=MatchFormat.ODI,
        home_team="Team A",
        away_team="Team B",
        innings_idx=innings_idx,
        batting_team="Team A",
        bowling_team="Team B",
        score=score,
        wickets=wickets,
        overs_completed=40.0,
        legal_balls_bowled=240,
        balls_left=balls_left,
        target=target,
        current_run_rate=score / 40.0,
        required_run_rate=(target - score) / (balls_left / 6.0) if balls_left > 0 else 0.0,
    )


@pytest.mark.unit
def test_moment_id_is_stable_and_deterministic() -> None:
    """The generated moment_id must be completely deterministic and stable across executions."""
    id1 = compute_deterministic_moment_id("match_123", 2, 19, 4)
    id2 = compute_deterministic_moment_id("match_123", 2, 19, 4)
    id3 = compute_deterministic_moment_id("match_123", 2, 19, 5)

    assert id1 == id2
    assert len(id1) == 16
    assert id1 != id3


@pytest.mark.unit
def test_moment_score_is_monotonic_in_swing() -> None:
    """moment_score must be strictly non-decreasing with increasing swing."""
    engine = MomentEngine()

    swings = [0.0, 0.05, 0.10, 0.18, 0.25, 0.40, 0.60, 0.85]
    scores: list[float] = []

    ball = Ball(
        ball_id="b1",
        match_id="m1",
        innings=2,
        over=10,
        ball_in_over=1,
        bowler="Bowler A",
        batter="Batter A",
        non_striker="Batter B",
        runs_batter=0,
        runs_extras=0,
        runs_total=0,
        is_legal=True,
    )

    for s in swings:
        flags = engine._detect_flags(ball, _make_dummy_state(), swing=s)  # type: ignore[arg-type]
        score = engine._compute_score(s, flags)
        scores.append(score)

    for i in range(len(scores) - 1):
        assert scores[i] <= scores[i + 1], (
            f"Monotonicity violated: swing {swings[i]} gave score {scores[i]}, "
            f"but higher swing {swings[i+1]} gave score {scores[i+1]}"
        )


@pytest.mark.unit
def test_five_hand_picked_balls_with_expected_flags() -> None:
    """Verify 5 hand-picked ball scenarios produce their expected flags."""
    engine = MomentEngine()

    # Ball 1: Clean Wicket
    b1 = Ball(
        ball_id="b1",
        match_id="m1",
        innings=1,
        over=5,
        ball_in_over=2,
        bowler="Starc",
        batter="Roy",
        non_striker="Bairstow",
        runs_batter=0,
        runs_extras=0,
        runs_total=0,
        is_legal=True,
        is_wicket=True,
        wicket_kind="bowled",
        player_dismissed="Roy",
    )
    st_before = _make_dummy_state(score=30, wickets=0)
    st_after = _make_dummy_state(score=30, wickets=1)
    ev1, score1 = engine.process_ball(b1, st_before, st_after)
    assert ev1.metadata["flags"]["wicket"] is True
    assert ev1.moment_type == MomentType.WICKET
    assert ev1.metadata["verified_facts"]["is_wicket"] is True
    assert score1 >= 0.20

    # Ball 2: Huge Six
    engine.reset_context()
    b2 = Ball(
        ball_id="b2",
        match_id="m1",
        innings=1,
        over=10,
        ball_in_over=4,
        bowler="Rashid",
        batter="Miller",
        non_striker="Klaasen",
        runs_batter=6,
        runs_extras=0,
        runs_total=6,
        is_legal=True,
    )
    ev2, score2 = engine.process_ball(b2, st_before, _make_dummy_state(score=36, wickets=0))
    assert ev2.metadata["flags"]["six"] is True
    assert ev2.moment_type == MomentType.SIX
    assert ev2.metadata["verified_facts"]["runs_batter"] == 6

    # Ball 3: Crisp Boundary Four
    engine.reset_context()
    b3 = Ball(
        ball_id="b3",
        match_id="m1",
        innings=1,
        over=12,
        ball_in_over=1,
        bowler="Cummins",
        batter="Kohli",
        non_striker="Rohit",
        runs_batter=4,
        runs_extras=0,
        runs_total=4,
        is_legal=True,
    )
    ev3, _ = engine.process_ball(b3, st_before, _make_dummy_state(score=34, wickets=0))
    assert ev3.metadata["flags"]["four"] is True
    assert ev3.moment_type == MomentType.BOUNDARY

    # Ball 4: Batter Reaches Fifty
    engine.reset_context()
    engine._context.batter_runs["Kohli"] = 48  # Batter was on 48
    b4 = Ball(
        ball_id="b4",
        match_id="m1",
        innings=1,
        over=22,
        ball_in_over=3,
        bowler="Zampa",
        batter="Kohli",
        non_striker="Rohit",
        runs_batter=4,  # reaches 52
        runs_extras=0,
        runs_total=4,
        is_legal=True,
    )
    ev4, _ = engine.process_ball(b4, st_before, _make_dummy_state(score=120, wickets=1))
    assert ev4.metadata["flags"]["fifty"] is True
    assert ev4.moment_type == MomentType.MILESTONE
    assert ev4.metadata["verified_facts"]["batter"] == "Kohli"

    # Ball 5: Hat-trick Delivery (Bowler took 2 wickets on previous 2 balls)
    engine.reset_context()
    engine._context.bowler_consecutive_wickets["Boult"] = 2  # on a hat-trick!
    b5 = Ball(
        ball_id="b5",
        match_id="m1",
        innings=1,
        over=18,
        ball_in_over=5,
        bowler="Boult",
        batter="Stokes",
        non_striker="Root",
        runs_batter=0,
        runs_extras=0,
        runs_total=0,
        is_legal=True,
    )
    ev5, score5 = engine.process_ball(b5, st_before, st_after)
    assert ev5.metadata["flags"]["hat_trick_ball"] is True
    assert score5 >= 0.15


@pytest.mark.unit
@pytest.mark.asyncio
async def test_full_match_moment_score_distribution_and_print() -> None:
    """
    Definition of done check:
    Replaying the full match yields a distribution table of moment_scores,
    printed to stdout so thresholds can be set from data, not guessed.
    """
    assert SAMPLE_MATCH_PATH.exists()

    estimator = WinProbabilityEstimator("app/engine/models/winprob_v1.joblib")
    engine = MomentEngine(winprob_estimator=estimator)
    feed = CricsheetReplayFeed(SAMPLE_MATCH_PATH, speed_multiplier=1_000_000.0)

    scores: list[float] = []
    swings: list[float] = []
    moments_collected: list[tuple[Any, float]] = []

    state_prev = feed.current_state

    async for ball in feed:
        state_curr = feed.current_state
        event, score = engine.process_ball(ball, state_prev, state_curr)
        scores.append(score)
        swings.append(event.metadata["swing"])
        moments_collected.append((event, score))
        state_prev = state_curr

    assert len(scores) == 606

    arr = np.array(scores)
    percentiles = {
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50 (median)": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }

    # Format distribution frequency bins
    bins = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]
    counts, _ = np.histogram(arr, bins=bins)

    table_lines = [
        "\n" + "=" * 70,
        "MOMENT SCORE EMPIRICAL DISTRIBUTION TABLE (Match 919615: IRE vs ZIM)",
        "=" * 70,
        f"{'Metric':<25} | {'Value':<15}",
        "-" * 70,
    ]
    for k, v in percentiles.items():
        table_lines.append(f"{k:<25} | {v:<15.4f}")

    table_lines.append("-" * 70)
    table_lines.append("HISTOGRAM BINS (moment_score frequency):")
    for i in range(len(bins) - 1):
        bin_label = f"[{bins[i]:.2f} - {bins[i+1]:.2f})"
        pct = (counts[i] / len(arr)) * 100
        bar = "#" * int(pct // 2)
        table_lines.append(f"{bin_label:<18} : {counts[i]:4d} balls ({pct:5.1f}%) {bar}")
    table_lines.append("=" * 70 + "\n")

    distribution_output = "\n".join(table_lines)
    print(distribution_output)

    # Sanity checks on the distribution
    assert percentiles["max"] <= 1.0
    assert percentiles["min"] >= 0.0
    assert percentiles["p95"] > percentiles["p50 (median)"]
