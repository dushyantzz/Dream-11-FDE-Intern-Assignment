"""
Tests for app/engine/winprob.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.models import MatchFormat, MatchState
from app.engine.winprob import (
    WinProbabilityEstimator,
    extract_features,
    first_innings_win_prob,
    train_winprob_model,
)


@pytest.mark.unit
def test_first_innings_win_prob_baseline() -> None:
    """First innings baseline maps projected score to sigmoid probability."""
    # Par score scenario (270 at 50 overs)
    p_even = first_innings_win_prob(score=135, balls_left=150, wickets=2, scheduled_balls=300)
    assert 0.40 <= p_even <= 0.65

    # Dominated scenario (e.g. 50/8 in 20 overs)
    p_low = first_innings_win_prob(score=50, balls_left=180, wickets=8, scheduled_balls=300)
    assert p_low < 0.25

    # Blistering start scenario (e.g. 200/1 in 25 overs)
    p_high = first_innings_win_prob(score=200, balls_left=150, wickets=1, scheduled_balls=300)
    assert p_high > 0.75


@pytest.mark.unit
def test_extract_features_clamping() -> None:
    """Features are bounded and sanitized."""
    feats = extract_features(
        runs_needed=50.0,
        balls_left=30.0,
        wickets_in_hand=6.0,
        required_rate=45.0,  # exceeds max allowed 36.0
        current_rate=-2.0,   # negative rate
    )
    assert feats[0] == 50.0
    assert feats[1] == 30.0
    assert feats[2] == 6.0
    assert feats[3] == 36.0  # clamped
    assert feats[4] == 0.0   # clamped


@pytest.mark.unit
def test_train_winprob_pipeline(tmp_path: Path) -> None:
    """Model training pipeline executes, evaluates, and writes artifacts."""
    train_dir = Path("data/train")
    if not train_dir.exists() or len(list(train_dir.glob("*.json"))) == 0:
        pytest.skip("data/train directory not populated")

    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "eval_report.md"

    bundle = train_winprob_model(
        train_dir=train_dir,
        model_save_path=model_path,
        report_path=report_path,
    )

    assert model_path.exists()
    assert report_path.exists()
    assert "model" in bundle
    assert "metrics" in bundle
    assert "logistic_regression" in bundle["metrics"]
    assert "gradient_boosting" in bundle["metrics"]
    assert bundle["metrics"]["logistic_regression"]["auc"] > 0.70

    report_text = report_path.read_text(encoding="utf-8")
    assert "Win Probability Model Evaluation Report" in report_text
    assert "AUC" in report_text


@pytest.mark.unit
def test_estimator_predictions() -> None:
    """Estimator handles innings 1, innings 2, and completed match states."""
    estimator = WinProbabilityEstimator()

    # Completed match
    st_won = MatchState(
        match_id="m1",
        batting_team="Team A",
        winner="Team A",
        is_match_complete=True,
    )
    assert estimator.predict(st_won) == 1.0

    st_lost = MatchState(
        match_id="m1",
        batting_team="Team A",
        winner="Team B",
        is_match_complete=True,
    )
    assert estimator.predict(st_lost) == 0.0

    # Innings 1
    st_inn1 = MatchState(
        match_id="m1",
        innings_idx=1,
        match_format=MatchFormat.ODI,
        score=200,
        wickets=2,
        balls_left=120,
    )
    p1 = estimator.predict(st_inn1)
    assert 0.0 < p1 < 1.0

    # Innings 2 with target reached
    st_inn2_won = MatchState(
        match_id="m1",
        innings_idx=2,
        score=220,
        target=219,
        balls_left=30,
        wickets=4,
    )
    assert estimator.predict(st_inn2_won) == 1.0

    # Innings 2 all out
    st_inn2_out = MatchState(
        match_id="m1",
        innings_idx=2,
        score=150,
        target=220,
        balls_left=30,
        wickets=10,
    )
    assert estimator.predict(st_inn2_out) == 0.0
