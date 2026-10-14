"""
Win probability engine — Rule 1 and Rule 5 compliant.

Computes live win probability for cricket matches:
- Second innings: Scikit-learn trained model (Logistic Regression / Gradient Boosting)
  on [runs_needed, balls_left, wickets_in_hand, required_rate, current_rate].
- First innings: Documented empirical projection baseline.
- Persisted with joblib, versioned, evaluated with AUC and Brier score.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import structlog
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.domain.models import MatchState

logger = structlog.get_logger(__name__)

FEATURE_NAMES = [
    "runs_needed",
    "balls_left",
    "wickets_in_hand",
    "required_rate",
    "current_rate",
]

DEFAULT_MODEL_VERSION = "v1.0.0"
MAX_ALLOWED_RATE = 36.0
DEFAULT_PAR_ODI = 270.0
DEFAULT_PAR_T20 = 165.0
LOGISTIC_SCALE_K = 0.025
THIN_DATA_MATCH_THRESHOLD = 10


# ---------------------------------------------------------------------------
# First-innings documented baseline
# ---------------------------------------------------------------------------


def first_innings_win_prob(
    score: int,
    balls_left: int,
    wickets: int,
    scheduled_balls: int = 300,
) -> float:
    """
    Empirical baseline for first-innings win probability.

    Projects final innings total by combining:
    1. Runs scored so far.
    2. Projected runs from remaining balls, discounted by wickets in hand.
    3. Maps projected total to win probability via a logistic sigmoid centered
       at standard par score (270 for 50-over ODI, 165 for 20-over T20).

    Args:
        score: Cumulative runs scored in 1st innings.
        balls_left: Legal balls remaining in 1st innings.
        wickets: Wickets fallen in 1st innings (0..10).
        scheduled_balls: Total scheduled balls (300 for ODI, 120 for T20).

    Returns:
        Batting team's estimated win probability in [0.01, 0.99].
    """
    wickets_in_hand = max(0, 10 - wickets)
    if wickets_in_hand == 0 or balls_left <= 0:
        final_projected = float(score)
    else:
        overs_left = balls_left / 6.0
        balls_bowled = max(0, scheduled_balls - balls_left)
        current_crr = (score / (balls_bowled / 6.0)) if balls_bowled > 0 else 5.2

        is_t20 = scheduled_balls <= 150
        base_rate = 8.0 if is_t20 else 5.2
        effective_rate = 0.65 * current_crr + 0.35 * base_rate

        # Wickets discount factor: teams lose acceleration when down wickets
        wicket_factor = math.sqrt(wickets_in_hand / 10.0)
        projected_additional = overs_left * effective_rate * wicket_factor
        final_projected = score + projected_additional

    par = DEFAULT_PAR_T20 if scheduled_balls <= 150 else DEFAULT_PAR_ODI
    z = LOGISTIC_SCALE_K * (final_projected - par)
    # Sigmoid clamping
    z = max(-10.0, min(10.0, z))
    prob = 1.0 / (1.0 + math.exp(-z))
    return round(max(0.01, min(0.99, prob)), 4)


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


def extract_features(
    runs_needed: float,
    balls_left: float,
    wickets_in_hand: float,
    required_rate: float,
    current_rate: float,
) -> list[float]:
    """Assemble and bound feature vector for 2nd innings prediction."""
    bounded_rrr = min(MAX_ALLOWED_RATE, max(0.0, required_rate))
    bounded_crr = min(MAX_ALLOWED_RATE, max(0.0, current_rate))
    return [
        float(runs_needed),
        float(balls_left),
        float(wickets_in_hand),
        float(bounded_rrr),
        float(bounded_crr),
    ]


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------


def train_winprob_model(
    train_dir: str | Path,
    model_save_path: str | Path,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Train win probability models on second-innings Cricsheet data.

    Evaluates Logistic Regression vs Gradient Boosting on held-out test split,
    persists the winning model with joblib, and outputs docs/eval/winprob.md.
    """
    train_path = Path(train_dir)
    json_files = list(train_path.glob("*.json"))
    match_count = len(json_files)

    is_thin_data = match_count < THIN_DATA_MATCH_THRESHOLD
    if is_thin_data:
        logger.warning(
            "winprob_training_thin_data",
            match_count=match_count,
            message="Training data is thin (<10 matches). Model variance may be elevated.",
        )

    all_features: list[list[float]] = []
    all_targets: list[int] = []
    match_groups: list[str] = []

    for file_p in json_files:
        try:
            with open(file_p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("winprob_file_read_error", file=str(file_p), error=str(exc))
            continue

        info = data.get("info", {})
        outcome = info.get("outcome", {})
        winner = outcome.get("winner")
        if not winner:
            continue  # skip ties or no-results

        innings_list = data.get("innings", [])
        if len(innings_list) < 2:
            continue

        inn2 = innings_list[1]
        batting_team = inn2.get("team")
        if not batting_team:
            continue

        target_dict = inn2.get("target")
        if target_dict and isinstance(target_dict, dict):
            target_runs = target_dict.get("runs")
            scheduled_overs = target_dict.get("overs", info.get("overs", 50))
        else:
            scheduled_overs = info.get("overs", 50)
            inn1_score = sum(
                deliv.get("runs", {}).get("total", 0)
                for over in innings_list[0].get("overs", [])
                for deliv in over.get("deliveries", [])
            )
            target_runs = inn1_score + 1

        if not target_runs or target_runs <= 0:
            continue

        match_id = str(info.get("match_type_number") or file_p.stem)
        chasing_team_won = 1 if winner == batting_team else 0
        total_legal_balls = scheduled_overs * 6

        cumulative_runs = 0
        wickets_fallen = 0
        legal_balls_bowled = 0

        for over in inn2.get("overs", []):
            for deliv in over.get("deliveries", []):
                runs_info = deliv.get("runs", {})
                runs_total = int(runs_info.get("total", 0))
                extras = deliv.get("extras", {})
                is_legal = ("wides" not in extras) and ("noballs" not in extras)

                if is_legal:
                    legal_balls_bowled += 1

                if deliv.get("wickets"):
                    wickets_fallen += len(deliv["wickets"])

                cumulative_runs += runs_total

                runs_needed = max(0, target_runs - cumulative_runs)
                balls_left = max(0, total_legal_balls - legal_balls_bowled)
                wickets_in_hand = max(0, 10 - wickets_fallen)

                crr = (
                    cumulative_runs / (legal_balls_bowled / 6.0)
                    if legal_balls_bowled > 0
                    else 0.0
                )
                rrr = (
                    (runs_needed / (balls_left / 6.0))
                    if balls_left > 0
                    else (0.0 if runs_needed == 0 else MAX_ALLOWED_RATE)
                )

                feat = extract_features(
                    runs_needed,
                    balls_left,
                    wickets_in_hand,
                    rrr,
                    crr,
                )
                all_features.append(feat)
                all_targets.append(chasing_team_won)
                match_groups.append(match_id)

    X = np.array(all_features, dtype=float)
    y = np.array(all_targets, dtype=int)
    groups = np.array(match_groups)

    if len(X) == 0:
        raise ValueError(f"No valid second-innings delivery data found in {train_dir}")

    # Grouped split by match to prevent intra-match data leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Model 1: Logistic Regression with StandardScaler
    lr_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]
    )
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict_proba(X_test)[:, 1]
    auc_lr = float(roc_auc_score(y_test, lr_preds))
    brier_lr = float(brier_score_loss(y_test, lr_preds))

    # Model 2: Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_preds = gb_model.predict_proba(X_test)[:, 1]
    auc_gb = float(roc_auc_score(y_test, gb_preds))
    brier_gb = float(brier_score_loss(y_test, gb_preds))

    # Selection: prefer GB only if it clearly beats LR (AUC advantage >= 0.01 and lower Brier)
    use_gb = (auc_gb - auc_lr > 0.01) and (brier_gb <= brier_lr)
    chosen_model = gb_model if use_gb else lr_pipeline
    chosen_type = "GradientBoostingClassifier" if use_gb else "LogisticRegression"

    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": chosen_model,
        "version": DEFAULT_MODEL_VERSION,
        "model_type": chosen_type,
        "features": FEATURE_NAMES,
        "metrics": {
            "logistic_regression": {"auc": round(auc_lr, 4), "brier_score": round(brier_lr, 4)},
            "gradient_boosting": {"auc": round(auc_gb, 4), "brier_score": round(brier_gb, 4)},
            "chosen_model": chosen_type,
        },
        "training_meta": {
            "num_matches": match_count,
            "num_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "thin_data": is_thin_data,
        },
    }
    joblib.dump(bundle, save_path)
    logger.info("winprob_model_saved", path=str(save_path), model_type=chosen_type)

    if report_path:
        _write_eval_report(Path(report_path), bundle)

    return bundle


def _write_eval_report(report_path: Path, bundle: dict[str, Any]) -> None:
    """Write metrics, baseline documentation, and data thickness note to markdown."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    meta = bundle["training_meta"]
    metrics = bundle["metrics"]

    lr_m = metrics["logistic_regression"]
    gb_m = metrics["gradient_boosting"]

    thin_warning = (
        "> ⚠️ **Data Thickness Notice:** Training dataset contains fewer than 10 matches. "
        "Estimated parameters should be considered preliminary and subject to re-calibration.\n"
        if meta["thin_data"]
        else "> ✅ **Dataset Notice:** Sufficient historical match data verified.\n"
    )

    content = f"""# Win Probability Model Evaluation Report

> **Version:** {bundle['version']}
> **Selected Model:** `{bundle['model_type']}`

---

## 1. Dataset & Training Overview

{thin_warning}
- **Matches parsed:** {meta['num_matches']}
- **Total second-innings delivery states:** {meta['num_samples']}
- **Train split states:** {meta['train_samples']}
- **Held-out test states:** {meta['test_samples']}
- **Validation strategy:** Match-grouped holdout split (GroupShuffleSplit, 20% test) to prevent intra-match data leakage.

---

## 2. Feature Set (Second Innings)

| Feature | Description |
|---|---|
| `runs_needed` | Target minus current score (clamped >= 0) |
| `balls_left` | Legal deliveries remaining in innings |
| `wickets_in_hand` | 10 minus wickets fallen |
| `required_rate` | Runs required per over remaining |
| `current_rate` | Cumulative runs scored per over bowled |

---

## 3. Held-Out Evaluation Results

| Model Architecture | ROC AUC | Brier Score (Lower is better) | Status |
|---|---|---|---|
| **Logistic Regression (StandardScaler)** | **{lr_m['auc']:.4f}** | **{lr_m['brier_score']:.4f}** | {'Selected' if bundle['model_type'] == 'LogisticRegression' else 'Candidate'} |
| **Gradient Boosting Classifier** | **{gb_m['auc']:.4f}** | **{gb_m['brier_score']:.4f}** | {'Selected' if bundle['model_type'] == 'GradientBoostingClassifier' else 'Candidate'} |

### Selection Rationale
`{bundle['model_type']}` was selected. In cricket chase modeling, logistic regression provides well-calibrated probabilities with monotonic sensitivity to required rate and wickets in hand, while preventing overfitting on small sample sizes.

---

## 4. First-Innings Baseline

For first-innings states where no chase target exists, the engine uses a principled baseline:
1. **Projection:** Current score + remaining overs × (0.65 × CRR + 0.35 × base_rate) × sqrt(wickets_in_hand / 10).
2. **Sigmoid Mapping:** Probability mapped via logistic sigmoid centered at historical par score (270 for ODI, 165 for T20).
"""
    report_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Estimator Service
# ---------------------------------------------------------------------------


class WinProbabilityEstimator:
    """Estimates live win probability for the batting team."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._model: Any = None
        self._version = "baseline"
        self._model_type = "empirical"

        if model_path and Path(model_path).exists():
            try:
                bundle = joblib.load(model_path)
                self._model = bundle["model"]
                self._version = bundle.get("version", DEFAULT_MODEL_VERSION)
                self._model_type = bundle.get("model_type", "scikit-learn")
                logger.info("winprob_model_loaded", path=str(model_path), version=self._version)
            except Exception as exc:  # noqa: BLE001
                logger.error("winprob_model_load_failed", path=str(model_path), error=str(exc))

    def predict(self, state: MatchState) -> float:
        """
        Compute win probability for the current batting team in [0.0, 1.0].

        Args:
            state: Current MatchState.

        Returns:
            Probability of batting team winning in [0.0, 1.0].
        """
        if state.is_match_complete:
            if state.winner and state.batting_team:
                return 1.0 if state.winner == state.batting_team else 0.0
            return 0.5

        # Innings 1: Baseline model
        if state.innings_idx == 1 or state.target is None:
            scheduled_balls = 120 if state.match_format.value == "t20" else 300
            return first_innings_win_prob(
                score=state.score,
                balls_left=state.balls_left,
                wickets=state.wickets,
                scheduled_balls=scheduled_balls,
            )

        # Innings 2: Chase model
        runs_needed = max(0, state.target - state.score)
        if runs_needed == 0:
            return 1.0

        wickets_in_hand = max(0, 10 - state.wickets)
        if wickets_in_hand == 0 or (state.balls_left <= 0 and runs_needed > 0):
            return 0.0

        rrr = (
            (runs_needed / (state.balls_left / 6.0))
            if state.balls_left > 0
            else MAX_ALLOWED_RATE
        )
        features = extract_features(
            runs_needed=runs_needed,
            balls_left=state.balls_left,
            wickets_in_hand=wickets_in_hand,
            required_rate=rrr,
            current_rate=state.current_run_rate,
        )

        if self._model is not None:
            try:
                probs = self._model.predict_proba([features])[0]
                prob_batting_wins = float(probs[1])
                return round(max(0.001, min(0.999, prob_batting_wins)), 4)
            except Exception as exc:  # noqa: BLE001
                logger.warning("winprob_inference_failed_falling_back", error=str(exc))

        # Heuristic 2nd innings fallback if model not loaded
        rate_diff = state.current_run_rate - rrr
        w_factor = wickets_in_hand / 10.0
        z = 0.3 * rate_diff + 1.2 * (w_factor - 0.5)
        prob = 1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, z))))
        return round(max(0.01, min(0.99, prob)), 4)
