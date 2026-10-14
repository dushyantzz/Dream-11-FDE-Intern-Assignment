# Win Probability Model Evaluation Report

> **Version:** v1.0.0
> **Selected Model:** `LogisticRegression`

---

## 1. Dataset & Training Overview

> ✅ **Dataset Notice:** Sufficient historical match data verified.

- **Matches parsed:** 50
- **Total second-innings delivery states:** 11851
- **Train split states:** 9216
- **Held-out test states:** 2635
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
| **Logistic Regression (StandardScaler)** | **0.9192** | **0.1545** | Selected |
| **Gradient Boosting Classifier** | **0.8126** | **0.2312** | Candidate |

### Selection Rationale
`LogisticRegression` was selected. In cricket chase modeling, logistic regression provides well-calibrated probabilities with monotonic sensitivity to required rate and wickets in hand, while preventing overfitting on small sample sizes.

---

## 4. First-Innings Baseline

For first-innings states where no chase target exists, the engine uses a principled baseline:
1. **Projection:** Current score + remaining overs × (0.65 × CRR + 0.35 × base_rate) × sqrt(wickets_in_hand / 10).
2. **Sigmoid Mapping:** Probability mapped via logistic sigmoid centered at historical par score (270 for ODI, 165 for T20).
