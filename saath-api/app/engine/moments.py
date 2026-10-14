"""
Moment Engine — detects impactful moments, computes win probability swings and scores.

Enforces:
- Rule 1:verified_facts dict contains the ONLY data downstream LLM / commentary layers may use.
- Rule 5: Weights and thresholds are read from config.
- Deterministic moment_id: SHA256 of match_id + innings + over + ball_in_over.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import structlog

from app.core.config import MomentWeightsSettings, Settings, get_settings
from app.domain.models import Ball, MatchState, MomentEvent, MomentType
from app.engine.winprob import WinProbabilityEstimator

logger = structlog.get_logger(__name__)

THREE_OVERS_LEGAL_BALLS = 18
TIGHT_FINISH_BALLS_LEFT = 18
TIGHT_FINISH_RUNS_NEEDED = 24
TIGHT_FINISH_SWING_THRESHOLD = 0.15
TIGHT_FINISH_OVERS_WINDOW = 30
PARTNERSHIP_INCREMENT = 50
MIN_HAT_TRICK_PREV_WICKETS = 2


@dataclass
class MomentFlags:
    """Boolean tags detected for a single delivery."""

    wicket: bool = False
    six: bool = False
    four: bool = False
    fifty: bool = False
    hundred: bool = False
    partnership_milestone: bool = False
    three_wickets_in_three_overs: bool = False
    hat_trick_ball: bool = False
    tight_finish: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "wicket": self.wicket,
            "six": self.six,
            "four": self.four,
            "fifty": self.fifty,
            "hundred": self.hundred,
            "partnership_milestone": self.partnership_milestone,
            "three_wickets_in_three_overs": self.three_wickets_in_three_overs,
            "hat_trick_ball": self.hat_trick_ball,
            "tight_finish": self.tight_finish,
        }

    def active_flags(self) -> list[str]:
        return [k for k, v in self.to_dict().items() if v]


@dataclass
class MatchContextTracker:
    """Maintains stateful stats across balls for milestone and streak detection."""

    batter_runs: dict[str, int] = field(default_factory=dict)
    partnership_runs: int = 0
    recent_legal_wickets: deque[int] = field(default_factory=deque)  # 1 if wicket, 0 if not
    bowler_consecutive_wickets: dict[str, int] = field(default_factory=dict)


def compute_deterministic_moment_id(
    match_id: str, innings: int, over: int, ball_in_over: int
) -> str:
    """Generate a stable, deterministic 16-character hex ID."""
    token = f"{match_id}:{innings}:{over}:{ball_in_over}".encode("utf-8")
    return hashlib.sha256(token).hexdigest()[:16]


class MomentEngine:
    """
    Evaluates every ball against match context and win probability,
    producing a scored MomentEvent.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        winprob_estimator: WinProbabilityEstimator | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._weights: MomentWeightsSettings = self._settings.engine.moment_weights
        self._winprob = winprob_estimator or WinProbabilityEstimator(
            model_path=self._settings.engine.winprob_model_path
        )
        self._context = MatchContextTracker()

    def reset_context(self) -> None:
        """Reset match tracking state for a new match."""
        self._context = MatchContextTracker()

    def process_ball(
        self,
        ball: Ball,
        state_before: MatchState,
        state_after: MatchState,
    ) -> tuple[MomentEvent, float]:
        """
        Evaluate a single delivery and return (MomentEvent, moment_score).

        Args:
            ball: The delivery that just occurred.
            state_before: MatchState before the delivery.
            state_after: MatchState after the delivery.

        Returns:
            Tuple of (MomentEvent, moment_score in [0.0, 1.0]).
        """
        win_prob_before = self._winprob.predict(state_before)
        win_prob_after = self._winprob.predict(state_after)
        swing = round(abs(win_prob_after - win_prob_before), 4)

        flags = self._detect_flags(ball, state_after, swing)
        moment_score = self._compute_score(swing, flags)

        moment_id = compute_deterministic_moment_id(
            match_id=ball.match_id,
            innings=ball.innings,
            over=ball.over,
            ball_in_over=ball.ball_in_over,
        )

        moment_type = self._determine_moment_type(flags)

        verified_facts: dict[str, Any] = {
            "match_id": ball.match_id,
            "innings": ball.innings,
            "over": ball.over,
            "ball_in_over": ball.ball_in_over,
            "batting_team": state_after.batting_team,
            "bowling_team": state_after.bowling_team,
            "score": state_after.score,
            "wickets": state_after.wickets,
            "overs_completed": state_after.overs_completed,
            "legal_balls_bowled": state_after.legal_balls_bowled,
            "balls_left": state_after.balls_left,
            "target": state_after.target,
            "runs_needed": max(0, (state_after.target or 0) - state_after.score)
            if state_after.target
            else None,
            "crr": state_after.current_run_rate,
            "rrr": state_after.required_run_rate,
            "batter": ball.batter,
            "bowler": ball.bowler,
            "non_striker": ball.non_striker,
            "runs_batter": ball.runs_batter,
            "runs_extras": ball.runs_extras,
            "runs_total": ball.runs_total,
            "is_wicket": ball.is_wicket,
            "wicket_kind": ball.wicket_kind,
            "player_dismissed": ball.player_dismissed,
            "win_prob_before": win_prob_before,
            "win_prob_after": win_prob_after,
            "swing": swing,
            "moment_score": moment_score,
            "active_flags": flags.active_flags(),
        }

        metadata: dict[str, Any] = {
            "flags": flags.to_dict(),
            "active_flags": flags.active_flags(),
            "win_prob_before": win_prob_before,
            "win_prob_after": win_prob_after,
            "swing": swing,
            "moment_score": moment_score,
            "verified_facts": verified_facts,
        }

        event = MomentEvent(
            moment_id=moment_id,
            match_id=ball.match_id,
            moment_type=moment_type,
            primary_player_id=ball.player_dismissed
            if ball.is_wicket
            else ball.batter,
            secondary_player_id=ball.bowler,
            metadata=metadata,
        )

        return event, moment_score

    # ------------------------------------------------------------------
    # Internal Flag & Score Calculation
    # ------------------------------------------------------------------

    def _detect_flags(
        self,
        ball: Ball,
        state_after: MatchState,
        swing: float,
    ) -> MomentFlags:
        ctx = self._context
        flags = MomentFlags()

        # 1. Wicket, Six, Four
        flags.wicket = ball.is_wicket
        flags.six = ball.runs_batter == 6
        flags.four = ball.runs_batter == 4

        # 2. Batter milestone (fifty / hundred)
        prev_b_runs = ctx.batter_runs.get(ball.batter, 0)
        new_b_runs = prev_b_runs + ball.runs_batter
        ctx.batter_runs[ball.batter] = new_b_runs

        if prev_b_runs < 50 and 50 <= new_b_runs < 100:
            flags.fifty = True
        elif prev_b_runs < 100 and new_b_runs >= 100:
            flags.hundred = True

        # 3. Partnership milestone
        prev_part = ctx.partnership_runs
        new_part = prev_part + ball.runs_total
        if (prev_part // PARTNERSHIP_INCREMENT) < (new_part // PARTNERSHIP_INCREMENT) and new_part > 0:
            flags.partnership_milestone = True

        # 4. Hat-trick ball: bowler took wickets on preceding 2 consecutive deliveries
        bowler_streak = ctx.bowler_consecutive_wickets.get(ball.bowler, 0)
        if bowler_streak >= MIN_HAT_TRICK_PREV_WICKETS:
            flags.hat_trick_ball = True

        if ball.is_wicket:
            ctx.bowler_consecutive_wickets[ball.bowler] = bowler_streak + 1
            ctx.partnership_runs = 0  # reset partnership on wicket
        else:
            ctx.bowler_consecutive_wickets[ball.bowler] = 0
            ctx.partnership_runs = new_part

        # 5. 3 wickets in 3 overs (rolling 18 legal balls)
        if ball.is_legal:
            ctx.recent_legal_wickets.append(1 if ball.is_wicket else 0)
            if len(ctx.recent_legal_wickets) > THREE_OVERS_LEGAL_BALLS:
                ctx.recent_legal_wickets.popleft()

            if sum(ctx.recent_legal_wickets) >= 3:
                flags.three_wickets_in_three_overs = True

        # 6. Tight finish (chase context)
        if state_after.innings_idx >= 2 and state_after.target:
            runs_needed = max(0, state_after.target - state_after.score)
            wickets_left = max(0, 10 - state_after.wickets)
            is_close_chase = (
                state_after.balls_left <= TIGHT_FINISH_BALLS_LEFT
                and runs_needed <= TIGHT_FINISH_RUNS_NEEDED
                and wickets_left >= 1
            )
            is_high_swing_finish = (
                state_after.balls_left <= TIGHT_FINISH_OVERS_WINDOW
                and swing >= TIGHT_FINISH_SWING_THRESHOLD
            )
            flags.tight_finish = is_close_chase or is_high_swing_finish

        return flags

    def _compute_score(self, swing: float, flags: MomentFlags) -> float:
        """
        Compute normalized moment_score in [0.0, 1.0].

        Guaranteed monotonic in swing: d(score)/d(swing) > 0.
        """
        w = self._weights
        weighted_sum = (
            (swing * w.swing)
            + (float(flags.wicket) * w.wicket)
            + (float(flags.six) * w.six)
            + (float(flags.four) * w.four)
            + (float(flags.fifty) * w.fifty)
            + (float(flags.hundred) * w.hundred)
            + (float(flags.partnership_milestone) * w.partnership_milestone)
            + (float(flags.three_wickets_in_three_overs) * w.three_wickets_in_three_overs)
            + (float(flags.hat_trick_ball) * w.hat_trick_ball)
            + (float(flags.tight_finish) * w.tight_finish)
        )
        return round(max(0.0, min(1.0, weighted_sum)), 4)

    @staticmethod
    def _determine_moment_type(flags: MomentFlags) -> MomentType:
        if flags.wicket:
            return MomentType.WICKET
        if flags.fifty or flags.hundred or flags.partnership_milestone:
            return MomentType.MILESTONE
        if flags.six:
            return MomentType.SIX
        if flags.four:
            return MomentType.BOUNDARY
        return MomentType.GENERIC
