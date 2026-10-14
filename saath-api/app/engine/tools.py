"""
Pure function tools over replayed cricket match data.

Rule 1 & Rule 3 compliant:
- Pure functions over domain models.
- Returns typed Pydantic models or None; never a guess, never a string.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.domain.models import Ball, MatchState
from app.engine.winprob import WinProbabilityEstimator


# ---------------------------------------------------------------------------
# Typed Output Models
# ---------------------------------------------------------------------------


class PlayerSummary(BaseModel):
    """Aggregate statistics for a player up to the current delivery."""

    model_config = ConfigDict(frozen=True)

    player_name: str
    runs: int = 0
    balls_faced: int = 0
    strike_rate: float = 0.0
    fours: int = 0
    sixes: int = 0
    overs_bowled: float = 0.0
    wickets: int = 0
    runs_conceded: int = 0
    economy: float = 0.0
    dots_bowled: int = 0


class HeadToHeadSummary(BaseModel):
    """Historical head-to-head match-up between a batter and bowler."""

    model_config = ConfigDict(frozen=True)

    batter: str
    bowler: str
    balls_faced: int = 0
    runs_scored: int = 0
    dismissals: int = 0
    strike_rate: float = 0.0
    dots: int = 0
    fours: int = 0
    sixes: int = 0


class PartnershipSummary(BaseModel):
    """Current batting partnership in the active innings."""

    model_config = ConfigDict(frozen=True)

    batter_1: str
    batter_2: str
    runs: int = 0
    balls: int = 0
    run_rate: float = 0.0


class RequiredRateSummary(BaseModel):
    """Chase equations for the second innings."""

    model_config = ConfigDict(frozen=True)

    target: int
    runs_needed: int
    balls_left: int
    overs_left: float
    required_rate: float
    current_rate: float
    rate_difference: float  # required_rate - current_rate


class WinProbSummary(BaseModel):
    """Live win probability snapshot."""

    model_config = ConfigDict(frozen=True)

    batting_team: str
    bowling_team: str
    win_prob_batting: float = Field(ge=0.0, le=1.0)
    win_prob_bowling: float = Field(ge=0.0, le=1.0)
    swing: float = 0.0


class LastNOversSummary(BaseModel):
    """Metrics aggregated over the last N completed/current overs."""

    model_config = ConfigDict(frozen=True)

    n_overs: int
    runs: int = 0
    wickets: int = 0
    legal_balls: int = 0
    run_rate: float = 0.0
    boundaries: int = 0


# ---------------------------------------------------------------------------
# Tool Implementations (Pure Functions)
# ---------------------------------------------------------------------------


def player_summary(balls: list[Ball], player_name: str) -> PlayerSummary | None:
    """
    Compute aggregate performance metrics for a player in the match so far.

    Returns None if the player has neither batted nor bowled in the deliveries provided.
    """
    if not player_name:
        return None

    runs = 0
    balls_faced = 0
    fours = 0
    sixes = 0

    legal_balls_bowled = 0
    wickets = 0
    runs_conceded = 0
    dots_bowled = 0

    has_participated = False

    for b in balls:
        if b.batter == player_name:
            has_participated = True
            runs += b.runs_batter
            if b.is_legal:
                balls_faced += 1
            if b.runs_batter == 4:
                fours += 1
            elif b.runs_batter == 6:
                sixes += 1

        if b.bowler == player_name:
            has_participated = True
            if b.is_legal:
                legal_balls_bowled += 1
                if b.runs_total == 0:
                    dots_bowled += 1

            runs_conceded += b.runs_total
            if b.is_wicket and b.wicket_kind not in {"run out", "retired hurt"}:
                wickets += 1

    if not has_participated:
        return None

    strike_rate = round((runs / balls_faced) * 100, 2) if balls_faced > 0 else 0.0
    overs = (legal_balls_bowled // 6) + (legal_balls_bowled % 6) / 10.0
    economy = (
        round(runs_conceded / (legal_balls_bowled / 6.0), 2)
        if legal_balls_bowled > 0
        else 0.0
    )

    return PlayerSummary(
        player_name=player_name,
        runs=runs,
        balls_faced=balls_faced,
        strike_rate=strike_rate,
        fours=fours,
        sixes=sixes,
        overs_bowled=overs,
        wickets=wickets,
        runs_conceded=runs_conceded,
        economy=economy,
        dots_bowled=dots_bowled,
    )


def batter_vs_bowler(
    balls: list[Ball], batter: str, bowler: str
) -> HeadToHeadSummary | None:
    """
    Compute head-to-head stats for batter against bowler in deliveries so far.

    Returns None if the batter has not faced this bowler.
    """
    if not batter or not bowler:
        return None

    h2h_balls = [b for b in balls if b.batter == batter and b.bowler == bowler]
    if not h2h_balls:
        return None

    balls_faced = sum(1 for b in h2h_balls if b.is_legal)
    runs_scored = sum(b.runs_batter for b in h2h_balls)
    dismissals = sum(1 for b in h2h_balls if b.is_wicket and b.player_dismissed == batter)
    dots = sum(1 for b in h2h_balls if b.runs_total == 0 and b.is_legal)
    fours = sum(1 for b in h2h_balls if b.runs_batter == 4)
    sixes = sum(1 for b in h2h_balls if b.runs_batter == 6)

    sr = round((runs_scored / balls_faced) * 100, 2) if balls_faced > 0 else 0.0

    return HeadToHeadSummary(
        batter=batter,
        bowler=bowler,
        balls_faced=balls_faced,
        runs_scored=runs_scored,
        dismissals=dismissals,
        strike_rate=sr,
        dots=dots,
        fours=fours,
        sixes=sixes,
    )


def partnership(
    balls: list[Ball], current_state: MatchState
) -> PartnershipSummary | None:
    """
    Compute active partnership in the current innings since the last wicket.

    Returns None if no balls have been bowled in this innings.
    """
    inn_balls = [b for b in balls if b.innings == current_state.innings_idx]
    if not inn_balls:
        return None

    # Walk backwards until the most recent wicket
    part_balls: list[Ball] = []
    for b in reversed(inn_balls):
        if b.is_wicket:
            break
        part_balls.append(b)

    if not part_balls:
        # Last delivery was a wicket, partnership just started
        latest = inn_balls[-1]
        return PartnershipSummary(
            batter_1=latest.batter,
            batter_2=latest.non_striker,
            runs=0,
            balls=0,
            run_rate=0.0,
        )

    runs = sum(b.runs_total for b in part_balls)
    legal_balls = sum(1 for b in part_balls if b.is_legal)
    run_rate = (
        round(runs / (legal_balls / 6.0), 2)
        if legal_balls > 0
        else 0.0
    )

    latest = part_balls[0]
    return PartnershipSummary(
        batter_1=latest.batter,
        batter_2=latest.non_striker,
        runs=runs,
        balls=legal_balls,
        run_rate=run_rate,
    )


def required_rate(current_state: MatchState) -> RequiredRateSummary | None:
    """
    Compute chase dynamics in the second innings.

    Returns None in the first innings or if the target is undefined.
    """
    if current_state.innings_idx < 2 or current_state.target is None:
        return None

    target = current_state.target
    score = current_state.score
    runs_needed = max(0, target - score)
    balls_left = max(0, current_state.balls_left)
    overs_left = round(balls_left / 6.0, 1)

    if runs_needed == 0:
        rrr = 0.0
    elif balls_left > 0:
        rrr = round(runs_needed / (balls_left / 6.0), 2)
    else:
        rrr = 36.0

    crr = current_state.current_run_rate
    diff = round(rrr - crr, 2)

    return RequiredRateSummary(
        target=target,
        runs_needed=runs_needed,
        balls_left=balls_left,
        overs_left=overs_left,
        required_rate=rrr,
        current_rate=crr,
        rate_difference=diff,
    )


def win_prob_now(
    current_state: MatchState,
    winprob_estimator: WinProbabilityEstimator | None = None,
    swing: float = 0.0,
) -> WinProbSummary:
    """Compute current win probability for both teams."""
    estimator = winprob_estimator or WinProbabilityEstimator()
    p_batting = estimator.predict(current_state)
    p_bowling = round(max(0.0, min(1.0, 1.0 - p_batting)), 4)

    return WinProbSummary(
        batting_team=current_state.batting_team,
        bowling_team=current_state.bowling_team,
        win_prob_batting=p_batting,
        win_prob_bowling=p_bowling,
        swing=swing,
    )


def last_n_overs(
    balls: list[Ball], current_state: MatchState, n_overs: int = 5
) -> LastNOversSummary | None:
    """
    Compute aggregate metrics over the last N overs (or fewer if match is early).

    Returns None if no deliveries have occurred in the current innings.
    """
    inn_balls = [b for b in balls if b.innings == current_state.innings_idx]
    if not inn_balls:
        return None

    # Take up to n_overs * 6 legal balls
    max_legal_balls = n_overs * 6
    selected_balls: list[Ball] = []
    legal_count = 0

    for b in reversed(inn_balls):
        selected_balls.append(b)
        if b.is_legal:
            legal_count += 1
            if legal_count >= max_legal_balls:
                break

    runs = sum(b.runs_total for b in selected_balls)
    wickets = sum(1 for b in selected_balls if b.is_wicket)
    boundaries = sum(1 for b in selected_balls if b.runs_batter in {4, 6})
    run_rate = (
        round(runs / (legal_count / 6.0), 2)
        if legal_count > 0
        else 0.0
    )

    return LastNOversSummary(
        n_overs=n_overs,
        runs=runs,
        wickets=wickets,
        legal_balls=legal_count,
        run_rate=run_rate,
        boundaries=boundaries,
    )
