"""
CricsheetReplayFeed — loads a Cricsheet JSON match file and replays it ball-by-ball.

Implements app.adapters.base.MatchFeed. Computes live MatchState per ball
(score, wickets, balls_left, target, current run rate, required run rate),
handling extras, wides/no-balls, rain-reduced targets, super overs, and missing fields.
Replays at configurable speed using an injected clock.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from app.adapters.base import MatchFeed
from app.domain.models import (
    Ball,
    InningsState,
    InningsStatus,
    MatchFormat,
    MatchState,
    PlayerStats,
    Roster,
)

MAX_WICKETS_PER_INNINGS = 10
INSTANT_REPLAY_SPEED_THRESHOLD = 1_000_000.0
MIN_SLEEP_THRESHOLD_S = 0.01


class CricsheetReplayFeed(MatchFeed):
    """
    Replay feed that streams Ball deliveries from a Cricsheet JSON file,
    updating the running MatchState after each ball.
    """

    def __init__(
        self,
        source: str | Path | dict[str, Any],
        *,
        speed_multiplier: float = 1000.0,
        nominal_interval_s: float = 1.0,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        """
        Initialize the replay feed.

        Args:
            source: File path (str/Path) or parsed dict of Cricsheet JSON.
            speed_multiplier: Speed factor (e.g. 1000.0 means 1000x faster than real-time).
            nominal_interval_s: Real-time interval between balls in seconds.
            sleep_fn: Injected sleep/clock function for tests. Defaults to asyncio.sleep.
        """
        if isinstance(source, (str, Path)):
            with open(source, encoding="utf-8") as f:
                self._data: dict[str, Any] = json.load(f)
            self._source_path = str(source)
        else:
            self._data = source
            self._source_path = "memory"

        self._speed_multiplier = max(0.001, speed_multiplier)
        self._nominal_interval_s = max(0.0, nominal_interval_s)
        self._sleep_fn = sleep_fn

        # Parse match-level metadata and rosters
        self._info: dict[str, Any] = self._data.get("info", {})
        self._teams: list[str] = self._info.get("teams", [])
        self._rosters: dict[str, Roster] = self._build_rosters()
        self._match_format = self._determine_format()
        self._metadata: dict[str, Any] = self._build_metadata()

        # Initialize base match state
        self._current_state = self._init_match_state()

    @property
    def roster(self) -> dict[str, Roster]:
        """Team rosters keyed by team name."""
        return self._rosters

    @property
    def metadata(self) -> dict[str, Any]:
        """Match metadata dictionary."""
        return self._metadata

    @property
    def current_state(self) -> MatchState:
        """Current running MatchState snapshot."""
        return self._current_state

    # ------------------------------------------------------------------
    # Async Iterator Replay Implementation
    # ------------------------------------------------------------------

    async def __aiter__(self) -> AsyncIterator[Ball]:
        """Replay deliveries sequentially, updating running match state."""
        innings_list: list[dict[str, Any]] = self._data.get("innings", [])
        outcome = self._info.get("outcome", {})
        winner = outcome.get("winner")
        win_margin = self._format_win_margin(outcome)

        inn_states_history: list[InningsState] = []
        first_innings_score: int | None = None

        self._start_time = time.perf_counter()
        self._ball_count = 0

        for inn_idx, inn_dict in enumerate(innings_list, start=1):
            batting_team = inn_dict.get("team", "")
            bowling_team = next((t for t in self._teams if t != batting_team), "")
            is_super_over = bool(inn_dict.get("super_over", False))

            # Determine scheduled overs and target (handles rain-reduced target)
            target_dict = inn_dict.get("target")
            if target_dict and isinstance(target_dict, dict):
                target_runs: int | None = target_dict.get("runs")
                scheduled_overs: int = target_dict.get("overs", self._info.get("overs", 50))
            else:
                scheduled_overs = self._info.get("overs", 50)
                if inn_idx > 1 and first_innings_score is not None:
                    target_runs = first_innings_score + 1
                else:
                    target_runs = None

            scheduled_legal_balls = scheduled_overs * 6

            cumulative_runs = 0
            wickets_fallen = 0
            legal_balls_bowled = 0
            player_stats: dict[str, PlayerStats] = {}

            overs_data: list[dict[str, Any]] = inn_dict.get("overs", [])

            for over_dict in overs_data:
                over_num = over_dict.get("over", 0)
                deliveries: list[dict[str, Any]] = over_dict.get("deliveries", [])

                for deliv_idx, deliv in enumerate(deliveries, start=1):
                    batter = str(deliv.get("batter", ""))
                    bowler = str(deliv.get("bowler", ""))
                    non_striker = str(deliv.get("non_striker", ""))

                    runs_info = deliv.get("runs", {})
                    runs_batter = int(runs_info.get("batter", 0))
                    runs_extras = int(runs_info.get("extras", 0))
                    runs_total = int(runs_info.get("total", 0))

                    extras_dict = {str(k): int(v) for k, v in deliv.get("extras", {}).items()}

                    # Rule: Wides and no-balls do NOT count as legal balls
                    is_wide = "wides" in extras_dict
                    is_noball = "noballs" in extras_dict
                    is_legal = not (is_wide or is_noball)

                    if is_legal:
                        legal_balls_bowled += 1
                        ball_in_over = (legal_balls_bowled - 1) % 6 + 1
                    else:
                        ball_in_over = legal_balls_bowled % 6 + 1

                    wickets_list: list[dict[str, Any]] = deliv.get("wickets", [])
                    is_wicket = len(wickets_list) > 0
                    wicket_kind: str | None = None
                    player_dismissed: str | None = None
                    fielders: list[str] = []

                    if is_wicket:
                        w_info = wickets_list[0]
                        wickets_fallen += len(wickets_list)
                        wicket_kind = str(w_info.get("kind", "wicket"))
                        player_dismissed = str(w_info.get("player_out", batter))
                        fielders = [
                            f.get("name", str(f)) if isinstance(f, dict) else str(f)
                            for f in w_info.get("fielders", [])
                        ]

                    cumulative_runs += runs_total
                    balls_left = max(0, scheduled_legal_balls - legal_balls_bowled)
                    overs_completed = (legal_balls_bowled // 6) + (legal_balls_bowled % 6) / 10.0

                    # Run Rate Calculations
                    if legal_balls_bowled > 0:
                        crr = round(cumulative_runs / (legal_balls_bowled / 6.0), 2)
                    else:
                        crr = 0.0

                    if target_runs is not None:
                        runs_needed = target_runs - cumulative_runs
                        if runs_needed <= 0:
                            rrr: float | None = 0.0
                        elif balls_left > 0:
                            rrr = round(runs_needed / (balls_left / 6.0), 2)
                        else:
                            rrr = float("inf")
                    else:
                        rrr = None

                    # Update player stats
                    if batter:
                        curr_batter = player_stats.get(
                            batter, PlayerStats(player_id=batter, player_name=batter)
                        )
                        new_bf = curr_batter.balls_faced + (1 if is_legal else 0)
                        new_runs = curr_batter.runs_scored + runs_batter
                        sr = round((new_runs / new_bf) * 100, 2) if new_bf > 0 else 0.0
                        player_stats[batter] = PlayerStats(
                            player_id=batter,
                            player_name=batter,
                            runs_scored=new_runs,
                            balls_faced=new_bf,
                            wickets_taken=curr_batter.wickets_taken,
                            overs_bowled=curr_batter.overs_bowled,
                            strike_rate=sr,
                        )

                    if is_wicket and bowler:
                        curr_bowler = player_stats.get(
                            bowler, PlayerStats(player_id=bowler, player_name=bowler)
                        )
                        player_stats[bowler] = PlayerStats(
                            player_id=bowler,
                            player_name=bowler,
                            runs_scored=curr_bowler.runs_scored,
                            balls_faced=curr_bowler.balls_faced,
                            wickets_taken=curr_bowler.wickets_taken + 1,
                            overs_bowled=curr_bowler.overs_bowled,
                        )

                    ball = Ball(
                        ball_id=f"inn{inn_idx}_ov{over_num}_del{deliv_idx}",
                        match_id=self._current_state.match_id,
                        innings=inn_idx,
                        over=over_num,
                        ball_in_over=ball_in_over,
                        bowler=bowler,
                        batter=batter,
                        non_striker=non_striker,
                        runs_batter=runs_batter,
                        runs_extras=runs_extras,
                        runs_total=runs_total,
                        is_legal=is_legal,
                        is_wicket=is_wicket,
                        wicket_kind=wicket_kind,
                        player_dismissed=player_dismissed,
                        fielders=fielders,
                        extras_detail=extras_dict,
                        is_super_over=is_super_over,
                    )

                    # Check innings complete condition
                    is_inn_complete = (
                        wickets_fallen >= MAX_WICKETS_PER_INNINGS
                        or balls_left == 0
                        or (target_runs is not None and cumulative_runs >= target_runs)
                    )

                    # Update running state
                    self._current_state = MatchState(
                        match_id=self._current_state.match_id,
                        match_format=self._match_format,
                        home_team=self._teams[0] if self._teams else "",
                        away_team=self._teams[1] if len(self._teams) > 1 else "",
                        venue=str(self._info.get("venue", "")),
                        innings_idx=inn_idx,
                        batting_team=batting_team,
                        bowling_team=bowling_team,
                        score=cumulative_runs,
                        wickets=wickets_fallen,
                        overs_completed=overs_completed,
                        legal_balls_bowled=legal_balls_bowled,
                        balls_left=balls_left,
                        target=target_runs,
                        current_run_rate=crr,
                        required_run_rate=rrr,
                        is_super_over=is_super_over,
                        is_innings_complete=is_inn_complete,
                        is_match_complete=False,
                        winner=None,
                        win_margin=None,
                        outcome=outcome,
                        innings=inn_states_history,
                    )

                    # Configurable clock replay delay
                    await self._sleep()
                    yield ball

            # Innings ended, record in historical innings
            if inn_idx == 1:
                first_innings_score = cumulative_runs

            inn_states_history.append(
                InningsState(
                    innings_number=min(max(1, inn_idx), 4),
                    batting_team=batting_team,
                    bowling_team=bowling_team,
                    total_runs=cumulative_runs,
                    wickets_fallen=min(MAX_WICKETS_PER_INNINGS, wickets_fallen),
                    overs_completed=overs_completed,
                    current_over=legal_balls_bowled // 6,
                    current_ball=legal_balls_bowled % 6,
                    status=InningsStatus.COMPLETED,
                    player_stats=player_stats,
                )
            )

        # Mark final match state complete
        self._current_state = MatchState(
            match_id=self._current_state.match_id,
            match_format=self._match_format,
            home_team=self._teams[0] if self._teams else "",
            away_team=self._teams[1] if len(self._teams) > 1 else "",
            venue=str(self._info.get("venue", "")),
            innings_idx=self._current_state.innings_idx,
            batting_team=self._current_state.batting_team,
            bowling_team=self._current_state.bowling_team,
            score=self._current_state.score,
            wickets=self._current_state.wickets,
            overs_completed=self._current_state.overs_completed,
            legal_balls_bowled=self._current_state.legal_balls_bowled,
            balls_left=self._current_state.balls_left,
            target=self._current_state.target,
            current_run_rate=self._current_state.current_run_rate,
            required_run_rate=self._current_state.required_run_rate,
            is_super_over=self._current_state.is_super_over,
            is_innings_complete=True,
            is_match_complete=True,
            winner=winner,
            win_margin=win_margin,
            outcome=outcome,
            innings=inn_states_history,
            is_complete=True,
            result_summary=f"{winner} won by {win_margin}" if winner else "Match completed",
        )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    async def _sleep(self) -> None:
        """Execute sleep according to clock / speed multiplier settings."""
        self._ball_count += 1
        if self._sleep_fn is not None:
            delay = self._nominal_interval_s / self._speed_multiplier
            await self._sleep_fn(delay)
            return

        if (
            self._speed_multiplier >= INSTANT_REPLAY_SPEED_THRESHOLD
            or self._nominal_interval_s <= 0.0
        ):
            return

        nominal_step = self._nominal_interval_s / self._speed_multiplier
        target_time = self._start_time + (self._ball_count * nominal_step)
        delay = target_time - time.perf_counter()
        if delay >= MIN_SLEEP_THRESHOLD_S:
            await asyncio.sleep(delay)

    def _determine_format(self) -> MatchFormat:
        match_type = str(self._info.get("match_type", "")).lower()
        if "t20" in match_type:
            return MatchFormat.T20
        if "test" in match_type:
            return MatchFormat.TEST
        return MatchFormat.ODI

    def _build_rosters(self) -> dict[str, Roster]:
        players_data = self._info.get("players", {})
        rosters: dict[str, Roster] = {}
        for team in self._teams:
            player_list = [str(p) for p in players_data.get(team, [])]
            rosters[team] = Roster(team_name=team, players=player_list)
        return rosters

    def _build_metadata(self) -> dict[str, Any]:
        return {
            "teams": self._teams,
            "match_type": self._info.get("match_type", "ODI"),
            "overs": self._info.get("overs", 50),
            "venue": self._info.get("venue", ""),
            "dates": self._info.get("dates", []),
            "event": self._info.get("event", {}),
            "outcome": self._info.get("outcome", {}),
        }

    def _init_match_state(self) -> MatchState:
        match_id = str(self._info.get("match_type_number") or Path(self._source_path).stem)
        return MatchState(
            match_id=match_id,
            match_format=self._match_format,
            home_team=self._teams[0] if self._teams else "",
            away_team=self._teams[1] if len(self._teams) > 1 else "",
            venue=str(self._info.get("venue", "")),
            outcome=self._info.get("outcome", {}),
        )

    @staticmethod
    def _format_win_margin(outcome: dict[str, Any]) -> str:
        by = outcome.get("by", {})
        if "wickets" in by:
            return f"{by['wickets']} wickets"
        if "runs" in by:
            return f"{by['runs']} runs"
        if "result" in outcome:
            return str(outcome["result"])
        return ""


# ---------------------------------------------------------------------------
# CLI Command to replay a match to stdout
# ---------------------------------------------------------------------------


async def replay_to_stdout(file_path: str, speed: float = 1000.0) -> None:
    """Replay a match to stdout in real-time or fast-forward."""
    feed = CricsheetReplayFeed(file_path, speed_multiplier=speed, nominal_interval_s=0.5)

    meta = feed.metadata
    teams = meta.get("teams", ["Team 1", "Team 2"])
    print("=" * 70)
    print(f"CRICKET REPLAY: {teams[0]} vs {teams[1]}")
    print(
        f"Venue: {meta.get('venue')} | Format: {meta.get('match_type')} ({meta.get('overs')} overs)"
    )
    print("=" * 70)

    prev_over = -1
    async for ball in feed:
        st = feed.current_state
        if ball.over != prev_over:
            print(f"\n--- Over {ball.over + 1} (Batting: {st.batting_team}) ---")
            prev_over = ball.over

        w_str = f" [WICKET! {ball.player_dismissed} ({ball.wicket_kind})]" if ball.is_wicket else ""
        ext_str = f" (Extras: {ball.runs_extras})" if ball.runs_extras > 0 else ""
        rrr_str = f" | RRR: {st.required_run_rate:.2f}" if st.required_run_rate is not None else ""

        print(
            f"[{st.overs_completed:4.1f}] {ball.bowler} to {ball.batter}: "
            f"{ball.runs_total} run(s){ext_str}{w_str} | "
            f"Score: {st.score}/{st.wickets} (CRR: {st.current_run_rate:.2f}{rrr_str})"
        )

    final = feed.current_state
    print("\n" + "=" * 70)
    print("MATCH COMPLETE")
    print(f"Result: {final.result_summary}")
    for inn in final.innings:
        print(
            f"  {inn.batting_team}: {inn.total_runs}/{inn.wickets_fallen} ({inn.overs_completed:.1f} ov)"
        )
    print("=" * 70)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Replay Cricsheet JSON match to stdout")
    parser.add_argument("file", help="Path to Cricsheet JSON file")
    parser.add_argument(
        "--speed",
        type=float,
        default=1000.0,
        help="Replay speed multiplier (default: 1000x)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(replay_to_stdout(args.file, speed=args.speed))
    except KeyboardInterrupt:
        print("\nReplay stopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
