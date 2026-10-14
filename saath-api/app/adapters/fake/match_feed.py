"""
FakeMatchFeedAdapter — deterministic match-state fixture.

Rule 7: Returns a hardcoded MatchState so engine tests have stable input.
"""

from __future__ import annotations

from datetime import datetime

from app.domain.models import (
    InningsState,
    InningsStatus,
    MatchFormat,
    MatchState,
    MomentEvent,
    MomentType,
    PlayerStats,
)

_DEFAULT_MATCH = MatchState(
    match_id="match-001",
    match_format=MatchFormat.T20,
    home_team="Mumbai Indians",
    away_team="Chennai Super Kings",
    venue="Wankhede Stadium",
    match_datetime=datetime(2024, 4, 14, 19, 30),
    innings=[
        InningsState(
            innings_number=1,
            batting_team="Mumbai Indians",
            bowling_team="Chennai Super Kings",
            total_runs=185,
            wickets_fallen=4,
            overs_completed=20.0,
            current_over=20,
            current_ball=0,
            status=InningsStatus.COMPLETED,
            player_stats={
                "p001": PlayerStats(
                    player_id="p001",
                    player_name="Rohit Sharma",
                    runs_scored=72,
                    balls_faced=48,
                    strike_rate=150.0,
                    fantasy_points=42.0,
                ),
                "p002": PlayerStats(
                    player_id="p002",
                    player_name="Jasprit Bumrah",
                    wickets_taken=0,
                    overs_bowled=0.0,
                    economy_rate=0.0,
                    fantasy_points=0.0,
                ),
            },
        )
    ],
)

_DEFAULT_EVENTS: list[MomentEvent] = [
    MomentEvent(
        moment_id="moment-001",
        match_id="match-001",
        moment_type=MomentType.BOUNDARY,
        primary_player_id="p001",
        metadata={"runs": 4, "shot": "pull"},
    ),
    MomentEvent(
        moment_id="moment-002",
        match_id="match-001",
        moment_type=MomentType.SIX,
        primary_player_id="p001",
        metadata={"runs": 6, "shot": "slog-sweep"},
    ),
]


class FakeMatchFeedAdapter:
    """
    Returns hardcoded match fixtures.

    Accepts an optional override ``match`` and ``events`` for per-test control.
    """

    def __init__(
        self,
        match: MatchState | None = None,
        events: list[MomentEvent] | None = None,
    ) -> None:
        self._match = match or _DEFAULT_MATCH
        self._events = events or _DEFAULT_EVENTS

    async def get_match_state(self, match_id: str) -> MatchState:
        return self._match

    async def get_recent_events(self, match_id: str, limit: int) -> list[MomentEvent]:
        return self._events[:limit]
