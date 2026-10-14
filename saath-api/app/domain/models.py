"""
Domain models — pure Pydantic data classes representing the match world.

These are shared across engine, adapters, and API layers.
All models are frozen Pydantic v2 models (Rule 1 & Rule 5 compliance).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class MomentType(StrEnum):
    WICKET = "wicket"
    BOUNDARY = "boundary"
    SIX = "six"
    MILESTONE = "milestone"
    OVER_END = "over_end"
    INNINGS_END = "innings_end"
    MATCH_START = "match_start"
    MATCH_END = "match_end"
    GENERIC = "generic"


class MatchFormat(StrEnum):
    T20 = "t20"
    ODI = "odi"
    TEST = "test"


class InningsStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    NOT_STARTED = "not_started"


# ---------------------------------------------------------------------------
# Core Domain Models
# ---------------------------------------------------------------------------


class PlayerStats(BaseModel):
    """Aggregate stats for an individual player."""

    model_config = ConfigDict(frozen=True)

    player_id: str
    player_name: str
    runs_scored: int = 0
    balls_faced: int = 0
    wickets_taken: int = 0
    overs_bowled: float = 0.0
    catches: int = 0
    fantasy_points: float = 0.0
    strike_rate: float = 0.0
    economy_rate: float = 0.0


class InningsState(BaseModel):
    """Snapshot of a single innings state."""

    model_config = ConfigDict(frozen=True)

    innings_number: int = Field(ge=1, le=4)
    batting_team: str
    bowling_team: str
    total_runs: int = Field(default=0, ge=0)
    wickets_fallen: int = Field(default=0, ge=0, le=10)
    overs_completed: float = Field(default=0.0, ge=0.0)
    current_over: int = Field(default=0, ge=0)
    current_ball: int = Field(default=0, ge=0, le=6)
    status: InningsStatus = InningsStatus.NOT_STARTED
    player_stats: dict[str, PlayerStats] = Field(default_factory=dict)


class Ball(BaseModel):
    """Represents a single delivery event in a cricket match."""

    model_config = ConfigDict(frozen=True)

    ball_id: str
    match_id: str
    innings: int
    over: int
    ball_in_over: int
    bowler: str
    batter: str
    non_striker: str
    runs_batter: int
    runs_extras: int
    runs_total: int
    is_legal: bool
    is_wicket: bool = False
    wicket_kind: str | None = None
    player_dismissed: str | None = None
    fielders: list[str] = Field(default_factory=list)
    extras_detail: dict[str, int] = Field(default_factory=dict)
    is_super_over: bool = False
    timestamp: datetime | None = None


class MatchState(BaseModel):
    """
    Running state of a cricket match per ball.

    Contains live scoring, balls left, target, CRR, RRR, and match outcome.
    """

    model_config = ConfigDict(frozen=True)

    match_id: str
    match_format: MatchFormat = MatchFormat.ODI
    home_team: str = ""
    away_team: str = ""
    venue: str = ""
    match_datetime: datetime | None = None

    # Running state per ball
    innings_idx: int = 1
    batting_team: str = ""
    bowling_team: str = ""
    score: int = 0
    wickets: int = 0
    overs_completed: float = 0.0
    legal_balls_bowled: int = 0
    balls_left: int = 0
    target: int | None = None
    current_run_rate: float = 0.0
    required_run_rate: float | None = None
    is_super_over: bool = False
    is_innings_complete: bool = False
    is_match_complete: bool = False
    winner: str | None = None
    win_margin: str | None = None
    outcome: dict[str, Any] = Field(default_factory=dict)

    # Historical innings states for backward compatibility
    innings: list[InningsState] = Field(default_factory=list)
    is_complete: bool = False
    result_summary: str = ""


class Roster(BaseModel):
    """Team roster containing player names and key roles."""

    model_config = ConfigDict(frozen=True)

    team_name: str
    players: list[str] = Field(default_factory=list)
    captain: str | None = None
    wicket_keeper: str | None = None


class MomentEvent(BaseModel):
    """A discrete in-match event that triggers commentary generation."""

    model_config = ConfigDict(frozen=True)

    moment_id: str
    match_id: str
    moment_type: MomentType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    primary_player_id: str | None = None
    secondary_player_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Slot(BaseModel):
    """Verified data slot item for Rule 1 template filling."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: str | int | float
    source: str = "verified"


class Reaction(BaseModel):
    """Personalized commentary reaction generated for a specific user."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    moment_id: str
    persona: str = "analyst"
    language: str = "en"
    commentary_text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class UserContext(BaseModel):
    """User-specific context for personalizing commentary and insights."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    username: str = ""
    fantasy_picks: list[str] = Field(default_factory=list)
    preferred_persona: str = "analyst"
    preferred_language: str = "en"

    # Backward compatibility attributes
    favourite_team: str | None = None
    favourite_player_ids: list[str] = Field(default_factory=list)
    fantasy_team_player_ids: list[str] = Field(default_factory=list)
    language_preference: str = "en"
    tone_preference: str = "enthusiastic"


class CostRecord(BaseModel):
    """Audit and billing record for LLM execution cost."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SlotMap(BaseModel):
    """Verified data slots to fill into an LLM template. Rule 1 enforcement."""

    model_config = ConfigDict(frozen=True)

    slots: dict[str, str | int | float] = Field(default_factory=dict)

    def get(self, key: str, default: str = "") -> str:
        """Return slot value as string."""
        return str(self.slots.get(key, default))

    def keys(self) -> set[str]:
        return set(self.slots.keys())


class CommentaryResponse(BaseModel):
    """Final rendered commentary returned to API callers."""

    model_config = ConfigDict(frozen=True)

    moment_id: str
    match_id: str
    commentary_text: str
    moment_type: MomentType
    is_fallback: bool = False
    model_used: str | None = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
