"""
FakeUserContextAdapter — deterministic user context fixture.

Rule 7: Returns a hardcoded UserContext for hermetic tests.
"""

from __future__ import annotations

from app.domain.models import UserContext

_DEFAULT_USER = UserContext(
    user_id="user-001",
    favourite_team="Mumbai Indians",
    favourite_player_ids=["p001"],
    fantasy_team_player_ids=["p001", "p002"],
    language_preference="en",
    tone_preference="enthusiastic",
)


class FakeUserContextAdapter:
    """Returns a hardcoded UserContext fixture."""

    def __init__(self, user: UserContext | None = None) -> None:
        self._user = user or _DEFAULT_USER

    async def get_user_context(self, user_id: str) -> UserContext:
        return self._user
