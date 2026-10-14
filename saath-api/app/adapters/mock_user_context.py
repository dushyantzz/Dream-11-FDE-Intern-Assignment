"""
MockUserContextProvider — in-memory implementation of UserContextProvider.

Includes 3 sample users with fantasy picks, preferred persona, and preferred language.
"""

from __future__ import annotations

from app.domain.models import UserContext

SAMPLE_USERS: dict[str, UserContext] = {
    "usr_analyst": UserContext(
        user_id="usr_analyst",
        username="StatsSavvy",
        fantasy_picks=["Rohit Sharma", "Virat Kohli", "Jasprit Bumrah"],
        preferred_persona="analyst",
        preferred_language="en",
        favourite_team="India",
        favourite_player_ids=["ply_rohit", "ply_virat"],
        fantasy_team_player_ids=["ply_rohit", "ply_virat", "ply_bumrah"],
        language_preference="en",
        tone_preference="analytical",
    ),
    "usr_roaster": UserContext(
        user_id="usr_roaster",
        username="BanterBoss",
        fantasy_picks=["Babar Azam", "Shaheen Afridi", "Mohammad Rizwan"],
        preferred_persona="roaster",
        preferred_language="hi",
        favourite_team="Pakistan",
        favourite_player_ids=["ply_babar"],
        fantasy_team_player_ids=["ply_babar", "ply_shaheen"],
        language_preference="hi",
        tone_preference="enthusiastic",
    ),
    "usr_fanatic": UserContext(
        user_id="usr_fanatic",
        username="CricketDost",
        fantasy_picks=["Hardik Pandya", "Suryakumar Yadav", "KL Rahul"],
        preferred_persona="enthusiastic",
        preferred_language="hinglish",
        favourite_team="Mumbai Indians",
        favourite_player_ids=["ply_hardik"],
        fantasy_team_player_ids=["ply_hardik", "ply_surya", "ply_kl"],
        language_preference="hinglish",
        tone_preference="enthusiastic",
    ),
}


class MockUserContextProvider:
    """Provides user context with fantasy picks, persona, and language preferences."""

    def __init__(self, users: dict[str, UserContext] | None = None) -> None:
        self._users = dict(SAMPLE_USERS if users is None else users)

    async def get_user_context(self, user_id: str) -> UserContext:
        """Fetch user context by user_id or return a default user profile."""
        if user_id in self._users:
            return self._users[user_id]

        # Default fallback user profile
        return UserContext(
            user_id=user_id,
            username=f"User_{user_id}",
            fantasy_picks=["Rohit Sharma", "Jasprit Bumrah"],
            preferred_persona="analyst",
            preferred_language="en",
            favourite_team="Generic Team",
            fantasy_team_player_ids=["p001", "p002"],
        )
