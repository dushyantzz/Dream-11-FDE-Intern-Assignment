"""
FallbackRenderer — Rule 4 enforcement.

When the LLM path fails (provider error, timeout, budget, or slot verification
failure), the pipeline calls FallbackRenderer.render() to produce a
deterministic, data-filled response that is always valid.
"""

from __future__ import annotations

from pathlib import Path

import structlog
import yaml

from app.domain.models import MomentType, SlotMap

logger = structlog.get_logger(__name__)

_DEFAULT_TEMPLATES: dict[str, str] = {
    MomentType.WICKET: "{player_name} is OUT! {bowling_team} strikes back. {wickets} wickets down.",
    MomentType.BOUNDARY: "FOUR! {player_name} finds the gap. Score: {runs}/{wickets}.",
    MomentType.SIX: "SIX! {player_name} goes big! {runs}/{wickets} after {overs} overs.",
    MomentType.MILESTONE: "{player_name} reaches {milestone} — a brilliant knock!",
    MomentType.OVER_END: "End of over {over_number}. {batting_team} needs {runs_required} from {balls_remaining}.",
    MomentType.INNINGS_END: "{batting_team} finish at {runs}/{wickets} in {overs} overs.",
    MomentType.MATCH_START: "{home_team} vs {away_team} — play begins at {venue}!",
    MomentType.MATCH_END: "{winning_team} win the match. Final score: {result_summary}.",
    MomentType.GENERIC: "Live from {venue}: {batting_team} vs {bowling_team}.",
}


class FallbackRenderer:
    """
    Renders deterministic commentary from YAML templates + slot map.

    The fallback is guaranteed to succeed: if a slot is missing in the
    provided map, it uses a safe default placeholder.
    """

    def __init__(self, template_path: str | None = None) -> None:
        self._templates: dict[str, str] = dict(_DEFAULT_TEMPLATES)
        if template_path:
            self._load_yaml(template_path)

    def _load_yaml(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.warning("fallback_template_file_not_found", path=path)
            return
        with p.open() as fh:
            data: dict[str, str] = yaml.safe_load(fh) or {}
        self._templates.update(data)
        logger.info("fallback_templates_loaded", path=path, count=len(data))

    def render(self, moment_type: MomentType, slot_map: SlotMap) -> str:
        """
        Return a fully rendered fallback string.

        Guaranteed to return a non-empty string — never raises.
        """
        template = self._templates.get(str(moment_type), self._templates[MomentType.GENERIC])

        # Safe fill: missing slots become "<slot_name>"
        import re

        def _fill(m: re.Match[str]) -> str:
            key = m.group(1)
            return slot_map.get(key, f"<{key}>")

        result = re.sub(r"\{([a-z][a-z0-9_]*)\}", _fill, template)
        logger.info("fallback_rendered", moment_type=moment_type, length=len(result))
        return result
