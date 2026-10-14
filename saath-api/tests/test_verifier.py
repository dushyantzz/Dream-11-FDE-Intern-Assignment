"""
Tests for slot verifier — Rule 1 enforcement.
"""

from __future__ import annotations

import pytest

from app.domain.models import SlotMap
from app.engine.verifier import (
    SlotVerificationError,
    check_for_bare_numbers,
    verify_and_render,
)


@pytest.mark.unit
def test_verify_and_render_fills_all_slots() -> None:
    """All slot tokens must be replaced with verified data."""
    template = "{player_name} scored {runs} runs off {balls} balls."
    slot_map = SlotMap(slots={"player_name": "Test Player", "runs": 45, "balls": 30})
    result = verify_and_render(template, slot_map)
    assert result == "Test Player scored 45 runs off 30 balls."
    assert "{" not in result


@pytest.mark.unit
def test_verify_strict_raises_on_missing_slot() -> None:
    """In strict mode, a missing slot must raise SlotVerificationError."""
    template = "{player_name} hit {undefined_slot}."
    slot_map = SlotMap(slots={"player_name": "Player"})
    with pytest.raises(SlotVerificationError) as exc_info:
        verify_and_render(template, slot_map, strict=True)
    assert "undefined_slot" in exc_info.value.missing_slots


@pytest.mark.unit
def test_verify_non_strict_skips_missing_slot() -> None:
    """In non-strict mode, missing slots are skipped (token left as-is)."""
    template = "{player_name} hit {unknown_slot}."
    slot_map = SlotMap(slots={"player_name": "Player"})
    result = verify_and_render(template, slot_map, strict=False)
    assert "Player" in result


@pytest.mark.unit
def test_check_for_bare_numbers_clean_template() -> None:
    """A clean template with only slot tokens should have no bare numbers."""
    template = "{player_name} scored {runs} off {balls}."
    assert check_for_bare_numbers(template) == []


@pytest.mark.unit
def test_check_for_bare_numbers_detects_violations() -> None:
    """Raw numbers outside slot tokens are Rule 1 violations."""
    template = "Player scored 45 runs."
    bare = check_for_bare_numbers(template)
    assert len(bare) > 0
