"""
SlotVerifier — Rule 1 enforcement.

Ensures every slot token present in an LLM-generated template exists in the
verified SlotMap before rendering. Also validates that the LLM has not
emitted raw numeric facts outside of slot tokens.
"""

from __future__ import annotations

import re

import structlog

from app.domain.models import SlotMap

logger = structlog.get_logger(__name__)

# Regex to find {slot_name} tokens in template strings
_SLOT_RE = re.compile(r"\{([a-z][a-z0-9_]*)\}")

# Regex to detect numeric literals
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


class SlotVerificationError(Exception):
    """Raised when the LLM template fails slot verification."""

    def __init__(self, message: str, missing_slots: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing_slots = missing_slots or []


def verify_and_render(
    template: str,
    slot_map: SlotMap,
    *,
    strict: bool = True,
) -> str:
    """
    Verify all slot tokens in ``template`` are present in ``slot_map``,
    then render the template by filling slots from the verified data.

    Args:
        template: Raw LLM output containing ``{slot_name}`` tokens.
        slot_map: Verified data values keyed by slot name.
        strict: If True, raise on missing slots. If False, skip them.

    Returns:
        Fully rendered string with all slots filled.

    Raises:
        SlotVerificationError: In strict mode, if any slot is missing.
    """
    found_slots = _SLOT_RE.findall(template)
    missing = [s for s in found_slots if s not in slot_map.keys()]

    if missing:
        logger.warning(
            "slot_verification_missing",
            missing_slots=missing,
            available_slots=list(slot_map.keys()),
            strict=strict,
        )
        if strict:
            raise SlotVerificationError(
                f"Template references undefined slots: {missing}",
                missing_slots=missing,
            )

    # Fill all available slots; leave unknown ones as-is if not strict
    result = template
    for slot_name in found_slots:
        if slot_name in slot_map.keys():
            result = result.replace(f"{{{slot_name}}}", slot_map.get(slot_name))

    logger.debug("slot_verification_ok", rendered_length=len(result))
    return result


def check_for_bare_numbers(text: str) -> list[str]:
    """
    Return a list of bare numeric literals found outside slot tokens.

    An empty list means the text is clean (Rule 1 compliant).
    """
    text_without_slots = _SLOT_RE.sub("", text)
    return _NUMBER_RE.findall(text_without_slots)
