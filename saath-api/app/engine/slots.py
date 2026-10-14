"""
Slots engine — Rule 1 and Rule 4 enforcement.

Transforms verified moment facts and typed tool results into a strict SlotMap.
Generates explicit slot allowlists for prompts and renders templates safely,
raising on unknown or missing tokens.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.domain.models import MomentEvent

_SLOT_TOKEN_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")


class SlotValue(BaseModel):
    """A verified slot value with type and provenance tracking."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: str | int | float | bool
    type: str
    source: str


class SlotRenderError(ValueError):
    """Raised when template rendering encounters unknown or missing slots."""

    def __init__(self, message: str, missing_token: str | None = None) -> None:
        super().__init__(message)
        self.missing_token = missing_token


def build_slots(
    moment_event: MomentEvent,
    tool_results: dict[str, BaseModel | None],
) -> dict[str, SlotValue]:
    """
    Construct a verified slot dictionary from moment facts and tool outputs.

    Crucial rule: If a tool returns None, its slots are omitted entirely,
    removing them from the allowlist rather than inventing a placeholder.
    """
    slots: dict[str, SlotValue] = {}

    # 1. Ingest verified_facts from MomentEvent
    facts = moment_event.metadata.get("verified_facts", {})
    for k, v in facts.items():
        if v is not None and isinstance(v, (str, int, float, bool)):
            slots[k] = SlotValue(
                name=k,
                value=v,
                type=type(v).__name__,
                source=f"verified_facts:{k}",
            )

    # 2. Ingest typed tool results
    for tool_name, tool_model in tool_results.items():
        if tool_model is None:
            # Tool returned None: do NOT invent any slots
            continue

        model_dict = tool_model.model_dump()
        for field_name, val in model_dict.items():
            if val is None or not isinstance(val, (str, int, float, bool)):
                continue

            type_name = type(val).__name__

            # Specific scoped slot name (e.g. required_rate_runs_needed)
            scoped_key = f"{tool_name}_{field_name}"
            slots[scoped_key] = SlotValue(
                name=scoped_key,
                value=val,
                type=type_name,
                source=f"tool:{tool_name}.{field_name}",
            )

            # Natural field alias if not already taken
            if field_name not in slots:
                slots[field_name] = SlotValue(
                    name=field_name,
                    value=val,
                    type=type_name,
                    source=f"tool:{tool_name}.{field_name}",
                )

    return slots


def get_slot_allowlist(slots: dict[str, SlotValue]) -> list[str]:
    """
    Generate the sorted allowlist of valid slot names for this moment.

    Only names present in this list may be used by the prompt/LLM.
    """
    return sorted(slots.keys())


def render(text: str, slots: dict[str, SlotValue] | dict[str, Any]) -> str:
    """
    Substitute slot tokens in text using verified slots.

    Raises:
        SlotRenderError: If text references any slot token not present in slots
                         or if the slot value is None.
    """
    tokens = _SLOT_TOKEN_RE.findall(text)

    # Check for any missing or undefined slot tokens
    for token in tokens:
        if token not in slots:
            raise SlotRenderError(
                f"Unknown slot token {token!r}. Available slots: {sorted(slots.keys())}",
                missing_token=token,
            )

        val_obj = slots[token]
        raw_val = val_obj.value if isinstance(val_obj, SlotValue) else val_obj
        if raw_val is None:
            raise SlotRenderError(
                f"Slot token {token!r} has null value in slot map.",
                missing_token=token,
            )

    rendered = text
    for token in set(tokens):
        val_obj = slots[token]
        raw_val = val_obj.value if isinstance(val_obj, SlotValue) else val_obj
        rendered = rendered.replace(f"{{{token}}}", str(raw_val))

    return rendered
