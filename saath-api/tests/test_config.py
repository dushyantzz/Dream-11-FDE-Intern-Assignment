"""
Tests for app/core/config.py — Rule 5 validation.

Asserts that Settings() loads from the test environment without errors.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import ApiSettings, LLMSettings, Settings


@pytest.mark.unit
def test_settings_defaults_are_valid(fake_settings: Settings) -> None:
    """Settings with fake/default values must pass pydantic validation."""
    assert fake_settings.llm.provider == "fake"
    assert fake_settings.cache.backend == "memory"
    assert fake_settings.api.port > 0
    assert fake_settings.llm.budget_usd > 0
    assert fake_settings.llm.timeout_s > 0


@pytest.mark.unit
def test_llm_settings_temperature_bounds() -> None:
    with pytest.raises(ValidationError):
        LLMSettings(temperature=3.0)


@pytest.mark.unit
def test_api_settings_port_bounds() -> None:
    with pytest.raises(ValidationError):
        ApiSettings(port=0)


@pytest.mark.unit
def test_logging_level_validation() -> None:
    from app.core.config import LoggingSettings

    with pytest.raises(ValidationError):
        LoggingSettings(level="VERBOSE")
