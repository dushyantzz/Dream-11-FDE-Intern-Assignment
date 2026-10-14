"""
Pydantic-settings configuration.

All tunables are loaded from config.yaml first, then overridden by environment
variables using the SAATH__ prefix with __ as a nested separator.

Rule 5 enforcement: No magic numbers in application code — everything lives here.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Sub-settings models
# ---------------------------------------------------------------------------


class ApiSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    p99_latency_ms: int = Field(default=2000, gt=0)
    request_timeout_s: float = Field(default=10.0, gt=0)


class LLMPriceEntry(BaseModel):
    input: float = Field(gt=0)
    output: float = Field(gt=0)


class LLMSettings(BaseModel):
    provider: str = "fake"
    model: str = "gpt-4o-mini"
    timeout_s: float = Field(default=8.0, gt=0)
    budget_usd: float = Field(default=0.01, gt=0)
    max_tokens: int = Field(default=512, gt=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    prices: dict[str, LLMPriceEntry] = Field(default_factory=dict)


class VerifierSettings(BaseModel):
    strict: bool = True


class CacheSettings(BaseModel):
    backend: str = "memory"
    redis_url: str = "redis://localhost:6379/0"
    ttl_s: int = Field(default=300, gt=0)
    max_memory_items: int = Field(default=1000, gt=0)


class MatchFeedSettings(BaseModel):
    backend: str = "fake"
    base_url: str = "https://api.example.com/v1"
    timeout_s: float = Field(default=3.0, gt=0)
    api_key: str = ""


class UserContextSettings(BaseModel):
    backend: str = "fake"
    base_url: str = "https://users.example.com/v1"
    timeout_s: float = Field(default=2.0, gt=0)


class EngineSettings(BaseModel):
    fallback_template_path: str = "app/engine/templates/fallback.yaml"
    max_retry_attempts: int = Field(default=1, ge=0)


class LoggingSettings(BaseModel):
    level: str = "INFO"
    format: str = "json"
    render_json_logs: bool = True

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"logging.level must be one of {valid}")
        return v.upper()

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        valid = {"json", "console"}
        if v.lower() not in valid:
            raise ValueError(f"logging.format must be one of {valid}")
        return v.lower()


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


def _load_yaml_config(path: str) -> dict[str, Any]:
    """Load config.yaml; return empty dict if file not found."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as fh:
        return yaml.safe_load(fh) or {}


class Settings(BaseSettings):
    """
    Root settings object.

    Loaded from config.yaml first; env vars (SAATH__*) override individual fields.
    """

    model_config = SettingsConfigDict(
        env_prefix="SAATH__",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    config_path: str = "config.yaml"

    api: ApiSettings = Field(default_factory=ApiSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    verifier: VerifierSettings = Field(default_factory=VerifierSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    match_feed: MatchFeedSettings = Field(default_factory=MatchFeedSettings)
    user_context: UserContextSettings = Field(default_factory=UserContextSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @classmethod
    def from_yaml_and_env(cls) -> Settings:
        """Load settings merging YAML file + environment variable overrides."""
        config_path = os.environ.get("SAATH__CONFIG_PATH", "config.yaml")
        yaml_data = _load_yaml_config(config_path)
        # pydantic-settings will also pick up env vars on top
        return cls(**yaml_data)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton settings loader — call everywhere."""
    return Settings.from_yaml_and_env()
