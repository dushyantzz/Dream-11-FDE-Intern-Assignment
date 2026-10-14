"""
LLM Gateway — Rule 2 enforcement.

This is the ONLY entry point for all LLM calls in the entire application.
No other module may import or call an LLM SDK directly.

Responsibilities:
- Route to the correct LLM provider (selected from config).
- Enforce per-request token budget and timeout.
- Record cost and latency metrics.
- Provide structured audit logging (Rule 6).
- Surface a uniform interface to callers regardless of provider.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from app.adapters.ports import LLMPort
from app.core.config import Settings

logger = structlog.get_logger(__name__)


class BudgetExceededError(Exception):
    """Raised when the estimated cost exceeds the per-request budget."""


class LLMTimeoutError(Exception):
    """Raised when the LLM call exceeds the configured timeout."""


class LLMGateway:
    """
    Single gateway for all LLM traffic.

    Selects the provider adapter from config, enforces budget + timeout,
    logs every call with cost and latency, and raises typed exceptions on
    failure so the engine pipeline can trigger the fallback ladder.

    Rule 2: All production LLM calls funnel through this class.
    """

    def __init__(self, settings: Settings, provider: LLMPort | None = None) -> None:
        self._settings = settings
        self._provider: LLMPort = provider or self._build_provider(settings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        *,
        override_model: str | None = None,
        override_budget_usd: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Send a prompt and return ``(response_text, metadata_dict)``.

        metadata_dict contains: model, prompt_tokens, completion_tokens,
        cost_usd, latency_ms.

        Raises:
            BudgetExceededError: if estimated cost exceeds budget.
            LLMTimeoutError: if the call exceeds llm.timeout_s.
            Exception: re-raised from the underlying provider.
        """
        llm_cfg = self._settings.llm
        model = override_model or llm_cfg.model
        budget = override_budget_usd if override_budget_usd is not None else llm_cfg.budget_usd
        timeout_s = llm_cfg.timeout_s

        # Rough prompt-token estimate (4 chars ≈ 1 token)
        prompt_tokens_est = max(1, len(prompt) // 4)
        cost_est = self._provider.estimate_cost_usd(
            prompt_tokens=prompt_tokens_est,
            completion_tokens=llm_cfg.max_tokens,
            model=model,
        )

        if cost_est > budget:
            logger.warning(
                "llm_budget_exceeded",
                cost_est_usd=cost_est,
                budget_usd=budget,
                model=model,
            )
            raise BudgetExceededError(
                f"Estimated cost ${cost_est:.6f} exceeds budget ${budget:.6f}"
            )

        start = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._provider.complete(
                    prompt,
                    model=model,
                    max_tokens=llm_cfg.max_tokens,
                    temperature=llm_cfg.temperature,
                ),
                timeout=timeout_s,
            )
        except TimeoutError as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error(
                "llm_timeout",
                timeout_s=timeout_s,
                elapsed_ms=round(elapsed_ms, 1),
                model=model,
            )
            raise LLMTimeoutError(f"LLM call timed out after {timeout_s}s") from exc

        elapsed_ms = (time.monotonic() - start) * 1000
        completion_tokens_est = max(1, len(response) // 4)
        actual_cost = self._provider.estimate_cost_usd(
            prompt_tokens=prompt_tokens_est,
            completion_tokens=completion_tokens_est,
            model=model,
        )

        logger.info(
            "llm_call_complete",
            model=model,
            prompt_tokens_est=prompt_tokens_est,
            completion_tokens_est=completion_tokens_est,
            cost_usd=round(actual_cost, 6),
            latency_ms=round(elapsed_ms, 1),
        )

        metadata: dict[str, Any] = {
            "model": model,
            "prompt_tokens": prompt_tokens_est,
            "completion_tokens": completion_tokens_est,
            "cost_usd": actual_cost,
            "latency_ms": elapsed_ms,
        }
        return response, metadata

    # ------------------------------------------------------------------
    # Provider factory
    # ------------------------------------------------------------------

    @staticmethod
    def _build_provider(settings: Settings) -> LLMPort:
        """Instantiate the correct provider adapter from config."""
        provider_name = settings.llm.provider
        if provider_name == "fake":
            from app.adapters.fake.llm import FakeLLMAdapter

            return FakeLLMAdapter()

        # Real providers would be imported here inside app/llm/providers/
        raise ValueError(
            f"Unknown LLM provider: {provider_name!r}. "
            "Add a concrete LLMPort implementation in app/llm/providers/."
        )
