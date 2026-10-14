"""
FakeLLMAdapter — deterministic test double for the LLM provider.

Rule 7: All engine tests use this adapter; no real LLM calls in unit tests.

By default the adapter returns a configurable template string containing
slot tokens so that the SlotVerifier and renderer can be exercised.
"""

from __future__ import annotations


class FakeLLMAdapter:
    """
    Configurable LLM test double.

    Usage::

        adapter = FakeLLMAdapter(response="{player_name} hit {runs} runs!")
        result = await adapter.complete(prompt="...", model="fake", ...)
    """

    def __init__(
        self,
        response: str = "{player_name} scored {runs} off {balls} balls!",
        raise_on_call: Exception | None = None,
        call_count: int = 0,
    ) -> None:
        self._response = response
        self._raise_on_call = raise_on_call
        self.call_count = call_count

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Return the configured response or raise the configured exception."""
        self.call_count += 1
        if self._raise_on_call is not None:
            raise self._raise_on_call
        return self._response

    def estimate_cost_usd(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> float:
        """Always returns zero cost for fakes."""
        return 0.0


class BrokenLLMAdapter(FakeLLMAdapter):
    """Always raises a RuntimeError — used to test fallback paths."""

    def __init__(self) -> None:
        super().__init__(raise_on_call=RuntimeError("Injected LLM failure for testing"))
