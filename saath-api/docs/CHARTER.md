# saath-api — Engineering Charter

> **Version:** 1.0.0
> **Status:** BINDING — Every pull request is reviewed against every rule below.
> A PR that violates any rule MUST NOT be merged until the violation is resolved.

---

## Purpose

`saath-api` is a real-time commentary & insight engine for fantasy-sports moments.
It pairs a structured domain model (match state, player statistics, live events) with
an LLM layer to produce contextual, personalized, and factually-grounded text.

The charter below defines **seven non-negotiable architectural rules** that govern every
line of code committed to this repository.

---

## Rule 1 — The LLM emits templates, not facts

**The LLM MUST NEVER emit numeric facts, player names, or team names directly.**

Every piece of verified data (scores, run counts, player averages, odds, etc.) is
represented inside an LLM prompt as a **slot token** in the form `{slot_name}`.
The LLM output itself will contain those same tokens verbatim.
`app/engine/` code fills the slots from verified, type-checked domain objects **after**
the LLM has responded.

### Rationale
LLMs hallucinate numbers. By separating "text shape" from "data values" we make
hallucination structurally impossible: the rendered string can only contain facts that
passed through our data pipeline.

### Enforcement
- `app/llm/gateway.py` MUST strip any response that contains digits not wrapped in
  a `{...}` token before returning it to the caller.
- The `SlotVerifier` in `app/engine/verifier.py` MUST confirm every token in the
  template exists in the slot map before rendering.
- Tests MUST assert that final rendered strings match the slot-filled domain data, not
  the raw LLM string.

---

## Rule 2 — One gateway for all LLM traffic

**Every call to any LLM provider goes through `app/llm/gateway.py` — no exceptions.**

No file outside `app/llm/` may import an LLM SDK directly.

### Rationale
A single gateway enables: cost accounting, rate-limit enforcement, per-request budgets,
audit logging, fallback orchestration, and provider swapping — all in one place.

### Enforcement
- `ruff` rule `TID252` (banned imports) in `pyproject.toml` bans direct SDK imports
  outside `app/llm/`.
- All LLM adapter implementations live in `app/llm/providers/` and are accessed only
  through the `LLMGateway` interface.

---

## Rule 3 — All external dependencies behind adapters

Every external system is accessed **only** through an interface defined in `app/adapters/`.

| External system | Interface |
|---|---|
| Match / live feed | `MatchFeedPort` |
| User context service | `UserContextPort` |
| LLM provider(s) | `LLMPort` (consumed by gateway) |
| Cache (Redis, etc.) | `CachePort` |
| Feature store | `FeatureStorePort` |

### Enforcement
- `app/engine/` MUST only import from `app/domain/` and `app/adapters/` (interfaces).
- Concrete adapter implementations live in `app/adapters/impl/`.
- The DI container in `app/core/container.py` is the **only** place concrete adapters
  are instantiated.

---

## Rule 4 — Graceful degradation on every request

**The HTTP stream MUST NEVER stall, raise an unhandled 5xx, or return an empty body.**

The degradation ladder (tried top-to-bottom):
1. Full LLM path (slot-filled template from live LLM response).
2. Cached template from a previous successful call.
3. Deterministic fallback template from `app/engine/templates/fallback.yaml`.

### Enforcement
- `app/engine/pipeline.py` MUST be wrapped in a try/except that logs the exception
  and calls `FallbackRenderer.render()` on any exception.
- Integration tests MUST assert that injecting a `BrokenLLM` adapter still returns
  HTTP 200 with a non-empty body.

---

## Rule 5 — No magic numbers; all tunables in config

**Every numeric threshold, weight, model name, price constant, and timeout is declared
in `config.yaml` and overridable via environment variables.**

`pydantic-settings` validates the merged config at startup.

### Enforcement
- `ruff` rule `PLR2004` (magic value comparison) is enabled.
- `mypy --strict` is enforced; every config value has an explicit annotated type.

---

## Rule 6 — Structured JSON logging with correlation IDs

**Every log line MUST be valid JSON and MUST contain:**
`timestamp`, `level`, `logger`, `request_id`, `match_id`, `moment_id`, `message`.

### Enforcement
- `app/core/logging.py` configures structlog and is imported first in `app/main.py`.
- No `print()` statements anywhere in `app/` (ruff rule `T201`).

---

## Rule 7 — Deterministic offline testing with FakeLLM

**Every piece of engine logic MUST be testable without a real LLM or network.**

`app/adapters/fake/` contains `FakeLLMAdapter`, `FakeMatchFeedAdapter`,
`FakeUserContextAdapter`, `FakeCacheAdapter`.

### Enforcement
- `make test` MUST pass with zero network calls.
- Coverage gate: `pytest --cov` MUST report >= 80% line coverage on `app/`.

---

## Glossary

| Term | Definition |
|---|---|
| **Slot token** | A `{snake_case}` placeholder filled with verified data post-LLM |
| **Moment** | A discrete in-match event triggering commentary |
| **Adapter** | A concrete implementation of a port interface |
| **Port** | An abstract interface (Protocol/ABC) in `app/adapters/` |
| **Gateway** | The single LLM entry-point `app/llm/gateway.py` |
| **Fallback** | A deterministic non-LLM path that always returns a valid response |
| **Budget** | Per-request USD cost cap enforced by the gateway |
