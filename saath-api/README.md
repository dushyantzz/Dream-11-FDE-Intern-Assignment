# saath-api

> **Real-time commentary & insight engine for fantasy-sports moments.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## Quick Start

```bash
# 1. Install dependencies (requires uv)
make install

# 2. Copy env template
cp .env.example .env

# 3. Run dev server
make run

# 4. Verify DoD
curl http://localhost:8000/healthz
```

## CI Gate

```bash
make lint    # ruff + mypy
make test    # pytest (unit only, zero network)
make check   # lint + test combined
```

## Project Structure

```
saath-api/
├── app/
│   ├── api/          # FastAPI routers & request/response schemas
│   ├── core/         # Config (pydantic-settings), logging, DI container
│   ├── domain/       # Pure data models (MatchState, MomentEvent, SlotMap…)
│   ├── adapters/     # Port interfaces + concrete adapters
│   │   └── fake/     # FakeLLM, FakeMatchFeed, FakeCache, FakeUserContext
│   ├── engine/       # Pipeline, SlotVerifier, FallbackRenderer
│   ├── llm/          # LLM Gateway (single entry point — Rule 2)
│   ├── graph/        # Graph-based reasoning (scaffold)
│   └── eval/         # Evaluation harness (scaffold)
├── tests/            # Pytest unit tests (hermetic, no network)
├── docs/
│   └── CHARTER.md    # Binding architectural rules
├── config.yaml       # All tunables (Rule 5)
├── .env.example      # Environment variable template
├── Makefile          # make run / test / lint / check
└── Dockerfile        # Multi-stage, non-root
```

## Architecture Charter

See [`docs/CHARTER.md`](docs/CHARTER.md) for the seven binding rules.

| Rule | Summary |
|------|---------|
| 1 | LLM emits `{slot_tokens}`, never raw facts |
| 2 | All LLM calls go through `app/llm/gateway.py` |
| 3 | Every external dep behind an interface in `app/adapters/` |
| 4 | Every request degrades gracefully (LLM → cache → fallback) |
| 5 | No magic numbers — all tunables in `config.yaml` |
| 6 | Structured JSON logs with `request_id`, `match_id`, `moment_id` |
| 7 | Deterministic offline tests via Fake* adapters |
