"""
Structured JSON logging — Rule 6 enforcement.

Configures structlog to emit JSON log lines with mandatory correlation IDs:
  request_id, match_id, moment_id

Correlation IDs are stored in contextvars and injected into every log line
automatically via the `add_correlation_ids` processor.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import structlog

from app.core.config import Settings

# ---------------------------------------------------------------------------
# Context variables — set at request boundary, read by log processor
# ---------------------------------------------------------------------------

_request_id: ContextVar[str] = ContextVar("request_id", default="")
_match_id: ContextVar[str | None] = ContextVar("match_id", default=None)
_moment_id: ContextVar[str | None] = ContextVar("moment_id", default=None)


def set_request_context(
    request_id: str | None = None,
    match_id: str | None = None,
    moment_id: str | None = None,
) -> None:
    """Inject correlation IDs into the current async context."""
    _request_id.set(request_id or str(uuid4()))
    _match_id.set(match_id)
    _moment_id.set(moment_id)


def get_request_id() -> str:
    """Return the current request_id (generates one if not set)."""
    rid = _request_id.get()
    if not rid:
        rid = str(uuid4())
        _request_id.set(rid)
    return rid


# ---------------------------------------------------------------------------
# structlog processor: injects correlation IDs into every log event
# ---------------------------------------------------------------------------


def _add_correlation_ids(
    logger: Any,
    method: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    event_dict["request_id"] = get_request_id()
    event_dict["match_id"] = _match_id.get()
    event_dict["moment_id"] = _moment_id.get()
    return event_dict


# ---------------------------------------------------------------------------
# Configuration entry point
# ---------------------------------------------------------------------------


def configure_logging(settings: Settings) -> None:
    """
    Configure structlog + stdlib logging.

    Must be called once at application startup, before any other module logs.
    """
    log_level = settings.logging.level
    use_json = settings.logging.format == "json"

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _add_correlation_ids,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if use_json:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Quieten noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
