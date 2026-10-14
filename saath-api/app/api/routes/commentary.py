"""
Commentary router — /v1/commentary

Scaffold for the commentary generation endpoints.
No business logic yet — just wiring and request/response schemas.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated

import structlog
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.logging import set_request_context
from app.domain.models import CommentaryResponse, MomentEvent, MomentType

if TYPE_CHECKING:
    from app.core.container import AppContainer

router = APIRouter()
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class CommentaryRequest(BaseModel):
    match_id: str = Field(..., description="Unique match identifier")
    moment_id: str = Field(..., description="Unique moment/event identifier")
    moment_type: MomentType = Field(default=MomentType.GENERIC)
    user_id: str = Field(..., description="Requesting user identifier")
    primary_player_id: str | None = Field(default=None)
    metadata: dict[str, str | int | float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _get_container(request: Request) -> AppContainer:

    container: AppContainer = request.app.state.container
    return container


@router.post("/commentary", response_model=CommentaryResponse, status_code=200)
async def generate_commentary(
    body: CommentaryRequest,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> CommentaryResponse:
    """
    Generate commentary for a match moment.

    Rule 4: Always returns 200; falls back to deterministic template on any error.
    """
    set_request_context(
        match_id=body.match_id,
        moment_id=body.moment_id,
    )
    logger.info(
        "commentary_request_received",
        match_id=body.match_id,
        moment_id=body.moment_id,
        moment_type=body.moment_type,
        user_id=body.user_id,
    )

    container = _get_container(request)
    event = MomentEvent(
        moment_id=body.moment_id,
        match_id=body.match_id,
        moment_type=body.moment_type,
        primary_player_id=body.primary_player_id,
        metadata=body.metadata,
        timestamp=body.timestamp,
    )

    from app.engine.pipeline import CommentaryPipeline

    pipeline = CommentaryPipeline(
        settings=settings,
        llm_gateway=container.llm_gateway,
        match_feed=container.match_feed,
        user_context=container.user_context,
        cache=container.cache,
    )

    response = await pipeline.generate(event=event, user_id=body.user_id)
    return response
