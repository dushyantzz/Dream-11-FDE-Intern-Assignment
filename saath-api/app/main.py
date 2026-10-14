"""saath-api — FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import commentary, health
from app.core.config import get_settings
from app.core.container import AppContainer
from app.core.logging import configure_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup → yield → shutdown."""
    settings = get_settings()
    configure_logging(settings)

    container = AppContainer(settings)
    app.state.container = container

    logger.info(
        "saath-api starting up",
        version="0.1.0",
        llm_provider=settings.llm.provider,
        cache_backend=settings.cache.backend,
    )

    yield

    logger.info("saath-api shutting down")


def create_app() -> FastAPI:
    """Factory function that creates and wires the FastAPI application."""
    app = FastAPI(
        title="saath-api",
        description="Real-time commentary & insight engine for fantasy-sports moments",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.state.settings = get_settings()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health.router, tags=["ops"])
    app.include_router(commentary.router, prefix="/v1", tags=["commentary"])

    return app


app = create_app()
