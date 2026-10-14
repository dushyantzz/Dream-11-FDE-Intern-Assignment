"""
Health-check router — GET /healthz

Returns 200 + JSON with basic liveness metadata.
No auth required; used by load balancers and K8s probes.
"""

from __future__ import annotations

import platform
import time
from typing import Any

from fastapi import APIRouter

router = APIRouter()

_START_TIME = time.time()


@router.get("/healthz", response_model=dict[str, Any])
async def healthz() -> dict[str, Any]:
    """Liveness probe — always returns 200 if the process is alive."""
    return {
        "status": "ok",
        "service": "saath-api",
        "version": "0.1.0",
        "uptime_s": round(time.time() - _START_TIME, 1),
        "python": platform.python_version(),
    }


@router.get("/readyz", response_model=dict[str, Any])
async def readyz() -> dict[str, Any]:
    """
    Readiness probe — extend this to check adapter connectivity
    (cache, match feed) before declaring ready.
    """
    return {"status": "ready"}
