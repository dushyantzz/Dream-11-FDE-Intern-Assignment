"""Tests for GET /healthz — Definition of Done check."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
def test_healthz_returns_200(test_client: TestClient) -> None:
    """GET /healthz must return 200 with status=ok (DoD requirement)."""
    response = test_client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "saath-api"
    assert "version" in data
    assert "uptime_s" in data


@pytest.mark.unit
def test_readyz_returns_200(test_client: TestClient) -> None:
    """GET /readyz must return 200."""
    response = test_client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
