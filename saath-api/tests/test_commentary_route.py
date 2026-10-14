"""
Tests for POST /v1/commentary route.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
def test_commentary_route_happy_path(test_client: TestClient) -> None:
    """POST /v1/commentary returns 200 with rendered commentary."""
    payload = {
        "match_id": "match_ind_vs_pak_2026",
        "moment_id": "evt_over_18_4",
        "moment_type": "boundary",
        "user_id": "usr_9988",
        "primary_player_id": "ply_001",
    }
    response = test_client.post("/v1/commentary", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "commentary_text" in data
    assert len(data["commentary_text"]) > 0
    assert data["match_id"] == "match_ind_vs_pak_2026"
    assert data["moment_id"] == "evt_over_18_4"
    assert "is_fallback" in data
    assert "latency_ms" in data
    assert "cost_usd" in data


@pytest.mark.unit
def test_commentary_route_validation_error(test_client: TestClient) -> None:
    """Missing required fields returns 422 Unprocessable Entity."""
    payload = {
        "match_id": "match_ind_vs_pak_2026",
    }
    response = test_client.post("/v1/commentary", json=payload)
    assert response.status_code == 422
