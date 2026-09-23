"""
Phase 1 Health & Readiness Integration Test Suite
"""

import pytest
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport
from backend.app.main import app


@pytest.mark.asyncio
async def test_liveness_returns_200_ok():
    """Verify /health returns HTTP 200 OK liveness check without I/O dependencies."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["data"]["status"] == "ok"
    assert json_data["error"] is None
    assert "request_id" in json_data


@pytest.mark.asyncio
@patch("backend.app.api.health._check_postgres", return_value=True)
@patch("backend.app.api.health._check_redis", return_value=True)
@patch("backend.app.api.health._check_minio", return_value=True)
@patch("backend.app.api.health._check_llm_provider", return_value=True)
async def test_readiness_healthy_stack(mock_llm, mock_minio, mock_redis, mock_pg):
    """Verify /ready returns HTTP 200 OK with all checks 'ok' when dependencies are healthy."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/ready")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["data"]["status"] == "ready"
    assert json_data["data"]["checks"] == {
        "postgres": "ok",
        "redis": "ok",
        "minio": "ok",
        "llm_provider": "ok"
    }
    assert json_data["error"] is None


@pytest.mark.asyncio
@patch("backend.app.api.health._check_postgres", return_value=True)
@patch("backend.app.api.health._check_redis", return_value=False)
@patch("backend.app.api.health._check_minio", return_value=True)
@patch("backend.app.api.health._check_llm_provider", return_value=True)
async def test_readiness_dependency_failure(mock_llm, mock_minio, mock_redis, mock_pg):
    """Verify /ready returns HTTP 503 when a dependency fails, while /health remains HTTP 200 OK."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        ready_response = await ac.get("/ready")
        health_response = await ac.get("/health")

    # Readiness should return HTTP 503 Service Unavailable
    assert ready_response.status_code == 503
    ready_json = ready_response.json()
    assert ready_json["data"]["status"] == "not_ready"
    assert ready_json["data"]["checks"]["redis"] == "failed"
    assert ready_json["data"]["checks"]["postgres"] == "ok"
    assert ready_json["error"]["code"] == "SERVICE_UNAVAILABLE"

    # Liveness must remain HTTP 200 OK
    assert health_response.status_code == 200
    assert health_response.json()["data"]["status"] == "ok"
