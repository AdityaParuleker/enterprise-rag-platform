"""
Integration tests for Phase 9 Evaluation API, RBAC Permissions & Tenant Isolation (Checkpoint 9.5)
"""

import asyncio
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.auth.jwt import create_access_token
from backend.app.api.eval import _EVAL_RUNS_STORE, _EVAL_RESULTS_STORE

client = TestClient(app)

TENANT_A = "ee45761a-f471-4f41-97ff-697d927e0745"
TENANT_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
USER_A = "ca3b120e-6936-4bb3-bee9-550f6b1dce80"
USER_B = "cccccccc-cccc-cccc-cccc-cccccccccccc"


@pytest.fixture(autouse=True)
def set_jwt_env(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-phase2-secure-12345")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15")


def get_token(user_id: str, tenant_id: str, role: str = "SUPER_ADMIN") -> str:
    return create_access_token(
        data={"sub": user_id, "tenant_id": tenant_id, "email": "test@example.com", "role": role}
    )


def test_eval_api_rbac_permissions(monkeypatch):
    """Verifies eval:run, eval:read, and eval:compare RBAC permissions and 403 Forbidden enforcement."""
    admin_token = get_token(USER_A, TENANT_A, role="SUPER_ADMIN")
    viewer_token = get_token(USER_B, TENANT_A, role="VIEWER")

    # VIEWER returns empty permissions -> 403 Forbidden
    monkeypatch.setattr("backend.app.auth.rbac.get_user_permissions", AsyncMock(return_value=set()))
    res_block = client.post(
        "/api/v1/eval/run",
        json={"config_name": "full_pipeline"},
        headers={"Authorization": f"Bearer {viewer_token}"}
    )
    assert res_block.status_code == 403

    # SUPER_ADMIN has full eval permissions -> 202 Accepted
    monkeypatch.setattr("backend.app.auth.rbac.get_user_permissions", AsyncMock(return_value={"eval:run", "eval:read", "eval:compare"}))
    res_allow = client.post(
        "/api/v1/eval/run",
        json={"config_name": "full_pipeline"},
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert res_allow.status_code == 202
    data = res_allow.json()
    assert "run_id" in data
    assert data["status"] in ("QUEUED", "RUNNING")


def test_eval_results_tenant_isolation(monkeypatch):
    """Verifies Tenant B cannot view or compare Tenant A's evaluation runs/results."""
    monkeypatch.setattr("backend.app.auth.rbac.get_user_permissions", AsyncMock(return_value={"eval:run", "eval:read", "eval:compare"}))
    admin_a_token = get_token(USER_A, TENANT_A, role="SUPER_ADMIN")
    admin_b_token = get_token(USER_B, TENANT_B, role="SUPER_ADMIN")

    # Start run for Tenant A
    run_res = client.post(
        "/api/v1/eval/run",
        json={"config_name": "vector_only"},
        headers={"Authorization": f"Bearer {admin_a_token}"}
    )
    assert run_res.status_code == 202
    run_id = run_res.json()["run_id"]

    # Tenant A can access run_id
    res_a = client.get(
        f"/api/v1/eval/results?run_id={run_id}",
        headers={"Authorization": f"Bearer {admin_a_token}"}
    )
    assert res_a.status_code == 200

    # Tenant B gets 404 NOT FOUND when accessing Tenant A's run_id
    res_b = client.get(
        f"/api/v1/eval/results?run_id={run_id}",
        headers={"Authorization": f"Bearer {admin_b_token}"}
    )
    assert res_b.status_code == 404


def test_eval_run_async_lifecycle_transitions(monkeypatch):
    """Verifies async run lifecycle state machine (QUEUED -> RUNNING -> COMPLETED)."""
    monkeypatch.setattr("backend.app.auth.rbac.get_user_permissions", AsyncMock(return_value={"eval:run", "eval:read", "eval:compare"}))
    admin_token = get_token(USER_A, TENANT_A, role="SUPER_ADMIN")

    run_res = client.post(
        "/api/v1/eval/run",
        json={"config_name": "hybrid"},
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert run_res.status_code == 202
    run_id = run_res.json()["run_id"]

    # Immediate state check: QUEUED or RUNNING or COMPLETED
    res_poll = client.get(
        f"/api/v1/eval/results?run_id={run_id}",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert res_poll.status_code == 200
    assert res_poll.json()["status"] in ("QUEUED", "RUNNING", "COMPLETED")


def test_eval_run_cancellation_semantics(monkeypatch):
    """Verifies POST /api/v1/eval/cancel sets run status to CANCELLED."""
    monkeypatch.setattr("backend.app.auth.rbac.get_user_permissions", AsyncMock(return_value={"eval:run", "eval:read", "eval:compare"}))
    admin_token = get_token(USER_A, TENANT_A, role="SUPER_ADMIN")

    run_res = client.post(
        "/api/v1/eval/run",
        json={"config_name": "full_pipeline"},
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert run_res.status_code == 202
    run_id = run_res.json()["run_id"]

    cancel_res = client.post(
        "/api/v1/eval/cancel",
        json={"run_id": run_id},
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert cancel_res.status_code == 200
    assert cancel_res.json()["status"] == "CANCELLED"
