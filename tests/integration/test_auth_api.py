"""
Phase 2 Integration Tests: Auth API Endpoints
Tests Registration, Login, Token Refresh Rotation, Logout, and Password Reset.
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta, timezone
from httpx import AsyncClient, ASGITransport

from backend.app.main import app
from backend.app.auth.password import hash_password


class MockTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


@pytest.fixture(autouse=True)
def set_jwt_env(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-phase2-integration-999")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15")


@pytest.mark.asyncio
async def test_register_new_tenant_and_user_success():
    mock_conn = AsyncMock()
    mock_conn.fetchrow.side_effect = [
        None,  # existing user check
        {"id": uuid.UUID("11111111-1111-1111-1111-111111111111")},  # tenant
        {"id": uuid.UUID("22222222-2222-2222-2222-222222222222"), "email": "newadmin@example.com"},  # user
        {"id": uuid.UUID("33333333-3333-3333-3333-333333333333")}  # role
    ]
    mock_conn.transaction = MagicMock(return_value=MockTransaction())

    with patch("backend.app.api.auth.get_db_connection", new_callable=AsyncMock) as mock_db:
        mock_db.return_value = mock_conn

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/api/v1/auth/register", json={
                "email": "newadmin@example.com",
                "password": "Password123!",
                "tenant_name": "Acme Corp"
            })

        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["email"] == "newadmin@example.com"
        assert body["data"]["role"] == "TENANT_ADMIN"
        assert body["data"]["tenant_id"] == "11111111-1111-1111-1111-111111111111"
        assert body["data"]["user_id"] == "22222222-2222-2222-2222-222222222222"
        assert "request_id" in body


@pytest.mark.asyncio
async def test_register_duplicate_email_conflict():
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {"id": uuid.uuid4()}  # email already exists
    mock_conn.transaction = MagicMock(return_value=MockTransaction())

    with patch("backend.app.api.auth.get_db_connection", new_callable=AsyncMock) as mock_db:
        mock_db.return_value = mock_conn

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/api/v1/auth/register", json={
                "email": "existing@example.com",
                "password": "Password123!"
            })

        assert resp.status_code == 409
        body = resp.json()
        assert "already exists" in body["error"]["message"]


@pytest.mark.asyncio
async def test_login_valid_credentials():
    user_uuid = uuid.UUID("22222222-2222-2222-2222-222222222222")
    tenant_uuid = uuid.UUID("11111111-1111-1111-1111-111111111111")
    hashed_pwd = hash_password("ValidPassword123")

    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {
        "id": user_uuid,
        "tenant_id": tenant_uuid,
        "email": "user@example.com",
        "password_hash": hashed_pwd,
        "is_active": True
    }

    with patch("backend.app.api.auth.get_db_connection", new_callable=AsyncMock) as mock_db:
        mock_db.return_value = mock_conn

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/api/v1/auth/login", json={
                "email": "user@example.com",
                "password": "ValidPassword123"
            })

        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body["data"]
        assert "refresh_token" in body["data"]
        assert body["data"]["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_invalid_password_returns_401():
    hashed_pwd = hash_password("CorrectPassword")
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {
        "id": uuid.uuid4(),
        "tenant_id": uuid.uuid4(),
        "email": "user@example.com",
        "password_hash": hashed_pwd,
        "is_active": True
    }

    with patch("backend.app.api.auth.get_db_connection", new_callable=AsyncMock) as mock_db:
        mock_db.return_value = mock_conn

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/api/v1/auth/login", json={
                "email": "user@example.com",
                "password": "WrongPassword"
            })

        assert resp.status_code == 401
        body = resp.json()
        assert "Invalid email or password" in body["error"]["message"]


@pytest.mark.asyncio
async def test_refresh_token_rotation_success():
    token_uuid = uuid.UUID("44444444-4444-4444-4444-444444444444")
    user_uuid = uuid.UUID("22222222-2222-2222-2222-222222222222")
    tenant_uuid = uuid.UUID("11111111-1111-1111-1111-111111111111")
    future_exp = datetime.now(timezone.utc) + timedelta(days=7)

    mock_conn = AsyncMock()
    mock_conn.fetchrow.side_effect = [
        {"id": token_uuid, "user_id": user_uuid, "expires_at": future_exp, "revoked_at": None},
        {"id": user_uuid, "tenant_id": tenant_uuid, "email": "user@example.com", "is_active": True}
    ]
    mock_conn.transaction = MagicMock(return_value=MockTransaction())

    with patch("backend.app.api.auth.get_db_connection", new_callable=AsyncMock) as mock_db:
        mock_db.return_value = mock_conn

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/api/v1/auth/refresh", json={
                "refresh_token": "valid_old_refresh_token_123"
            })

        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body["data"]
        assert "refresh_token" in body["data"]
        assert body["data"]["refresh_token"] != "valid_old_refresh_token_123"


@pytest.mark.asyncio
async def test_refresh_token_revoked_reuse_fails_with_401():
    token_uuid = uuid.UUID("44444444-4444-4444-4444-444444444444")
    user_uuid = uuid.UUID("22222222-2222-2222-2222-222222222222")

    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {
        "id": token_uuid,
        "user_id": user_uuid,
        "expires_at": datetime.now(timezone.utc) + timedelta(days=7),
        "revoked_at": datetime.now(timezone.utc)
    }
    mock_conn.transaction = MagicMock(return_value=MockTransaction())

    with patch("backend.app.api.auth.get_db_connection", new_callable=AsyncMock) as mock_db:
        mock_db.return_value = mock_conn

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/api/v1/auth/refresh", json={
                "refresh_token": "revoked_token_reuse_attempt"
            })

        assert resp.status_code == 401
        assert "revoked" in resp.json()["error"]["message"].lower()


@pytest.mark.asyncio
async def test_logout_revokes_token():
    mock_conn = AsyncMock()
    with patch("backend.app.api.auth.get_db_connection", new_callable=AsyncMock) as mock_db:
        mock_db.return_value = mock_conn

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/api/v1/auth/logout", json={
                "refresh_token": "token_to_logout"
            })

        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "logged_out"
        assert mock_conn.execute.called


@pytest.mark.asyncio
async def test_password_reset_success():
    hashed_pwd = hash_password("OldPassword123")
    user_uuid = uuid.uuid4()

    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {
        "id": user_uuid,
        "password_hash": hashed_pwd
    }
    mock_conn.transaction = MagicMock(return_value=MockTransaction())

    with patch("backend.app.api.auth.get_db_connection", new_callable=AsyncMock) as mock_db:
        mock_db.return_value = mock_conn

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/api/v1/auth/password-reset", json={
                "email": "user@example.com",
                "old_password": "OldPassword123",
                "new_password": "NewSecretPassword456!"
            })

        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "password_reset_success"
