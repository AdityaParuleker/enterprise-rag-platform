"""
Phase 2 Unit Tests: RBAC Dependency & Tenant Isolation Checks
"""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException

from backend.app.auth.rbac import get_current_user, get_user_permissions, require_permission


@pytest.fixture
def mock_request():
    class DummyState:
        pass

    class DummyRequest:
        def __init__(self):
            self.state = DummyState()

    return DummyRequest()


@pytest.mark.asyncio
async def test_get_current_user_valid_token(mock_request, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-123")
    from backend.app.auth.jwt import create_access_token

    token = create_access_token({"sub": "user-uuid-1", "tenant_id": "tenant-uuid-1", "email": "user@test.com"})
    auth_header = f"Bearer {token}"

    user_data = await get_current_user(mock_request, authorization=auth_header)

    assert user_data["user_id"] == "user-uuid-1"
    assert user_data["tenant_id"] == "tenant-uuid-1"
    assert mock_request.state.user_id == "user-uuid-1"
    assert mock_request.state.tenant_id == "tenant-uuid-1"


@pytest.mark.asyncio
async def test_get_current_user_missing_or_invalid_header(mock_request):
    with pytest.raises(HTTPException) as exc1:
        await get_current_user(mock_request, authorization=None)
    assert exc1.value.status_code == 401

    with pytest.raises(HTTPException) as exc2:
        await get_current_user(mock_request, authorization="Basic invalid")
    assert exc2.value.status_code == 401


@pytest.mark.asyncio
async def test_require_permission_allowed_and_denied(mock_request):
    user_info = {"user_id": "user-1", "tenant_id": "tenant-1"}

    with patch("backend.app.auth.rbac.get_user_permissions", new_callable=AsyncMock) as mock_get_perms:
        mock_get_perms.return_value = {"upload_document", "view_document"}

        dep = require_permission("upload_document")
        result = await dep(mock_request, user=user_info)
        assert result == user_info

        dep_denied = require_permission("manage_users")
        with pytest.raises(HTTPException) as exc_info:
            await dep_denied(mock_request, user=user_info)
        assert exc_info.value.status_code == 403
        assert "manage_users" in exc_info.value.detail


@pytest.mark.asyncio
async def test_tenant_boundary_isolation_query(monkeypatch):
    """Verify get_user_permissions includes both user_id AND tenant_id in SQL query."""
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = [{"name": "upload_document"}]

    with patch("backend.app.auth.rbac.get_db_connection", new_callable=AsyncMock) as mock_db:
        mock_db.return_value = mock_conn

        u_id = "11111111-1111-1111-1111-111111111111"
        t_id = "22222222-2222-2222-2222-222222222222"

        perms = await get_user_permissions(u_id, t_id)
        assert perms == {"upload_document"}

        # Verify SQL call executed with both user_uuid and tenant_uuid
        args = mock_conn.fetch.call_args[0]
        assert "ur.user_id = $1 AND ur.tenant_id = $2" in args[0]
        assert str(args[1]) == u_id
        assert str(args[2]) == t_id
