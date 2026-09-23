"""
Phase 2 Unit Tests: JWT Access Token Issuance & Verification
"""

import os
import pytest
from datetime import timedelta
from fastapi import HTTPException

from backend.app.auth.jwt import create_access_token, verify_token


@pytest.fixture(autouse=True)
def set_jwt_env(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-phase2-secure-12345")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15")


def test_create_and_verify_valid_token():
    payload = {
        "sub": "user-uuid-123",
        "tenant_id": "tenant-uuid-456",
        "email": "test@example.com"
    }
    token = create_access_token(payload)
    assert isinstance(token, str)

    decoded = verify_token(token)
    assert decoded["sub"] == "user-uuid-123"
    assert decoded["tenant_id"] == "tenant-uuid-456"
    assert decoded["email"] == "test@example.com"
    assert "exp" in decoded
    assert "iat" in decoded
    assert "jti" in decoded


def test_expired_token_raises_401():
    payload = {"sub": "user-1", "tenant_id": "tenant-1", "email": "a@b.com"}
    token = create_access_token(payload, expires_delta=timedelta(seconds=-10))

    with pytest.raises(HTTPException) as exc_info:
        verify_token(token)
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


def test_malformed_token_raises_401():
    with pytest.raises(HTTPException) as exc_info:
        verify_token("invalid.jwt.token.string")
    assert exc_info.value.status_code == 401


def test_invalid_signature_raises_401():
    payload = {"sub": "user-1", "tenant_id": "tenant-1", "email": "a@b.com"}
    token = create_access_token(payload)

    # Decode payload and re-sign with wrong secret
    tampered_token = token[:-5] + "XXXXX"
    with pytest.raises(HTTPException) as exc_info:
        verify_token(tampered_token)
    assert exc_info.value.status_code == 401


def test_missing_required_claims():
    # Missing tenant_id
    token_no_tenant = create_access_token({"sub": "user-1", "email": "a@b.com"})
    with pytest.raises(HTTPException) as exc_info:
        verify_token(token_no_tenant)
    assert exc_info.value.status_code == 401
    assert "tenant_id" in exc_info.value.detail

    # Missing sub
    token_no_sub = create_access_token({"tenant_id": "tenant-1", "email": "a@b.com"})
    with pytest.raises(HTTPException) as exc_info:
        verify_token(token_no_sub)
    assert exc_info.value.status_code == 401
    assert "sub" in exc_info.value.detail


def test_missing_jwt_secret_raises_value_error(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    with pytest.raises(ValueError) as exc_info:
        create_access_token({"sub": "1", "tenant_id": "1"})
    assert "JWT_SECRET environment variable is missing or empty" in str(exc_info.value)
