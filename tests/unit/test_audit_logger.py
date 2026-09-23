"""
Unit tests for Checkpoint 8.4 — AuditLogger & Data Safety.
"""

import pytest
from unittest.mock import AsyncMock
from backend.app.auth.audit_logger import AuditLogger, sanitize_audit_detail


def test_sanitize_audit_detail_redacts_sensitive_keys():
    """Verify passwords, tokens, API keys, and raw prompts are redacted from audit detail dicts."""
    raw_detail = {
        "user_id": "u123",
        "action": "LOGIN",
        "password": "super-secret-password",
        "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        "nested": {
            "api_key": "ak_12345",
            "safe_field": "ok"
        }
    }

    sanitized = sanitize_audit_detail(raw_detail)

    assert sanitized["user_id"] == "u123"
    assert sanitized["password"] == "[REDACTED_AUDIT_PAYLOAD]"
    assert sanitized["jwt"] == "[REDACTED_AUDIT_PAYLOAD]"
    assert sanitized["nested"]["api_key"] == "[REDACTED_AUDIT_PAYLOAD]"
    assert sanitized["nested"]["safe_field"] == "ok"


@pytest.mark.asyncio
async def test_audit_logger_log_event():
    """Verify log_event executes SQL query with sanitized payload."""
    logger = AuditLogger()
    mock_conn = AsyncMock()

    log_id = await logger.log_event(
        tenant_id="t1",
        action="GUARDRAIL_TRIGGERED",
        resource_type="CHAT",
        user_id="u1",
        detail={"password": "123", "guardrail_type": "INPUT_INJECTION"},
        conn=mock_conn
    )

    assert log_id is not None
    mock_conn.execute.assert_awaited_once()
    sql_args = mock_conn.execute.call_args[0]
    assert sql_args[1] == log_id
    assert sql_args[2] == "t1"
    assert "[REDACTED_AUDIT_PAYLOAD]" in sql_args[8]
