"""
Structured Audit Logging Utility (Section 6.6 & Phase 8 Specifications)
Writes sanitized security audit events to the PostgreSQL audit_logs table with tenant isolation.
"""

import json
import uuid
from typing import Optional, Dict, Any, List
from backend.app.db.connection import get_db_pool

SENSITIVE_KEYS = {
    "password", "secret", "token", "access_token", "jwt", "api_key",
    "prompt", "completion", "raw_text", "file_bytes"
}


def sanitize_audit_detail(detail: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitizes audit detail payload ensuring no raw passwords, JWTs, API keys, or raw prompt/completion text are stored."""
    if not detail:
        return {}

    sanitized = {}
    for k, v in detail.items():
        k_lower = k.lower()
        if any(sk in k_lower for sk in SENSITIVE_KEYS):
            sanitized[k] = "[REDACTED_AUDIT_PAYLOAD]"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_audit_detail(v)
        else:
            sanitized[k] = v

    return sanitized


class AuditLogger:
    """Asynchronous audit logger persisting security events to audit_logs database table."""

    async def log_event(
        self,
        tenant_id: str,
        action: str,
        resource_type: str,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
        conn = None
    ) -> str:
        log_id = str(uuid.uuid4())
        sanitized_detail = sanitize_audit_detail(detail or {})

        sql = """
            INSERT INTO audit_logs (id, tenant_id, user_id, action, resource_type, resource_id, ip_address, detail)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """

        detail_json = json.dumps(sanitized_detail)

        if conn:
            await conn.execute(sql, log_id, tenant_id, user_id, action, resource_type, resource_id, ip_address, detail_json)
        else:
            pool = await get_db_pool()
            if pool:
                async with pool.acquire() as db_conn:
                    await db_conn.execute(sql, log_id, tenant_id, user_id, action, resource_type, resource_id, ip_address, detail_json)

        return log_id

    async def list_logs(
        self,
        tenant_id: str,
        user_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        pool = await get_db_pool()
        if not pool:
            return []

        async with pool.acquire() as conn:
            if user_id:
                rows = await conn.fetch(
                    """
                    SELECT id, tenant_id, user_id, action, resource_type, resource_id, ip_address, created_at, detail
                    FROM audit_logs
                    WHERE tenant_id = $1 AND user_id = $2
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    tenant_id, user_id, limit
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, tenant_id, user_id, action, resource_type, resource_id, ip_address, created_at, detail
                    FROM audit_logs
                    WHERE tenant_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    tenant_id, limit
                )

            return [
                {
                    "id": str(r["id"]),
                    "tenant_id": str(r["tenant_id"]),
                    "user_id": str(r["user_id"]) if r["user_id"] else None,
                    "action": r["action"],
                    "resource_type": r["resource_type"],
                    "resource_id": str(r["resource_id"]) if r["resource_id"] else None,
                    "ip_address": r["ip_address"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                    "detail": json.loads(r["detail"]) if isinstance(r["detail"], str) else (r["detail"] or {})
                }
                for r in rows
            ]
