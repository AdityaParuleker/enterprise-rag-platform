"""
RBAC Authorization Module (Section 6.2)
Provides get_current_user and require_permission FastAPI dependencies.
Strictly scopes permission evaluation to user_id AND tenant_id.
"""

from typing import Dict, Optional, Set
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Request, status

from backend.app.auth.jwt import verify_token
from backend.app.db.connection import get_db_connection


async def get_current_user(request: Request, authorization: Optional[str] = Header(None)) -> Dict:
    """Extract and verify JWT Bearer token from Authorization header. Sets request.state."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = authorization.split(" ", 1)[1].strip()
    payload = verify_token(token)

    user_id = payload.get("sub")
    tenant_id = payload.get("tenant_id")
    email = payload.get("email", "")

    # Populate request.state for downstream handling
    request.state.user_id = user_id
    request.state.tenant_id = tenant_id
    request.state.email = email

    return {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "email": email
    }


async def get_user_permissions(user_id: str, tenant_id: str) -> Set[str]:
    """Resolve permissions set for user strictly bounded by user_id AND tenant_id."""
    try:
        user_uuid = UUID(user_id)
        tenant_uuid = UUID(tenant_id)
    except ValueError:
        return set()

    try:
        conn = await get_db_connection()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database service unavailable ({exc}). Please ensure PostgreSQL is running on port 5432."
        )
    try:
        rows = await conn.fetch(
            """
            SELECT DISTINCT p.name
            FROM user_roles ur
            JOIN role_permissions rp ON ur.role_id = rp.role_id
            JOIN permissions p ON rp.permission_id = p.id
            WHERE ur.user_id = $1 AND ur.tenant_id = $2
            """,
            user_uuid,
            tenant_uuid
        )
        return {r["name"] for r in rows}
    finally:
        await conn.close()


def require_permission(permission_name: str):
    """FastAPI dependency factory enforcing specified granular permission."""
    async def permission_dependency(
        request: Request,
        user: Dict = Depends(get_current_user)
    ) -> Dict:
        user_id = user["user_id"]
        tenant_id = user["tenant_id"]

        permissions = await get_user_permissions(user_id, tenant_id)

        if permission_name not in permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission_name}' required"
            )

        return user

    return permission_dependency
