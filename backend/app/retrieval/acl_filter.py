"""
Document Access Control Filter Module (Phase 5 — Checkpoint 5.2)
Provides active tenant role resolution and parameterized SQL authorization clauses.
Strictly pushes document ACL authorization into candidate retrieval queries.
"""

import uuid
from typing import List, Tuple, Any, Optional
import asyncpg

from backend.app.db.connection import get_db_pool


class DocumentACLFilter:
    """
    Retrieval-time document access control filter.
    Resolves tenant-scoped user roles and generates parameterized SQL authorization clauses.
    """

    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool

    async def get_user_role_ids(
        self,
        user_id: str,
        tenant_id: str,
        conn: Optional[asyncpg.Connection] = None
    ) -> List[str]:
        """
        Fetch active role IDs for (user_id, tenant_id) strictly bounded by tenant and user active status.
        """
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty UUID string")
        if not tenant_id or not isinstance(tenant_id, str):
            raise ValueError("tenant_id must be a non-empty UUID string")

        try:
            u_uuid = uuid.UUID(user_id.strip())
            t_uuid = uuid.UUID(tenant_id.strip())
        except ValueError as e:
            raise ValueError(f"Invalid UUID format for user_id or tenant_id: {e}")

        query = """
        SELECT ur.role_id
        FROM user_roles ur
        JOIN users u ON ur.user_id = u.id
        JOIN tenants t ON ur.tenant_id = t.id
        WHERE ur.user_id = $1
          AND ur.tenant_id = $2
          AND u.is_active = TRUE
          AND t.is_active = TRUE;
        """

        if conn is not None:
            rows = await conn.fetch(query, u_uuid, t_uuid)
        else:
            pool = self.db_pool if self.db_pool is not None else await get_db_pool()
            rows = await pool.fetch(query, u_uuid, t_uuid)

        return [str(row["role_id"]) for row in rows]

    def build_acl_where_clause(
        self,
        user_id: str,
        tenant_id: str,
        user_roles: Optional[List[str]] = None,
        param_offset: int = 1
    ) -> Tuple[str, List[Any]]:
        """
        Construct a parameterized SQL WHERE predicate enforcing document access control.

        Authorized conditions:
        1. Document owner match: d.owner_id = $user_id
        2. Document tenant visibility: d.visibility = 'tenant'
        3. Explicit document_access entry (user, role, or tenant matching access_level IN ('view', 'edit', 'owner'))

        Args:
            user_id: Requesting user UUID string.
            tenant_id: Requesting tenant UUID string.
            user_roles: List of resolved active role UUID strings.
            param_offset: Starting parameter index for $N placeholders (default 1).

        Returns:
            Tuple of (sql_clause_string, parameter_values_list).
        """
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty UUID string")
        if not tenant_id or not isinstance(tenant_id, str):
            raise ValueError("tenant_id must be a non-empty UUID string")
        if param_offset < 1:
            raise ValueError("param_offset must be >= 1")

        try:
            u_uuid = uuid.UUID(user_id.strip())
            t_uuid = uuid.UUID(tenant_id.strip())
        except ValueError as e:
            raise ValueError(f"Invalid UUID format for user_id or tenant_id: {e}")

        role_uuids = []
        if user_roles:
            for r in user_roles:
                try:
                    role_uuids.append(uuid.UUID(str(r).strip()))
                except ValueError as e:
                    raise ValueError(f"Invalid role UUID format '{r}': {e}")

        p1 = f"${param_offset}"
        p2 = f"${param_offset + 1}"
        p3 = f"${param_offset + 2}"

        clause = (
            f"(\n"
            f"    d.owner_id = {p1}::uuid\n"
            f" OR d.visibility = 'tenant'\n"
            f" OR EXISTS (\n"
            f"     SELECT 1 FROM document_access da\n"
            f"     WHERE da.document_id = d.id\n"
            f"       AND (\n"
            f"            (da.principal_type = 'user' AND da.principal_id = {p1}::uuid)\n"
            f"         OR (da.principal_type = 'role' AND da.principal_id = ANY({p3}::uuid[]))\n"
            f"         OR (da.principal_type = 'tenant' AND da.principal_id = {p2}::uuid)\n"
            f"       )\n"
            f"       AND da.access_level IN ('view', 'edit', 'owner')\n"
            f" )\n"
            f")"
        )

        params = [u_uuid, t_uuid, role_uuids]
        return clause, params

    async def build_acl_filter_for_user(
        self,
        user_id: str,
        tenant_id: str,
        conn: Optional[asyncpg.Connection] = None,
        param_offset: int = 1
    ) -> Tuple[str, List[Any]]:
        """
        Convenience method combining active role resolution and ACL WHERE clause building.
        Guarantees that caller never forgets to resolve tenant-scoped user roles.
        """
        role_ids = await self.get_user_role_ids(user_id, tenant_id, conn=conn)
        return self.build_acl_where_clause(
            user_id=user_id,
            tenant_id=tenant_id,
            user_roles=role_ids,
            param_offset=param_offset
        )


