"""
Security tests for Document Access Control (ACL) Enforcement (Phase 5 — Security Boundary).
Verifies that unauthorized users are strictly excluded by the ACL WHERE clause.
"""

import uuid
import pytest
from unittest.mock import AsyncMock

from backend.app.retrieval.acl_filter import DocumentACLFilter


@pytest.mark.asyncio
async def test_acl_security_clause_composition_and_parameter_alignment():
    """
    Verify complete SQL candidate query composition with tenant predicate and ACL clause.
    Ensures param_offset alignment ($1=tenant_id, $2=user_id, $3=tenant_id, $4=role_ids, $5=limit).
    """
    acl_filter = DocumentACLFilter()
    user_a_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())
    role_id = str(uuid.uuid4())

    # Build ACL clause starting at param_offset=2 ($2=user_id, $3=tenant_id, $4=role_ids)
    acl_clause, acl_params = acl_filter.build_acl_where_clause(
        user_id=user_a_id,
        tenant_id=tenant_id,
        user_roles=[role_id],
        param_offset=2
    )

    full_candidate_query = f"""
    SELECT c.id, c.text
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE c.tenant_id = $1::uuid
      AND d.is_latest = TRUE
      AND d.status = 'INDEXED'
      AND {acl_clause}
    LIMIT $5;
    """

    assert "WHERE c.tenant_id = $1::uuid" in full_candidate_query
    assert "d.owner_id = $2::uuid" in full_candidate_query
    assert "da.principal_id = $2::uuid" in full_candidate_query
    assert "ANY($4::uuid[])" in full_candidate_query
    assert "da.principal_id = $3::uuid" in full_candidate_query

    # Combine query parameters: [$1: tenant_id, $2: user_id, $3: tenant_id, $4: role_ids, $5: limit]
    all_params = [uuid.UUID(tenant_id)] + acl_params + [10]
    assert len(all_params) == 5
    assert all_params[0] == uuid.UUID(tenant_id)
    assert all_params[1] == uuid.UUID(user_a_id)
    assert all_params[2] == uuid.UUID(tenant_id)
    assert all_params[3] == [uuid.UUID(role_id)]
    assert all_params[4] == 10


def test_acl_security_boundary_conditions():
    """
    Verify security boundary rules for owner, tenant-visibility, and explicit ACL principals.
    """
    acl_filter = DocumentACLFilter()
    user_a = str(uuid.uuid4())
    user_b = str(uuid.uuid4())
    tenant = str(uuid.uuid4())
    role_admin = str(uuid.uuid4())

    # User A clause
    clause_a, params_a = acl_filter.build_acl_where_clause(user_a, tenant, [role_admin], param_offset=1)
    # User B clause
    clause_b, params_b = acl_filter.build_acl_where_clause(user_b, tenant, [], param_offset=1)

    # 1. User A parameters contain User A UUID
    assert params_a[0] == uuid.UUID(user_a)
    assert params_a[2] == [uuid.UUID(role_admin)]

    # 2. User B parameters contain User B UUID and empty role list
    assert params_b[0] == uuid.UUID(user_b)
    assert params_b[2] == []

    # 3. Clause includes owner, tenant visibility, user ACL, role ACL, and tenant ACL
    for clause in (clause_a, clause_b):
        assert "d.owner_id = $1::uuid" in clause
        assert "d.visibility = 'tenant'" in clause
        assert "da.principal_type = 'user' AND da.principal_id = $1::uuid" in clause
        assert "da.principal_type = 'role' AND da.principal_id = ANY($3::uuid[])" in clause
        assert "da.principal_type = 'tenant' AND da.principal_id = $2::uuid" in clause
        assert "da.access_level IN ('view', 'edit', 'owner')" in clause
