"""
Unit tests for DocumentACLFilter (Checkpoint 5.2 — Document Access Control Engine).
"""

import uuid
import pytest
from unittest.mock import AsyncMock

from backend.app.retrieval.acl_filter import DocumentACLFilter


@pytest.mark.asyncio
async def test_get_user_role_ids_validation():
    acl_filter = DocumentACLFilter()
    user_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())

    with pytest.raises(ValueError, match="user_id must be a non-empty UUID string"):
        await acl_filter.get_user_role_ids("", tenant_id)

    with pytest.raises(ValueError, match="tenant_id must be a non-empty UUID string"):
        await acl_filter.get_user_role_ids(user_id, "")

    with pytest.raises(ValueError, match="Invalid UUID format"):
        await acl_filter.get_user_role_ids("not-a-uuid", tenant_id)

    with pytest.raises(ValueError, match="Invalid UUID format"):
        await acl_filter.get_user_role_ids(user_id, "invalid-tenant")


@pytest.mark.asyncio
async def test_get_user_role_ids_query_execution():
    mock_conn = AsyncMock()
    role_id_1 = str(uuid.uuid4())
    role_id_2 = str(uuid.uuid4())
    mock_conn.fetch.return_value = [
        {"role_id": uuid.UUID(role_id_1)},
        {"role_id": uuid.UUID(role_id_2)},
    ]

    acl_filter = DocumentACLFilter()
    user_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())

    role_ids = await acl_filter.get_user_role_ids(user_id, tenant_id, conn=mock_conn)

    assert role_ids == [role_id_1, role_id_2]
    mock_conn.fetch.assert_called_once()
    sql_text = mock_conn.fetch.call_args[0][0]

    assert "FROM user_roles ur" in sql_text
    assert "JOIN users u ON ur.user_id = u.id" in sql_text
    assert "JOIN tenants t ON ur.tenant_id = t.id" in sql_text
    assert "WHERE ur.user_id = $1" in sql_text
    assert "AND ur.tenant_id = $2" in sql_text
    assert "AND u.is_active = TRUE" in sql_text
    assert "AND t.is_active = TRUE" in sql_text

    u_param, t_param = mock_conn.fetch.call_args[0][1], mock_conn.fetch.call_args[0][2]
    assert u_param == uuid.UUID(user_id)
    assert t_param == uuid.UUID(tenant_id)


def test_build_acl_where_clause_validation():
    acl_filter = DocumentACLFilter()
    user_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())

    with pytest.raises(ValueError, match="user_id must be a non-empty UUID string"):
        acl_filter.build_acl_where_clause("", tenant_id)

    with pytest.raises(ValueError, match="tenant_id must be a non-empty UUID string"):
        acl_filter.build_acl_where_clause(user_id, "")

    with pytest.raises(ValueError, match="param_offset must be >= 1"):
        acl_filter.build_acl_where_clause(user_id, tenant_id, param_offset=0)

    with pytest.raises(ValueError, match="Invalid role UUID format"):
        acl_filter.build_acl_where_clause(user_id, tenant_id, user_roles=["bad-role-uuid"])


def test_build_acl_where_clause_structure_and_placeholders():
    acl_filter = DocumentACLFilter()
    user_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())
    role1 = str(uuid.uuid4())
    role2 = str(uuid.uuid4())

    # Default offset = 1 -> $1, $2, $3
    clause, params = acl_filter.build_acl_where_clause(
        user_id=user_id,
        tenant_id=tenant_id,
        user_roles=[role1, role2],
        param_offset=1
    )

    assert "$1::uuid" in clause
    assert "$2::uuid" in clause
    assert "ANY($3::uuid[])" in clause
    assert "d.owner_id = $1::uuid" in clause
    assert "d.visibility = 'tenant'" in clause
    assert "da.access_level IN ('view', 'edit', 'owner')" in clause

    assert len(params) == 3
    assert params[0] == uuid.UUID(user_id)
    assert params[1] == uuid.UUID(tenant_id)
    assert params[2] == [uuid.UUID(role1), uuid.UUID(role2)]


def test_build_acl_where_clause_custom_offset():
    acl_filter = DocumentACLFilter()
    user_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())

    # Custom offset = 4 -> $4, $5, $6
    clause, params = acl_filter.build_acl_where_clause(
        user_id=user_id,
        tenant_id=tenant_id,
        user_roles=[],
        param_offset=4
    )

    assert "$4::uuid" in clause
    assert "$5::uuid" in clause
    assert "ANY($6::uuid[])" in clause
    assert "d.owner_id = $4::uuid" in clause
    assert params[2] == []


def test_build_acl_where_clause_literal_connectors():
    """Verify literal top-level OR connectors between owner, visibility, and EXISTS ACL branches."""
    acl_filter = DocumentACLFilter()
    user_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())

    clause, _ = acl_filter.build_acl_where_clause(user_id, tenant_id)

    # Verify top-level branches are joined by OR, preventing accidental OR->AND mutation
    assert "OR d.visibility = 'tenant'" in clause
    assert "OR EXISTS (" in clause
    assert "OR (da.principal_type = 'role' AND da.principal_id = ANY($3::uuid[]))" in clause
    assert "OR (da.principal_type = 'tenant' AND da.principal_id = $2::uuid)" in clause


@pytest.mark.asyncio
async def test_build_acl_filter_for_user_convenience_method():
    """Verify build_acl_filter_for_user automatically fetches active role_ids and builds WHERE clause."""
    mock_conn = AsyncMock()
    role_id = str(uuid.uuid4())
    mock_conn.fetch.return_value = [{"role_id": uuid.UUID(role_id)}]

    acl_filter = DocumentACLFilter()
    user_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())

    clause, params = await acl_filter.build_acl_filter_for_user(user_id, tenant_id, conn=mock_conn, param_offset=1)

    assert "d.owner_id = $1::uuid" in clause
    assert params[0] == uuid.UUID(user_id)
    assert params[1] == uuid.UUID(tenant_id)
    assert params[2] == [uuid.UUID(role_id)]

