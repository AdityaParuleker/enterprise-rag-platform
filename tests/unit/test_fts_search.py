"""
Unit tests for FTSSearchEngine (Checkpoint 5.4 — PostgreSQL Full-Text Search Engine).
Uses mocked asyncpg connections for fast, isolated execution.
"""

import uuid
import pytest
from unittest.mock import AsyncMock

from backend.app.retrieval.fts_search import FTSSearchEngine


@pytest.mark.asyncio
async def test_fts_search_keyword_match_safe_tsquery():
    """1. Keyword match — assert websearch_to_tsquery('english', $1) used, not raw string concat."""
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = []

    engine = FTSSearchEngine()
    tenant_id = str(uuid.uuid4())
    user_query = "quantum computing & encryption's \"security\""

    results = await engine.search(query=user_query, tenant_id=tenant_id, limit=10, conn=mock_conn)

    assert results == []
    mock_conn.fetch.assert_called_once()
    sql_text = mock_conn.fetch.call_args[0][0]
    sql_params = mock_conn.fetch.call_args[0][1:]

    # Assert safe websearch_to_tsquery is used
    assert "websearch_to_tsquery('english', $1)" in sql_text
    assert "c.tsv @@ websearch_to_tsquery('english', $1)" in sql_text
    assert "ts_rank_cd(c.tsv, websearch_to_tsquery('english', $1)) AS fts_score" in sql_text
    assert sql_params[0] == user_query


@pytest.mark.asyncio
async def test_fts_search_tenant_boundary_always_present():
    """2. Tenant boundary — assert tenant_id param always present in the query."""
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = []

    engine = FTSSearchEngine()
    tenant_id = str(uuid.uuid4())

    await engine.search(query="test query", tenant_id=tenant_id, limit=5, conn=mock_conn)

    sql_text = mock_conn.fetch.call_args[0][0]
    sql_params = mock_conn.fetch.call_args[0][1:]

    assert "c.tenant_id = $2::uuid" in sql_text
    assert "d.is_latest = TRUE" in sql_text
    assert "d.status = 'INDEXED'" in sql_text
    assert sql_params[1] == tenant_id


@pytest.mark.asyncio
async def test_fts_search_combined_with_metadata_filters():
    """3. Filters + FTS combined — assert metadata filter clause is appended correctly."""
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = []

    engine = FTSSearchEngine()
    tenant_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    metadata_filters = {
        "document_id": doc_id,
        "source_type": "pdf",
        "page_number": {"min": 1, "max": 10}
    }

    await engine.search(
        query="search text",
        tenant_id=tenant_id,
        limit=15,
        metadata_filters=metadata_filters,
        conn=mock_conn
    )

    sql_text = mock_conn.fetch.call_args[0][0]
    sql_params = mock_conn.fetch.call_args[0][1:]

    assert "c.document_id = ANY($3::uuid[])" in sql_text
    assert "d.source_type = ANY($4::text[])" in sql_text
    assert "c.page_number >= $5" in sql_text
    assert "c.page_number <= $6" in sql_text
    assert "LIMIT $7" in sql_text

    assert sql_params[2] == [uuid.UUID(doc_id)]
    assert sql_params[3] == ["pdf"]
    assert sql_params[4] == 1
    assert sql_params[5] == 10
    assert sql_params[6] == 15


@pytest.mark.asyncio
async def test_fts_search_combined_with_acl_filter():
    """4. ACL combined — assert ACL clause is appended correctly."""
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = []

    engine = FTSSearchEngine()
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    role_id = str(uuid.uuid4())

    await engine.search(
        query="retrieval keyword",
        tenant_id=tenant_id,
        limit=20,
        user_id=user_id,
        user_roles=[role_id],
        conn=mock_conn
    )

    sql_text = mock_conn.fetch.call_args[0][0]
    sql_params = mock_conn.fetch.call_args[0][1:]

    # Offset 3 -> $3=user_id, $4=tenant_id, $5=role_ids
    assert "d.owner_id = $3::uuid" in sql_text
    assert "da.principal_id = $3::uuid" in sql_text
    assert "ANY($5::uuid[])" in sql_text
    assert "da.principal_id = $4::uuid" in sql_text
    assert "LIMIT $6" in sql_text

    assert sql_params[2] == uuid.UUID(user_id)
    assert sql_params[3] == uuid.UUID(tenant_id)
    assert sql_params[4] == [uuid.UUID(role_id)]
    assert sql_params[5] == 20


@pytest.mark.asyncio
async def test_fts_search_coexisting_metadata_and_acl_offset_composition():
    """5. Coexisting composition — assert FTS ($1/$2) + metadata ($3/$4) + ACL ($5/$6/$7) + limit ($8) align perfectly."""
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = []

    engine = FTSSearchEngine()
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    role_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    metadata_filters = {
        "document_id": doc_id,
        "source_type": "pdf"
    }

    await engine.search(
        query="coexisting search",
        tenant_id=tenant_id,
        limit=25,
        metadata_filters=metadata_filters,
        user_id=user_id,
        user_roles=[role_id],
        conn=mock_conn
    )

    sql_text = mock_conn.fetch.call_args[0][0]
    sql_params = mock_conn.fetch.call_args[0][1:]

    # $1=query, $2=tenant_id
    assert "websearch_to_tsquery('english', $1)" in sql_text
    assert "c.tenant_id = $2::uuid" in sql_text

    # $3=document_id, $4=source_type
    assert "c.document_id = ANY($3::uuid[])" in sql_text
    assert "d.source_type = ANY($4::text[])" in sql_text

    # $5=user_id, $6=tenant_id, $7=role_ids
    assert "d.owner_id = $5::uuid" in sql_text
    assert "da.principal_id = $5::uuid" in sql_text
    assert "da.principal_id = $6::uuid" in sql_text
    assert "ANY($7::uuid[])" in sql_text

    # $8=limit
    assert "LIMIT $8" in sql_text

    # Exact parameter binding array verification
    assert len(sql_params) == 8
    assert sql_params[0] == "coexisting search"
    assert sql_params[1] == tenant_id
    assert sql_params[2] == [uuid.UUID(doc_id)]
    assert sql_params[3] == ["pdf"]
    assert sql_params[4] == uuid.UUID(user_id)
    assert sql_params[5] == uuid.UUID(tenant_id)
    assert sql_params[6] == [uuid.UUID(role_id)]
    assert sql_params[7] == 25

