"""
Unit tests for VectorSearchEngine (Checkpoint 4.5 — Basic RAG Vector Search).
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.app.retrieval.vector_search import VectorSearchEngine, VECTOR_DIMENSION


@pytest.mark.asyncio
async def test_vector_search_dimension_validation():
    engine = VectorSearchEngine()
    tenant_id = str(uuid.uuid4())

    # Invalid types
    with pytest.raises(ValueError, match="query_vector must be a list or tuple"):
        await engine.search("not_a_list", tenant_id)

    # 512 dimensions instead of 1024
    with pytest.raises(ValueError, match="Query vector dimension mismatch"):
        await engine.search([0.1] * 512, tenant_id)

    # 1025 dimensions
    with pytest.raises(ValueError, match="Query vector dimension mismatch"):
        await engine.search([0.1] * 1025, tenant_id)

    # Empty tenant_id
    with pytest.raises(ValueError, match="tenant_id must be a non-empty string"):
        await engine.search([0.1] * 1024, "")


@pytest.mark.asyncio
async def test_vector_search_parameterized_query_and_sql_structure():
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = []

    engine = VectorSearchEngine()
    tenant_id = str(uuid.uuid4())
    query_vector = [0.05] * VECTOR_DIMENSION

    results = await engine.search(query_vector, tenant_id=tenant_id, limit=5, conn=mock_conn)

    assert results == []
    mock_conn.fetch.assert_called_once()
    call_args = mock_conn.fetch.call_args[0]

    # 1. SQL string verification
    sql_text = call_args[0]
    assert "FROM chunks c" in sql_text
    assert "JOIN documents d ON c.document_id = d.id" in sql_text
    assert "c.tenant_id = $2::uuid" in sql_text
    assert "d.is_latest = TRUE" in sql_text
    assert "d.status = 'INDEXED'" in sql_text
    assert "ORDER BY distance ASC, c.id ASC" in sql_text
    assert "LIMIT $3" in sql_text


    # 2. Parameter values verification
    vector_param, tenant_param, limit_param = call_args[1], call_args[2], call_args[3]
    assert vector_param.startswith("[") and vector_param.endswith("]")
    assert tenant_param == tenant_id
    assert limit_param == 5


@pytest.mark.asyncio
async def test_vector_search_results_mapping_and_similarity_score():
    mock_conn = AsyncMock()
    chunk_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())

    mock_row = {
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "document_version": 2,
        "text": "Extracted context text for RAG retrieval.",
        "page_number": 1,
        "section_path": "Introduction > Overview",
        "metadata": '{"author": "Alice", "tags": ["test"]}',
        "distance": 0.15,
        "similarity_score": 0.85,
    }
    mock_conn.fetch.return_value = [mock_row]

    engine = VectorSearchEngine()
    query_vector = [0.1] * VECTOR_DIMENSION

    results = await engine.search(query_vector, tenant_id=tenant_id, limit=10, conn=mock_conn)

    assert len(results) == 1
    res = results[0]
    assert res["chunk_id"] == chunk_id
    assert res["document_id"] == doc_id
    assert res["document_version"] == 2
    assert res["text"] == "Extracted context text for RAG retrieval."
    assert res["page_number"] == 1
    assert res["section_path"] == "Introduction > Overview"
    assert res["metadata"] == {"author": "Alice", "tags": ["test"]}
    assert res["distance"] == 0.15
    assert res["similarity_score"] == 0.85


@pytest.mark.asyncio
async def test_vector_search_deterministic_ordering_and_top_k():
    mock_conn = AsyncMock()
    tenant_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    chunk1_id = "00000000-0000-0000-0000-000000000001"
    chunk2_id = "00000000-0000-0000-0000-000000000002"

    rows = [
        {
            "chunk_id": chunk1_id,
            "document_id": doc_id,
            "document_version": 1,
            "text": "Chunk 1 text",
            "page_number": 1,
            "section_path": "Sec 1",
            "metadata": {},
            "distance": 0.10,
            "similarity_score": 0.90,
        },
        {
            "chunk_id": chunk2_id,
            "document_id": doc_id,
            "document_version": 1,
            "text": "Chunk 2 text",
            "page_number": 2,
            "section_path": "Sec 2",
            "metadata": {},
            "distance": 0.20,
            "similarity_score": 0.80,
        }
    ]
    mock_conn.fetch.return_value = rows

    engine = VectorSearchEngine()
    query_vector = [0.0] * VECTOR_DIMENSION

    results = await engine.search(query_vector, tenant_id=tenant_id, limit=2, conn=mock_conn)

    assert len(results) == 2
    assert results[0]["chunk_id"] == chunk1_id
    assert results[0]["distance"] < results[1]["distance"]
    assert results[0]["similarity_score"] > results[1]["similarity_score"]


@pytest.mark.asyncio
async def test_vector_search_empty_limit_or_no_matches():
    mock_conn = AsyncMock()
    engine = VectorSearchEngine()
    tenant_id = str(uuid.uuid4())
    query_vector = [0.1] * VECTOR_DIMENSION

    # Zero or negative limit returns empty list without DB call
    assert await engine.search(query_vector, tenant_id, limit=0, conn=mock_conn) == []
    assert await engine.search(query_vector, tenant_id, limit=-5, conn=mock_conn) == []
    mock_conn.fetch.assert_not_called()

    # DB returning 0 rows returns empty list
    mock_conn.fetch.return_value = []
    res = await engine.search(query_vector, tenant_id, limit=10, conn=mock_conn)
    assert res == []


@pytest.mark.asyncio
async def test_vector_search_with_db_pool():
    mock_pool = AsyncMock()
    mock_pool.fetch.return_value = []

    engine = VectorSearchEngine(db_pool=mock_pool)
    tenant_id = str(uuid.uuid4())
    query_vector = [0.1] * VECTOR_DIMENSION

    res = await engine.search(query_vector, tenant_id=tenant_id, limit=3)
    assert res == []
    mock_pool.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_vector_search_combined_with_metadata_and_acl_filters():
    """Verify VectorSearchEngine appends metadata filters and ACL pushdown clauses with correct parameter offsets."""
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = []

    engine = VectorSearchEngine()
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    role_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    query_vector = [0.05] * VECTOR_DIMENSION
    metadata_filters = {"document_id": doc_id, "source_type": "pdf"}

    await engine.search(
        query_vector=query_vector,
        tenant_id=tenant_id,
        limit=10,
        metadata_filters=metadata_filters,
        user_id=user_id,
        user_roles=[role_id],
        conn=mock_conn
    )

    mock_conn.fetch.assert_called_once()
    sql_text = mock_conn.fetch.call_args[0][0]
    sql_params = mock_conn.fetch.call_args[0][1:]

    # $1=vector_str, $2=clean_tenant_id
    assert "(c.embedding <=> $1::vector) AS distance" in sql_text
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

    assert sql_params[1] == tenant_id
    assert sql_params[2] == [uuid.UUID(doc_id)]
    assert sql_params[3] == ["pdf"]
    assert sql_params[4] == uuid.UUID(user_id)
    assert sql_params[5] == uuid.UUID(tenant_id)
    assert sql_params[6] == [uuid.UUID(role_id)]
    assert sql_params[7] == 10

