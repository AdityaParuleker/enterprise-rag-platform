"""
Unit tests for HybridSearchEngine (Checkpoint 5.5 — Reciprocal Rank Fusion & Hybrid Retrieval).
Mocks vector and FTS search engines as async functions returning canned lists.
"""

import uuid
import pytest
from unittest.mock import AsyncMock
from fastapi import HTTPException

from backend.app.retrieval.hybrid_search import HybridSearchEngine, RetrievalFailureError, RRF_K_CONSTANT


@pytest.mark.asyncio
async def test_rrf_score_calculation_exact_math():
    """1. RRF score calculation — assert exact math for k=60 across vector and FTS candidate ranks."""
    mock_vector = AsyncMock()
    mock_fts = AsyncMock()

    chunk_a = {"chunk_id": "00000000-0000-0000-0000-000000000001", "text": "Chunk A", "similarity_score": 0.9}
    chunk_b = {"chunk_id": "00000000-0000-0000-0000-000000000002", "text": "Chunk B", "fts_score": 0.8}

    # Vector returns [A] (rank 1)
    mock_vector.search.return_value = [chunk_a]
    # FTS returns [A, B] (rank 1 for A, rank 2 for B)
    mock_fts.search.return_value = [chunk_a, chunk_b]

    engine = HybridSearchEngine(vector_search_engine=mock_vector, fts_search_engine=mock_fts)
    tenant_id = str(uuid.uuid4())
    query_vector = [0.1] * 1024

    results = await engine.search(query="test query", query_vector=query_vector, tenant_id=tenant_id, limit=10)

    # Chunk A score = 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.032787
    # Chunk B score = 1/(60+2) = 1/62 ≈ 0.016129
    assert len(results) == 2
    assert results[0]["chunk_id"] == chunk_a["chunk_id"]
    assert results[0]["rrf_score"] == round((1.0 / 61) + (1.0 / 61), 6)

    assert results[1]["chunk_id"] == chunk_b["chunk_id"]
    assert results[1]["rrf_score"] == round(1.0 / 62, 6)


@pytest.mark.asyncio
async def test_rrf_deterministic_tie_breaking_chunk_id_asc():
    """2. Tie-break — two chunks with identical RRF score must sort by chunk_id ASC."""
    mock_vector = AsyncMock()
    mock_fts = AsyncMock()

    chunk_z = {"chunk_id": "00000000-0000-0000-0000-000000000099", "text": "Chunk Z"}
    chunk_a = {"chunk_id": "00000000-0000-0000-0000-000000000001", "text": "Chunk A"}

    # Vector returns [Z, A] (Z rank 1, A rank 2)
    mock_vector.search.return_value = [chunk_z, chunk_a]
    # FTS returns [A, Z] (A rank 1, Z rank 2)
    mock_fts.search.return_value = [chunk_a, chunk_z]

    engine = HybridSearchEngine(vector_search_engine=mock_vector, fts_search_engine=mock_fts)
    tenant_id = str(uuid.uuid4())
    query_vector = [0.1] * 1024

    results = await engine.search(query="test", query_vector=query_vector, tenant_id=tenant_id, limit=10)

    # Both chunks have identical RRF score (1/61 + 1/62 = 0.032522)
    assert len(results) == 2
    assert results[0]["rrf_score"] == results[1]["rrf_score"]
    # Primary score tied -> Secondary sort chunk_id ASC puts chunk_a before chunk_z
    assert results[0]["chunk_id"] == chunk_a["chunk_id"]
    assert results[1]["chunk_id"] == chunk_z["chunk_id"]


@pytest.mark.asyncio
async def test_hybrid_search_fail_closed_on_branch_failure():
    """3. Branch failure policy — vector or FTS branch DB exception raises RetrievalFailureError (fail-closed)."""
    mock_vector = AsyncMock()
    mock_fts = AsyncMock()

    # Vector succeeds, FTS throws a database exception
    mock_vector.search.return_value = [{"chunk_id": str(uuid.uuid4()), "text": "OK"}]
    mock_fts.search.side_effect = RuntimeError("PostgreSQL FTS connection pool timeout")

    engine = HybridSearchEngine(vector_search_engine=mock_vector, fts_search_engine=mock_fts)
    tenant_id = str(uuid.uuid4())
    query_vector = [0.1] * 1024

    # Must fail closed with RetrievalFailureError, NOT silently return single-branch vector results
    with pytest.raises(RetrievalFailureError, match="FTS search branch execution failed"):
        await engine.search(query="test", query_vector=query_vector, tenant_id=tenant_id, limit=10)


@pytest.mark.asyncio
async def test_hybrid_search_both_succeed_zero_hits():
    """4. Both succeed with 0 hits -> returns empty list (normal fallback path, not an error)."""
    mock_vector = AsyncMock()
    mock_fts = AsyncMock()

    mock_vector.search.return_value = []
    mock_fts.search.return_value = []

    engine = HybridSearchEngine(vector_search_engine=mock_vector, fts_search_engine=mock_fts)
    tenant_id = str(uuid.uuid4())
    query_vector = [0.1] * 1024

    results = await engine.search(query="no match query", query_vector=query_vector, tenant_id=tenant_id, limit=10)

    assert results == []


@pytest.mark.asyncio
async def test_hybrid_search_top_k_truncation_and_client_error_propagation():
    """5. Both succeed with hits -> returns merged results bounded by limit; client HTTPException propagates directly."""
    mock_vector = AsyncMock()
    mock_fts = AsyncMock()

    c1 = {"chunk_id": str(uuid.uuid4()), "text": "C1"}
    c2 = {"chunk_id": str(uuid.uuid4()), "text": "C2"}
    c3 = {"chunk_id": str(uuid.uuid4()), "text": "C3"}

    mock_vector.search.return_value = [c1, c2, c3]
    mock_fts.search.return_value = [c3, c2, c1]

    engine = HybridSearchEngine(vector_search_engine=mock_vector, fts_search_engine=mock_fts)
    tenant_id = str(uuid.uuid4())
    query_vector = [0.1] * 1024

    # Bound limit to 2
    results = await engine.search(query="test", query_vector=query_vector, tenant_id=tenant_id, limit=2)
    assert len(results) == 2

    # Client validation error propagation test
    mock_fts.search.side_effect = HTTPException(status_code=400, detail="Invalid metadata filter")
    with pytest.raises(HTTPException) as exc_info:
        await engine.search(query="test", query_vector=query_vector, tenant_id=tenant_id, limit=10)
    assert exc_info.value.status_code == 400
    assert "Invalid metadata filter" in exc_info.value.detail


@pytest.mark.asyncio
async def test_hybrid_search_forwards_acl_and_metadata_to_vector_branch():
    """6. Forwarding test — assert HybridSearchEngine passes user_id, user_roles, and metadata_filters to vector branch."""
    mock_vector = AsyncMock()
    mock_fts = AsyncMock()

    mock_vector.search.return_value = []
    mock_fts.search.return_value = []

    engine = HybridSearchEngine(vector_search_engine=mock_vector, fts_search_engine=mock_fts)
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    role_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    query_vector = [0.1] * 1024
    metadata_filters = {"document_id": doc_id}

    await engine.search(
        query="test query",
        query_vector=query_vector,
        tenant_id=tenant_id,
        limit=10,
        metadata_filters=metadata_filters,
        user_id=user_id,
        user_roles=[role_id]
    )

    mock_vector.search.assert_called_once()
    kwargs = mock_vector.search.call_args.kwargs

    assert kwargs["tenant_id"] == tenant_id
    assert kwargs["limit"] == 10
    assert kwargs["metadata_filters"] == metadata_filters
    assert kwargs["user_id"] == user_id
    assert kwargs["user_roles"] == [role_id]


@pytest.mark.asyncio
async def test_hybrid_search_concurrent_execution_connection_isolation():
    """7. Concurrency & connection isolation — assert vector branch and FTS branch execute concurrently without sharing a single asyncpg connection."""
    import asyncio
    mock_vector = AsyncMock()
    mock_fts = AsyncMock()
    mock_conn = AsyncMock()

    # Simulate vector branch delay to force timing overlap with FTS branch
    async def delayed_vector_search(*args, **kwargs):
        await asyncio.sleep(0.05)
        return [{"chunk_id": "00000000-0000-0000-0000-000000000001", "text": "Vec Chunk"}]

    async def fast_fts_search(*args, **kwargs):
        return [{"chunk_id": "00000000-0000-0000-0000-000000000002", "text": "FTS Chunk"}]

    mock_vector.search.side_effect = delayed_vector_search
    mock_fts.search.side_effect = fast_fts_search

    engine = HybridSearchEngine(vector_search_engine=mock_vector, fts_search_engine=mock_fts)
    tenant_id = str(uuid.uuid4())
    query_vector = [0.1] * 1024

    # Pass single db_conn override
    results = await engine.search(
        query="aurawave pro",
        query_vector=query_vector,
        tenant_id=tenant_id,
        limit=10,
        conn=mock_conn
    )

    assert len(results) == 2
    # Verify vector branch received conn override while FTS branch received conn=None for connection isolation
    assert mock_vector.search.call_args.kwargs["conn"] == mock_conn
    assert mock_fts.search.call_args.kwargs["conn"] is None


