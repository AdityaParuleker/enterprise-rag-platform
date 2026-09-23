"""
Unit tests for ContextualCompressor & ACL Neighbor Stitching (Checkpoint 7.3).
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.app.retrieval.compression import ContextualCompressor


@pytest.mark.asyncio
async def test_contextual_compression_sentence_extraction():
    chunk_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    candidates = [
        {
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "text": "Company password length must be 12 chars. Weather today is sunny. Unrelated sentence.",
            "rerank_score": 0.95
        }
    ]

    compressor = ContextualCompressor()
    results = await compressor.compress_candidates(
        query="password length",
        candidates=candidates,
        tenant_id=tenant_id,
        user_id=user_id
    )

    assert len(results) == 1
    res = results[0]

    # Non-mutation check
    assert res["text"] == candidates[0]["text"]
    assert res["chunk_id"] == chunk_id

    # Compressed text contains extracted sentences
    assert "password length" in res["compressed_text"].lower()


@pytest.mark.asyncio
async def test_contextual_compression_non_mutation_contract():
    orig_cand = {
        "chunk_id": str(uuid.uuid4()),
        "document_id": str(uuid.uuid4()),
        "text": "Original text content.",
        "rerank_score": 1.2
    }

    candidates = [orig_cand]
    compressor = ContextualCompressor()
    results = await compressor.compress_candidates(
        query="test query",
        candidates=candidates,
        tenant_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4())
    )

    # Assert original dict was not mutated
    assert "compressed_text" not in orig_cand
    assert results[0]["compressed_text"] == "Original text content."
    assert results[0] is not orig_cand  # Must be shallow copy


@pytest.mark.asyncio
async def test_contextual_compression_fallback_on_zero_matching_sentences():
    chunk_id = str(uuid.uuid4())
    candidates = [
        {
            "chunk_id": chunk_id,
            "text": "Completely unrelated sentence text without overlap.",
            "rerank_score": 0.5
        }
    ]

    compressor = ContextualCompressor()
    results = await compressor.compress_candidates(
        query="password policy",
        candidates=candidates,
        tenant_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4())
    )

    # Fallback to full text when zero matching sentences
    assert results[0]["compressed_text"] == "Completely unrelated sentence text without overlap."


@pytest.mark.asyncio
async def test_contextual_compression_fallback_on_exception():
    candidates = [
        {
            "chunk_id": str(uuid.uuid4()),
            "text": "Valid text.",
            "prev_chunk_id": "invalid-uuid-format"
        }
    ]

    # Compressor handles exception gracefully
    compressor = ContextualCompressor()
    results = await compressor.compress_candidates(
        query="query",
        candidates=candidates,
        tenant_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4())
    )

    assert len(results) == 1
    assert results[0]["compressed_text"] == "Valid text."


@pytest.mark.asyncio
async def test_acl_protected_neighbor_chunk_stitching_success_and_unauthorized_rejection():
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    auth_neighbor_id = str(uuid.uuid4())
    unauth_neighbor_id = str(uuid.uuid4())

    mock_pool = MagicMock()

    # Authorized neighbor returns text; Unauthorized neighbor returns None (blocked by ACL SQL clause)
    async def mock_fetchrow(query, n_uuid, *args):
        if n_uuid == uuid.UUID(auth_neighbor_id):
            return {"text": "Preceding authorized sentence."}
        return None  # Unauthorized neighbor omitted

    mock_pool.fetchrow = AsyncMock(side_effect=mock_fetchrow)

    mock_acl = MagicMock()
    mock_acl.build_acl_where_clause.return_value = ("d.visibility = 'tenant'", [uuid.UUID(user_id), uuid.UUID(tenant_id), []])

    compressor = ContextualCompressor(db_pool=mock_pool, acl_filter=mock_acl)

    # 1. Authorized neighbor stitching
    cand_auth = [
        {
            "chunk_id": str(uuid.uuid4()),
            "text": "Target sentence text.",
            "prev_chunk_id": auth_neighbor_id
        }
    ]
    res_auth = await compressor.compress_candidates("sentence", cand_auth, tenant_id=tenant_id, user_id=user_id)
    assert "Preceding authorized sentence." in res_auth[0]["compressed_text"]
    assert "Target sentence text." in res_auth[0]["compressed_text"]

    # 2. Unauthorized neighbor stitching rejection
    cand_unauth = [
        {
            "chunk_id": str(uuid.uuid4()),
            "text": "Target sentence text.",
            "prev_chunk_id": unauth_neighbor_id
        }
    ]
    res_unauth = await compressor.compress_candidates("sentence", cand_unauth, tenant_id=tenant_id, user_id=user_id)
    # Unauthorized neighbor sentence MUST NOT be stitched
    assert "Preceding authorized sentence." not in res_unauth[0]["compressed_text"]
    assert res_unauth[0]["compressed_text"] == "Target sentence text."
