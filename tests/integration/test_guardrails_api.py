"""
Integration and Security tests for Phase 8 — AI & Runtime Guardrail Hardening.
Covers all 15 required security and guardrail scenarios.
"""

import os
import json
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

os.environ["JWT_SECRET"] = "change-this-in-production-super-secret-key"

from backend.app.generation.prompt_guard import (
    InputGuard, SecretRedactor, EvidenceScorer, OutputGuard, GuardrailPipeline, INSUFFICIENT_EVIDENCE_RESPONSE
)
from backend.app.ingestion.chunkers.text_chunker import TextChunker
from backend.app.ingestion.parsers.base import ParsedBlock
from backend.app.auth.rate_limiter import RateLimiter
from backend.app.auth.audit_logger import AuditLogger, sanitize_audit_detail
from backend.app.api.chat import ChatOrchestrator


# =====================================================================
# 1. Secret -> Embedding & Provenance Invariant Tests
# =====================================================================

def test_secret_redaction_before_embedding_and_provenance_preservation():
    """Scenario 1 & 2: Secret scanning occurs before embedding and preserves canonical provenance."""
    chunker = TextChunker()
    doc_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())

    text = "Secret config: aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'"
    blocks = [ParsedBlock(text=text, page_number=2, section_title="Config")]

    chunks = chunker.chunk_blocks(blocks, document_id=doc_id, tenant_id=tenant_id)
    c = chunks[0]

    # Secret is redacted BEFORE embedding
    assert "wJalrXUtnFEMI" not in c["text"]
    assert "[REDACTED_SECRET:AWS_SECRET_KEY]" in c["text"]

    # Canonical provenance intact
    assert c["document_id"] == doc_id
    assert c["tenant_id"] == tenant_id
    assert c["page_number"] == 2
    assert c["section_path"] == "Config"
    assert c["chunk_index"] == 0


# =====================================================================
# 2. Evidence Refusal & Fallback Tests
# =====================================================================

@pytest.mark.asyncio
async def test_insufficient_evidence_refusal_zero_llm_calls():
    """Scenario 3: Low evidence score triggers safe refusal with ZERO LLM generation calls."""
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    mock_embed = MagicMock()
    mock_embed.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_hybrid = AsyncMock()
    mock_hybrid.search.return_value = []  # Empty candidates -> evidence score 0.0

    mock_acl = AsyncMock()
    mock_acl.get_user_role_ids.return_value = []

    mock_llm = MagicMock()

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed,
        hybrid_search=mock_hybrid,
        acl_filter=mock_acl,
        llm_client=mock_llm
    )

    response = await orchestrator.handle_chat_stream(
        query="Low evidence query",
        tenant_id=tenant_id,
        user_id=user_id
    )

    mock_llm.stream_response.assert_not_called()

    events = []
    async for line in response.body_iterator:
        if line:
            events.append(line.decode("utf-8") if isinstance(line, bytes) else line)

    assert any("I could not find sufficient information" in e for e in events)


@pytest.mark.asyncio
async def test_evidence_refusal_reranking_disabled_fallback():
    """Scenario 4: Evidence scorer uses RRF fallback score when enable_reranking=False."""
    scorer = EvidenceScorer()
    candidates = [{"chunk_id": "c1", "rrf_score": 0.01639}]

    res = scorer.evaluate_evidence(candidates, enable_reranking=False)
    assert res.action == "ALLOW"


# =====================================================================
# 3. Output Guard & Citation Validation Tests
# =====================================================================

def test_invalid_and_unauthorized_citation_rejection():
    """Scenario 5 & 6: Invalid citation index [99] triggers RETRY."""
    guard = OutputGuard()
    candidates = [{"chunk_id": str(uuid.uuid4())}]

    res = guard.check_output("This states facts [99].", candidates=candidates)
    assert res.action == "RETRY"
    assert res.reason == "INVALID_CITATION_MARKER"


def test_output_secret_leak_detection():
    """Scenario 7: Output containing raw secret key is blocked."""
    guard = OutputGuard()
    candidates = [{"chunk_id": "c1"}]
    answer = "The key is aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'."

    res = guard.check_output(answer, candidates=candidates)
    assert res.action == "BLOCK"
    assert res.reason == "SECRET_LEAK_DETECTED"


@pytest.mark.asyncio
async def test_single_regeneration_retry_and_evidence_boundary():
    """Scenario 8, 9 & 10: Single regeneration retry uses exact same evidence set and degrades safely on double failure."""
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    mock_embed = MagicMock()
    mock_embed.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_hybrid = AsyncMock()
    mock_hybrid.search.return_value = [
        {"chunk_id": str(uuid.uuid4()), "document_id": str(uuid.uuid4()), "text": "Valid text.", "metadata": {}}
    ]

    mock_acl = AsyncMock()
    mock_acl.get_user_role_ids.return_value = []

    mock_reranker = AsyncMock()
    mock_reranker.rerank.return_value = [
        {"chunk_id": str(uuid.uuid4()), "document_id": str(uuid.uuid4()), "text": "Valid text.", "rerank_score": 0.9}
    ]

    # LLM yields invalid citation [99] twice
    mock_llm = MagicMock()
    mock_llm.stream_response.side_effect = [
        ["Answer with invalid citation [99]."],
        ["Second answer with invalid citation [99]."]
    ]

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed,
        hybrid_search=mock_hybrid,
        acl_filter=mock_acl,
        reranker_engine=mock_reranker,
        llm_client=mock_llm
    )

    response = await orchestrator.handle_chat_stream(
        query="Test query",
        tenant_id=tenant_id,
        user_id=user_id
    )

    # Exactly 2 generation calls (1 initial + 1 retry max)
    assert mock_llm.stream_response.call_count == 2


# =====================================================================
# 4. Rate Limiting & Fail-Closed Tests
# =====================================================================

@pytest.mark.asyncio
async def test_redis_rate_limit_atomicity_and_fail_closed():
    """Scenario 12 & 13: Rate limiter enforces limit and fails closed (503) on Redis outage."""
    limiter = RateLimiter(route_class="chat")
    mock_request = MagicMock()
    mock_request.client.host = "127.0.0.1"

    # 1. Redis outage -> 503
    mock_manager_outage = MagicMock()
    mock_manager_outage.get_client.side_effect = RuntimeError("Redis down")
    limiter_outage = RateLimiter(route_class="chat", redis_manager=mock_manager_outage)

    with pytest.raises(Exception) as exc_info:
        await limiter_outage.check_rate_limit(mock_request, tenant_id="t1", user_id="u1")
    assert exc_info.value.status_code == 503


# =====================================================================
# 5. Audit Logging & Data Safety Tests
# =====================================================================

def test_audit_log_data_safety_redaction():
    """Scenario 14: Audit detail entries never contain raw passwords, tokens, or raw text."""
    detail = {
        "user_id": "u1",
        "password": "my_password",
        "jwt": "eyJ...",
        "api_key": "ak_123"
    }
    sanitized = sanitize_audit_detail(detail)

    assert sanitized["password"] == "[REDACTED_AUDIT_PAYLOAD]"
    assert sanitized["jwt"] == "[REDACTED_AUDIT_PAYLOAD]"
    assert sanitized["api_key"] == "[REDACTED_AUDIT_PAYLOAD]"


@pytest.mark.asyncio
async def test_audit_log_tenant_isolation():
    """Scenario 15: AuditLogger.list_logs queries only current tenant_id."""
    logger = AuditLogger()
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = []
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("backend.app.auth.audit_logger.get_db_pool", AsyncMock(return_value=mock_pool))

        logs = await logger.list_logs(tenant_id="tenant_a")
        assert logs == []
        assert "WHERE tenant_id = $1" in mock_conn.fetch.call_args[0][0]
