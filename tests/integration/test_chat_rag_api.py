"""
Integration and Unit tests for End-to-End Chat API RAG Orchestration (Checkpoint 4.10 Corrections).
"""

import os
import json
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

# Ensure JWT_SECRET is configured for test token generation
os.environ["JWT_SECRET"] = "change-this-in-production-super-secret-key"

from backend.app.main import app
from backend.app.auth.jwt import create_access_token
from backend.app.api.chat import ChatOrchestrator


client = TestClient(app)


def get_test_auth_headers(user_id: str = None, tenant_id: str = None) -> dict:
    u_id = user_id or str(uuid.uuid4())
    t_id = tenant_id or str(uuid.uuid4())
    token = create_access_token({"sub": u_id, "tenant_id": t_id, "email": "test@example.com"})
    return {"Authorization": f"Bearer {token}"}


def test_chat_unauthenticated_returns_401():
    response = client.post("/api/v1/chat", json={"query": "What is the policy?"})
    assert response.status_code == 401


def test_chat_empty_query_returns_400():
    headers = get_test_auth_headers()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "backend.app.auth.rbac.get_user_permissions",
            AsyncMock(return_value={"chat"})
        )
        response = client.post("/api/v1/chat", json={"query": "   "}, headers=headers)
        assert response.status_code == 400
        assert "Query or message string is required" in str(response.json())


def test_chat_permission_denied_returns_403():
    headers = get_test_auth_headers()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "backend.app.auth.rbac.get_user_permissions",
            AsyncMock(return_value={"view_documents"})
        )
        response = client.post("/api/v1/chat", json={"query": "What is policy?"}, headers=headers)
        assert response.status_code == 403
        assert "Permission 'chat' required" in str(response.json())


@pytest.mark.asyncio
async def test_chat_orchestrator_success_flow_and_async_embed_awaited():
    user_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())

    mock_embed_provider = MagicMock()
    mock_embed_provider.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_vector_search = AsyncMock()
    mock_vector_search.search.return_value = [
        {
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "document_version": 1,
            "text": "Company security policy states password length must be 12 characters.",
            "page_number": 2,
            "section_path": "Security > Passwords",
            "metadata": {},
            "distance": 0.1,
            "similarity_score": 0.9,
        }
    ]

    mock_llm_client = MagicMock()
    mock_llm_client.stream_response.return_value = iter(["Password ", "length ", "is ", "12."])

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed_provider,
        vector_search=mock_vector_search,
        llm_client=mock_llm_client
    )

    query_str = "What is the minimum password length?"
    response = await orchestrator.handle_chat_stream(
        query=query_str,
        tenant_id=tenant_id,
        top_k=5
    )

    assert response.media_type == "text/event-stream"
    assert response.headers["Cache-Control"] == "no-cache"

    # 1. Verify embedding provider is awaited with exact query
    mock_embed_provider.embed.assert_awaited_once_with(query_str)

    # 2. Verify vector search called with exact unchanged tenant_id and candidate pool limit 30
    mock_vector_search.search.assert_called_once()
    call_kwargs = mock_vector_search.search.call_args[1]
    assert call_kwargs["tenant_id"] == tenant_id
    assert call_kwargs["limit"] == 30

    # 3. Verify LLM receives system prompt and user message containing retrieved context
    mock_llm_client.stream_response.assert_called_once()
    llm_kwargs = mock_llm_client.stream_response.call_args[1]
    assert "CRITICAL SECURITY & BEHAVIORAL DIRECTIVES" in llm_kwargs["system"]
    assert "Company security policy states password length must be 12 characters." in llm_kwargs["prompt"]
    assert query_str in llm_kwargs["prompt"]

    # Consume SSE events
    sse_events = []
    async for line in response.body_iterator:
        if line:
            sse_events.append(line.decode("utf-8") if isinstance(line, bytes) else line)

    parsed = [json.loads(e[6:-2]) for e in sse_events if e.startswith("data: ")]

    tokens = [p["content"] for p in parsed if p["type"] == "token"]
    assert "".join(tokens) == "Password length is 12."

    citations = [p["citation"] for p in parsed if p["type"] == "citation"]
    assert len(citations) == 1
    assert citations[0]["chunk_id"] == chunk_id

    dones = [p for p in parsed if p["type"] == "done"]
    assert len(dones) == 1
    assert dones[0]["metadata"]["citations_count"] == 1


@pytest.mark.asyncio
async def test_chat_orchestrator_fallback_does_not_call_llm():
    tenant_id = str(uuid.uuid4())

    mock_embed_provider = MagicMock()
    mock_embed_provider.embed = AsyncMock(return_value=[0.0] * 1024)

    mock_vector_search = AsyncMock()
    mock_vector_search.search.return_value = []

    mock_llm_client = MagicMock()

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed_provider,
        vector_search=mock_vector_search,
        llm_client=mock_llm_client
    )

    response = await orchestrator.handle_chat_stream(
        query="Where is the non-existent policy?",
        tenant_id=tenant_id
    )

    # Verify LLM is NOT called when context is missing
    mock_llm_client.stream_response.assert_not_called()

    sse_events = []
    async for line in response.body_iterator:
        if line:
            sse_events.append(line.decode("utf-8") if isinstance(line, bytes) else line)

    parsed = [json.loads(e[6:-2]) for e in sse_events if e.startswith("data: ")]

    tokens = [p["content"] for p in parsed if p["type"] == "token"]
    assert len(tokens) == 1
    assert any(phrase in tokens[0] for phrase in ["No relevant document context found", "could not find sufficient information"])

    citations = [p for p in parsed if p["type"] == "citation"]
    assert len(citations) == 0

    dones = [p for p in parsed if p["type"] == "done"]
    assert len(dones) == 1
    assert dones[0]["metadata"].get("fallback") is True or dones[0]["metadata"].get("refusal") is True


@pytest.mark.asyncio
async def test_chat_orchestrator_embedding_failure_event():
    tenant_id = str(uuid.uuid4())

    mock_embed_provider = MagicMock()
    mock_embed_provider.embed = AsyncMock(side_effect=RuntimeError("Embedding model offline"))

    orchestrator = ChatOrchestrator(embedding_provider=mock_embed_provider)

    response = await orchestrator.handle_chat_stream(
        query="Query triggering embed failure",
        tenant_id=tenant_id
    )

    sse_events = []
    async for line in response.body_iterator:
        if line:
            sse_events.append(line.decode("utf-8") if isinstance(line, bytes) else line)

    parsed = [json.loads(e[6:-2]) for e in sse_events if e.startswith("data: ")]

    errors = [p for p in parsed if p["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["code"] == "EMBEDDING_FAILURE"
    assert "Embedding model offline" in errors[0]["error"]


@pytest.mark.asyncio
async def test_chat_orchestrator_retrieval_failure_event():
    tenant_id = str(uuid.uuid4())

    mock_embed_provider = MagicMock()
    mock_embed_provider.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_vector_search = AsyncMock()
    mock_vector_search.search.side_effect = RuntimeError("Database connection timeout")

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed_provider,
        vector_search=mock_vector_search
    )

    response = await orchestrator.handle_chat_stream(
        query="Query triggering retrieval failure",
        tenant_id=tenant_id
    )

    sse_events = []
    async for line in response.body_iterator:
        if line:
            sse_events.append(line.decode("utf-8") if isinstance(line, bytes) else line)

    parsed = [json.loads(e[6:-2]) for e in sse_events if e.startswith("data: ")]

    errors = [p for p in parsed if p["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["code"] == "RETRIEVAL_FAILURE"
    assert "Database connection timeout" in errors[0]["error"]


@pytest.mark.asyncio
async def test_chat_orchestrator_generation_failure_event():
    tenant_id = str(uuid.uuid4())

    mock_embed_provider = MagicMock()
    mock_embed_provider.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_vector_search = AsyncMock()
    mock_vector_search.search.return_value = [
        {
            "chunk_id": str(uuid.uuid4()),
            "document_id": str(uuid.uuid4()),
            "document_version": 1,
            "text": "Context text.",
            "page_number": 1,
            "section_path": "Sec",
            "metadata": {},
            "distance": 0.1,
            "similarity_score": 0.9,
        }
    ]

    mock_llm_client = MagicMock()
    mock_llm_client.stream_response.side_effect = RuntimeError("Ollama LLM service unavailable")

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed_provider,
        vector_search=mock_vector_search,
        llm_client=mock_llm_client
    )

    response = await orchestrator.handle_chat_stream(
        query="Query triggering LLM failure",
        tenant_id=tenant_id
    )

    sse_events = []
    async for line in response.body_iterator:
        if line:
            sse_events.append(line.decode("utf-8") if isinstance(line, bytes) else line)

    parsed = [json.loads(e[6:-2]) for e in sse_events if e.startswith("data: ")]

    errors = [p for p in parsed if p["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["code"] == "GENERATION_FAILURE"
    assert "Ollama LLM service unavailable" in errors[0]["error"]


def test_chat_e2e_api_endpoint_with_auth_and_permission():
    user_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())
    headers = get_test_auth_headers(user_id=user_id, tenant_id=tenant_id)

    mock_orchestrator = AsyncMock()
    mock_response = MagicMock()
    mock_response.media_type = "text/event-stream"
    mock_orchestrator.handle_chat_stream.return_value = mock_response

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("backend.app.auth.rbac.get_user_permissions", AsyncMock(return_value={"chat"}))
        mp.setattr("backend.app.api.chat.ChatOrchestrator", lambda: mock_orchestrator)
        res = client.post(
            "/api/v1/chat",
            json={"query": "Test query", "filters": {"source_type": "pdf"}, "retrieval_mode": "hybrid"},
            headers=headers
        )
        assert res.status_code == 200
        mock_orchestrator.handle_chat_stream.assert_called_once_with(
            query="Test query",
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=None,
            top_k=10,
            filters={"source_type": "pdf"},
            retrieval_mode="hybrid",
            enable_reranking=True,
            rerank_top_k=5,
            enable_rewriting=True,
            enable_compression=True,
            strict_grounding=None
        )


def test_chat_invalid_metadata_filters_returns_400_synchronously():
    """Verify malformed metadata filter payload in /api/v1/chat raises HTTP 400 JSON before streaming starts."""
    headers = get_test_auth_headers()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("backend.app.auth.rbac.get_user_permissions", AsyncMock(return_value={"chat"}))
        response = client.post(
            "/api/v1/chat",
            json={"query": "What is security policy?", "filters": {"document_id": "invalid-uuid-string"}},
            headers=headers
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"]["code"] == "BAD_REQUEST"
        assert "Invalid document_id UUID format" in data["error"]["message"]


@pytest.mark.asyncio
async def test_chat_orchestrator_resolves_user_roles_and_forwards_to_hybrid_search():
    """Verify ChatOrchestrator resolves role_ids via DocumentACLFilter and passes them to HybridSearchEngine."""
    user_id = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())
    role_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    mock_embed_provider = MagicMock()
    mock_embed_provider.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_acl_filter = AsyncMock()
    mock_acl_filter.get_user_role_ids.return_value = [role_id]

    mock_hybrid_search = AsyncMock()
    mock_hybrid_search.search.return_value = []

    mock_llm_client = MagicMock()

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed_provider,
        hybrid_search=mock_hybrid_search,
        acl_filter=mock_acl_filter,
        llm_client=mock_llm_client
    )

    filters = {"document_id": doc_id}
    await orchestrator.handle_chat_stream(
        query="What is the policy?",
        tenant_id=tenant_id,
        user_id=user_id,
        top_k=5,
        filters=filters,
        retrieval_mode="hybrid"
    )

    # 1. Role resolution verified
    mock_acl_filter.get_user_role_ids.assert_awaited_once_with(
        user_id=user_id,
        tenant_id=tenant_id,
        conn=None
    )

    # 2. Forwarding to hybrid search engine verified with candidate pool limit 30
    mock_hybrid_search.search.assert_called_once_with(
        query="What is the policy?",
        query_vector=[0.1] * 1024,
        tenant_id=tenant_id,
        limit=30,
        metadata_filters=filters,
        user_id=user_id,
        user_roles=[role_id],
        conn=None
    )


@pytest.mark.asyncio
async def test_chat_orchestrator_retrieval_failure_emits_only_error_event():
    """Verify RetrievalFailureError emits code=RETRIEVAL_FAILURE and NO tokens/citations/LLM calls follow."""
    from backend.app.retrieval.hybrid_search import RetrievalFailureError

    tenant_id = str(uuid.uuid4())

    mock_embed_provider = MagicMock()
    mock_embed_provider.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_hybrid_search = AsyncMock()
    mock_hybrid_search.search.side_effect = RetrievalFailureError("FTS search branch execution failed: DB timeout")

    mock_llm_client = MagicMock()

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed_provider,
        hybrid_search=mock_hybrid_search,
        llm_client=mock_llm_client
    )

    response = await orchestrator.handle_chat_stream(
        query="Query triggering retrieval branch error",
        tenant_id=tenant_id
    )

    # LLM must NEVER be invoked after retrieval failure
    mock_llm_client.stream_response.assert_not_called()

    sse_events = []
    async for line in response.body_iterator:
        if line:
            sse_events.append(line.decode("utf-8") if isinstance(line, bytes) else line)

    parsed = [json.loads(e[6:-2]) for e in sse_events if e.startswith("data: ")]

    # Assert EXACTLY one error event was emitted
    assert len(parsed) == 1
    assert parsed[0]["type"] == "error"
    assert parsed[0]["code"] == "RETRIEVAL_FAILURE"
    assert "Retrieval failed" in parsed[0]["error"]

    # Assert NO tokens, citations, or done events exist in stream
    tokens = [p for p in parsed if p["type"] == "token"]
    citations = [p for p in parsed if p["type"] == "citation"]
    assert len(tokens) == 0
    assert len(citations) == 0


@pytest.mark.asyncio
async def test_chat_orchestrator_retrieval_mode_dispatch_vector_and_fts():
    """Verify retrieval_mode='vector' dispatches to vector_search and 'fts' dispatches to fts_search."""
    tenant_id = str(uuid.uuid4())

    mock_embed_provider = MagicMock()
    mock_embed_provider.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_vector_search = AsyncMock()
    mock_vector_search.search.return_value = []

    mock_fts_search = AsyncMock()
    mock_fts_search.search.return_value = []

    mock_hybrid_search = AsyncMock()

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed_provider,
        vector_search=mock_vector_search,
        fts_search=mock_fts_search,
        hybrid_search=mock_hybrid_search
    )

    # 1. Test vector mode
    await orchestrator.handle_chat_stream(
        query="vector test",
        tenant_id=tenant_id,
        retrieval_mode="vector"
    )
    mock_vector_search.search.assert_called_once()
    mock_fts_search.search.assert_not_called()
    mock_hybrid_search.search.assert_not_called()

    # 2. Test FTS mode
    mock_vector_search.search.reset_mock()
    await orchestrator.handle_chat_stream(
        query="fts test",
        tenant_id=tenant_id,
        retrieval_mode="fts"
    )
    mock_fts_search.search.assert_called_once()
    mock_vector_search.search.assert_not_called()
    mock_hybrid_search.search.assert_not_called()


# =====================================================================
# Phase 6.2 Cross-Encoder Reranking Integration Tests
# =====================================================================

@pytest.mark.asyncio
async def test_chat_orchestrator_reranking_enabled_success():
    """Verify full end-to-end flow with reranking enabled passes 30 candidates, reranks, and streams top_k."""
    tenant_id = str(uuid.uuid4())
    chunk1_id = str(uuid.uuid4())
    chunk2_id = str(uuid.uuid4())

    mock_embed = MagicMock()
    mock_embed.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_hybrid = AsyncMock()
    mock_hybrid.search.return_value = [
        {"chunk_id": chunk1_id, "document_id": str(uuid.uuid4()), "document_version": 1, "text": "First chunk text", "page_number": 1, "section_path": "S1", "metadata": {}, "rrf_score": 0.05},
        {"chunk_id": chunk2_id, "document_id": str(uuid.uuid4()), "document_version": 1, "text": "Second chunk text", "page_number": 2, "section_path": "S2", "metadata": {}, "rrf_score": 0.04}
    ]

    mock_provider = MagicMock()
    mock_provider.score = AsyncMock(return_value=[-0.5, 0.9])  # Second chunk gets higher rerank_score!

    from backend.app.retrieval.reranker import RerankerEngine
    reranker_engine = RerankerEngine(provider=mock_provider)

    mock_llm = MagicMock()
    mock_llm.stream_response.return_value = iter(["Answer ", "text."])

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed,
        hybrid_search=mock_hybrid,
        reranker_engine=reranker_engine,
        llm_client=mock_llm
    )

    response = await orchestrator.handle_chat_stream(
        query="Tell me about second chunk",
        tenant_id=tenant_id,
        enable_reranking=True,
        rerank_top_k=2
    )

    # Hybrid search called with candidate pool limit 30
    mock_hybrid.search.assert_called_once()
    assert mock_hybrid.search.call_args[1]["limit"] == 30

    # Provider scored both candidates
    mock_provider.score.assert_awaited_once_with(
        "Tell me about second chunk",
        ["First chunk text", "Second chunk text"]
    )

    # LLM prompt contains chunk2 FIRST due to higher rerank_score (0.9 vs -0.5)
    llm_prompt = mock_llm.stream_response.call_args[1]["prompt"]
    assert llm_prompt.index("Second chunk text") < llm_prompt.index("First chunk text")


@pytest.mark.asyncio
async def test_chat_orchestrator_reranking_disabled_bypass():
    """Verify enable_reranking=False uses top_k candidate limit and NEVER calls reranker provider."""
    tenant_id = str(uuid.uuid4())
    chunk1_id = str(uuid.uuid4())

    mock_embed = MagicMock()
    mock_embed.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_hybrid = AsyncMock()
    mock_hybrid.search.return_value = [
        {"chunk_id": chunk1_id, "document_id": str(uuid.uuid4()), "document_version": 1, "text": "Chunk text", "page_number": 1, "section_path": "S1", "metadata": {}, "rrf_score": 0.05}
    ]

    mock_provider = MagicMock()
    mock_provider.score = AsyncMock()

    from backend.app.retrieval.reranker import RerankerEngine
    reranker_engine = RerankerEngine(provider=mock_provider)

    mock_llm = MagicMock()
    mock_llm.stream_response.return_value = iter(["Answer."])

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed,
        hybrid_search=mock_hybrid,
        reranker_engine=reranker_engine,
        llm_client=mock_llm
    )

    await orchestrator.handle_chat_stream(
        query="Bypass test",
        tenant_id=tenant_id,
        top_k=5,
        enable_reranking=False
    )

    # Hybrid search called with top_k (5), NOT 30
    assert mock_hybrid.search.call_args[1]["limit"] == 5

    # Reranker provider score was ZERO calls
    mock_provider.score.assert_not_called()


@pytest.mark.asyncio
async def test_chat_orchestrator_acl_precedence_before_reranking():
    """Verify ACL filtering occurs in search engine BEFORE reranker receives candidates."""
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    role_id = str(uuid.uuid4())

    mock_embed = MagicMock()
    mock_embed.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_acl = AsyncMock()
    mock_acl.get_user_role_ids.return_value = [role_id]

    # Hybrid search engine returns ONLY ACL-authorized chunk
    authorized_chunk_id = str(uuid.uuid4())
    mock_hybrid = AsyncMock()
    mock_hybrid.search.return_value = [
        {"chunk_id": authorized_chunk_id, "document_id": str(uuid.uuid4()), "document_version": 1, "text": "Authorized text", "page_number": 1, "section_path": "S1", "metadata": {}, "rrf_score": 0.05}
    ]

    mock_provider = MagicMock()
    mock_provider.score = AsyncMock(return_value=[0.8])

    from backend.app.retrieval.reranker import RerankerEngine
    reranker_engine = RerankerEngine(provider=mock_provider)

    mock_llm = MagicMock()
    mock_llm.stream_response.return_value = iter(["Answer."])

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed,
        hybrid_search=mock_hybrid,
        acl_filter=mock_acl,
        reranker_engine=reranker_engine,
        llm_client=mock_llm
    )

    await orchestrator.handle_chat_stream(
        query="ACL test",
        tenant_id=tenant_id,
        user_id=user_id,
        enable_reranking=True
    )

    # 1. ACL role resolution happened first
    mock_acl.get_user_role_ids.assert_awaited_once_with(user_id=user_id, tenant_id=tenant_id, conn=None)

    # 2. Hybrid search received user_roles
    assert mock_hybrid.search.call_args[1]["user_roles"] == [role_id]

    # 3. Reranker provider received ONLY the single authorized candidate
    mock_provider.score.assert_awaited_once_with("ACL test", ["Authorized text"])


@pytest.mark.asyncio
async def test_chat_orchestrator_unauthorized_passage_never_sent_to_provider():
    """Verify unauthorized document text is NEVER passed to reranker provider."""
    tenant_id = str(uuid.uuid4())
    authorized_id = str(uuid.uuid4())

    mock_embed = MagicMock()
    mock_embed.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_hybrid = AsyncMock()
    # Hybrid search already filtered out unauthorized text "TOP_SECRET_REDACTED"
    mock_hybrid.search.return_value = [
        {"chunk_id": authorized_id, "document_id": str(uuid.uuid4()), "document_version": 1, "text": "Public document content", "page_number": 1, "section_path": "S1", "metadata": {}, "rrf_score": 0.05}
    ]

    mock_provider = MagicMock()
    mock_provider.score = AsyncMock(return_value=[0.9])

    from backend.app.retrieval.reranker import RerankerEngine
    reranker_engine = RerankerEngine(provider=mock_provider)

    mock_llm = MagicMock()
    mock_llm.stream_response.return_value = iter(["Answer."])

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed,
        hybrid_search=mock_hybrid,
        reranker_engine=reranker_engine,
        llm_client=mock_llm
    )

    await orchestrator.handle_chat_stream(
        query="Secret test",
        tenant_id=tenant_id,
        enable_reranking=True
    )

    passages_sent = mock_provider.score.call_args[0][1]
    assert len(passages_sent) == 1
    assert "TOP_SECRET_REDACTED" not in passages_sent[0]
    assert passages_sent[0] == "Public document content"


@pytest.mark.asyncio
async def test_chat_orchestrator_reranker_failure_fallback_in_api_stream():
    """Verify reranker provider failure triggers safe RRF fallback without breaking stream or failing call."""
    tenant_id = str(uuid.uuid4())
    chunk1_id = str(uuid.uuid4())
    chunk2_id = str(uuid.uuid4())

    mock_embed = MagicMock()
    mock_embed.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_hybrid = AsyncMock()
    mock_hybrid.search.return_value = [
        {"chunk_id": chunk1_id, "document_id": str(uuid.uuid4()), "document_version": 1, "text": "RRF First text", "page_number": 1, "section_path": "S1", "metadata": {}, "rrf_score": 0.05},
        {"chunk_id": chunk2_id, "document_id": str(uuid.uuid4()), "document_version": 1, "text": "RRF Second text", "page_number": 2, "section_path": "S2", "metadata": {}, "rrf_score": 0.04}
    ]

    # Provider raises HTTP / connection error
    mock_provider = MagicMock()
    mock_provider.score = AsyncMock(side_effect=RuntimeError("Reranker service HTTP 503 Service Unavailable"))

    from backend.app.retrieval.reranker import RerankerEngine
    reranker_engine = RerankerEngine(provider=mock_provider)

    mock_llm = MagicMock()
    mock_llm.stream_response.return_value = iter(["Fallback ", "answer."])

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed,
        hybrid_search=mock_hybrid,
        reranker_engine=reranker_engine,
        llm_client=mock_llm
    )

    response = await orchestrator.handle_chat_stream(
        query="Fallback test",
        tenant_id=tenant_id,
        enable_reranking=True,
        rerank_top_k=2
    )

    # Verify LLM was invoked with RRF order preserved (RRF First text before RRF Second text)
    llm_prompt = mock_llm.stream_response.call_args[1]["prompt"]
    assert llm_prompt.index("RRF First text") < llm_prompt.index("RRF Second text")

    # Verify SSE events stream cleanly
    sse_events = []
    async for line in response.body_iterator:
        if line:
            sse_events.append(line.decode("utf-8") if isinstance(line, bytes) else line)

    parsed = [json.loads(e[6:-2]) for e in sse_events if e.startswith("data: ")]
    tokens = [p["content"] for p in parsed if p["type"] == "token"]
    assert "".join(tokens) == "Fallback answer."


def test_chat_schema_validation_rerank_top_k_bounds():
    """Verify HTTP endpoint rejects out-of-bounds rerank_top_k (ge=1, le=30) with HTTP 422 validation error."""
    headers = get_test_auth_headers()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("backend.app.auth.rbac.get_user_permissions", AsyncMock(return_value={"chat"}))

        # Test rerank_top_k = 0 (< 1)
        res_low = client.post(
            "/api/v1/chat",
            json={"query": "Test query", "rerank_top_k": 0},
            headers=headers
        )
        assert res_low.status_code == 422

        # Test rerank_top_k = 31 (> 30)
        res_high = client.post(
            "/api/v1/chat",
            json={"query": "Test query", "rerank_top_k": 31},
            headers=headers
        )
        assert res_high.status_code == 422


@pytest.mark.asyncio
async def test_chat_orchestrator_provenance_and_citation_preservation():
    """Verify provenance metadata (document_id, chunk_id, page_number, section_path) are preserved in citations."""
    tenant_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())

    mock_embed = MagicMock()
    mock_embed.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_hybrid = AsyncMock()
    mock_hybrid.search.return_value = [
        {
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "document_version": 2,
            "text": "Provenance chunk text.",
            "page_number": 42,
            "section_path": "Appendix A > Compliance",
            "metadata": {"confidential": False},
            "rrf_score": 0.033
        }
    ]

    mock_provider = MagicMock()
    mock_provider.score = AsyncMock(return_value=[1.25])

    from backend.app.retrieval.reranker import RerankerEngine
    reranker_engine = RerankerEngine(provider=mock_provider)

    mock_llm = MagicMock()
    mock_llm.stream_response.return_value = iter(["Response."])

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed,
        hybrid_search=mock_hybrid,
        reranker_engine=reranker_engine,
        llm_client=mock_llm
    )

    response = await orchestrator.handle_chat_stream(
        query="Provenance query",
        tenant_id=tenant_id,
        enable_reranking=True
    )

    sse_events = []
    async for line in response.body_iterator:
        if line:
            sse_events.append(line.decode("utf-8") if isinstance(line, bytes) else line)

    parsed = [json.loads(e[6:-2]) for e in sse_events if e.startswith("data: ")]
    citations = [p["citation"] for p in parsed if p["type"] == "citation"]

    assert len(citations) == 1
    cit = citations[0]
    assert cit["document_id"] == doc_id
    assert cit["chunk_id"] == chunk_id
    assert cit["page_number"] == 42
    assert cit["section_path"] == "Appendix A > Compliance"
    assert cit["rerank_score"] == 1.25


@pytest.mark.asyncio
async def test_chat_orchestrator_positional_query_and_passage_order():
    """Verify provider score receives query and passages in exact positional sequence."""
    tenant_id = str(uuid.uuid4())
    c1 = str(uuid.uuid4())
    c2 = str(uuid.uuid4())
    c3 = str(uuid.uuid4())

    mock_embed = MagicMock()
    mock_embed.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_hybrid = AsyncMock()
    mock_hybrid.search.return_value = [
        {"chunk_id": c1, "document_id": str(uuid.uuid4()), "document_version": 1, "text": "Positional passage 1", "page_number": 1, "section_path": "S1", "metadata": {}},
        {"chunk_id": c2, "document_id": str(uuid.uuid4()), "document_version": 1, "text": "Positional passage 2", "page_number": 2, "section_path": "S2", "metadata": {}},
        {"chunk_id": c3, "document_id": str(uuid.uuid4()), "document_version": 1, "text": "Positional passage 3", "page_number": 3, "section_path": "S3", "metadata": {}}
    ]

    mock_provider = MagicMock()
    mock_provider.score = AsyncMock(return_value=[0.1, 0.2, 0.3])

    from backend.app.retrieval.reranker import RerankerEngine
    reranker_engine = RerankerEngine(provider=mock_provider)

    mock_llm = MagicMock()
    mock_llm.stream_response.return_value = iter(["Response."])

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed,
        hybrid_search=mock_hybrid,
        reranker_engine=reranker_engine,
        llm_client=mock_llm
    )

    query_str = "Exact positional question?"
    await orchestrator.handle_chat_stream(
        query=query_str,
        tenant_id=tenant_id,
        enable_reranking=True
    )

    mock_provider.score.assert_awaited_once_with(
        query_str,
        ["Positional passage 1", "Positional passage 2", "Positional passage 3"]
    )

