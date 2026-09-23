"""
Integration and Unit tests for Phase 7 — Query Rewriting, Contextual Compression & Conversation Memory.
"""

import os
import json
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

os.environ["JWT_SECRET"] = "change-this-in-production-super-secret-key"

from backend.app.main import app
from backend.app.auth.jwt import create_access_token
from backend.app.api.chat import ChatOrchestrator

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_test_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("JWT_SECRET", "change-this-in-production-super-secret-key")


def get_test_auth_headers(user_id: str = None, tenant_id: str = None) -> dict:
    u_id = user_id or str(uuid.uuid4())
    t_id = tenant_id or str(uuid.uuid4())
    token = create_access_token({"sub": u_id, "tenant_id": t_id, "email": "test7@example.com"})
    return {"Authorization": f"Bearer {token}"}


# =====================================================================
# 1. Conversation CRUD Authorization & Isolation Tests
# =====================================================================

def test_conversations_crud_endpoints_authorization():
    """Verify POST /conversations, GET /conversations, GET /conversations/{id}, and DELETE /conversations/{id} endpoints."""
    headers = get_test_auth_headers()
    conv_id = str(uuid.uuid4())

    mock_store = AsyncMock()
    mock_store.create_conversation.return_value = {"id": conv_id, "title": "Test Conv", "summary": None}
    mock_store.list_conversations.return_value = [{"id": conv_id, "title": "Test Conv"}]
    mock_store.get_conversation.return_value = {"id": conv_id, "title": "Test Conv"}
    mock_store.get_recent_messages.return_value = [{"role": "user", "content": "Hi"}]
    mock_store.delete_conversation.return_value = True

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("backend.app.auth.rbac.get_user_permissions", AsyncMock(return_value={"chat"}))
        mp.setattr("backend.app.api.chat.ConversationStore", lambda: mock_store)

        # 1. Create conversation
        res_create = client.post("/api/v1/conversations", json={"title": "Test Conv"}, headers=headers)
        assert res_create.status_code == 200
        assert res_create.json()["data"]["id"] == conv_id

        # 2. List conversations
        res_list = client.get("/api/v1/conversations", headers=headers)
        assert res_list.status_code == 200
        assert len(res_list.json()["data"]) == 1

        # 3. Get conversation details
        res_get = client.get(f"/api/v1/conversations/{conv_id}", headers=headers)
        assert res_get.status_code == 200
        assert res_get.json()["data"]["id"] == conv_id

        # 4. Delete conversation
        res_del = client.delete(f"/api/v1/conversations/{conv_id}", headers=headers)
        assert res_del.status_code == 200
        assert res_del.json()["data"]["message"] == "Conversation deleted successfully"


def test_conversation_unauthorized_user_or_tenant_returns_404():
    """Verify requesting another user's conversation ID returns HTTP 404 Not Found."""
    headers_user_b = get_test_auth_headers()
    conv_id_user_a = str(uuid.uuid4())

    mock_store = AsyncMock()
    mock_store.get_conversation.return_value = None  # DB query filters by tenant_id + user_id and returns None

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("backend.app.auth.rbac.get_user_permissions", AsyncMock(return_value={"chat"}))
        mp.setattr("backend.app.api.chat.ConversationStore", lambda: mock_store)

        res = client.get(f"/api/v1/conversations/{conv_id_user_a}", headers=headers_user_b)
        assert res.status_code == 404
        assert "not found or unauthorized" in res.json()["error"]["message"]


# =====================================================================
# 2. Query Rewriting & Multi-Branch Retrieval Tests
# =====================================================================

@pytest.mark.asyncio
async def test_chat_orchestrator_query_rewriting_enabled_coreference():
    """Verify multi-turn query rewriter resolves coreferences into rewritten queries."""
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())

    mock_store = AsyncMock()
    mock_store.get_conversation.return_value = {"id": conv_id, "summary": None}
    mock_store.get_recent_messages.return_value = [{"role": "user", "content": "Tell me about Acme Password Policy."}]

    mock_rewriter = AsyncMock()
    mock_rewriter.rewrite_query.return_value = ["What is the password length in Acme Password Policy?"]

    mock_embed = MagicMock()
    mock_embed.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_hybrid = AsyncMock()
    mock_hybrid.search.return_value = [
        {"chunk_id": str(uuid.uuid4()), "document_id": str(uuid.uuid4()), "document_version": 1, "text": "Acme Policy requires 12 chars.", "page_number": 1, "section_path": "S1", "metadata": {}}
    ]

    mock_acl = AsyncMock()
    mock_acl.get_user_role_ids.return_value = []

    mock_llm = MagicMock()
    mock_llm.stream_response.return_value = iter(["Answer ", "text."])

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed,
        hybrid_search=mock_hybrid,
        acl_filter=mock_acl,
        conversation_store=mock_store,
        query_rewriter=mock_rewriter,
        llm_client=mock_llm
    )

    response = await orchestrator.handle_chat_stream(
        query="What is its length?",
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conv_id,
        enable_rewriting=True
    )

    # 1. Query rewriter was called with context history
    mock_rewriter.rewrite_query.assert_awaited_once()

    # 2. Hybrid search was executed with rewritten query
    mock_hybrid.search.assert_called_once()
    assert mock_hybrid.search.call_args[1]["query"] == "What is the password length in Acme Password Policy?"


@pytest.mark.asyncio
async def test_chat_orchestrator_query_rewriting_disabled_bypass():
    """Verify enable_rewriting=False passes original user query with ZERO rewriter calls."""
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    mock_rewriter = AsyncMock()

    mock_embed = MagicMock()
    mock_embed.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_hybrid = AsyncMock()
    mock_hybrid.search.return_value = []

    mock_acl = AsyncMock()
    mock_acl.get_user_role_ids.return_value = []

    mock_llm = MagicMock()

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed,
        hybrid_search=mock_hybrid,
        acl_filter=mock_acl,
        query_rewriter=mock_rewriter,
        llm_client=mock_llm
    )

    await orchestrator.handle_chat_stream(
        query="Bypass rewriter query",
        tenant_id=tenant_id,
        user_id=user_id,
        enable_rewriting=False
    )

    mock_rewriter.rewrite_query.assert_not_called()
    assert mock_hybrid.search.call_args[1]["query"] == "Bypass rewriter query"


@pytest.mark.asyncio
async def test_sub_query_decomposition_rrf_merge_and_30_candidate_ceiling():
    """Verify 3 retrieval branches are RRF-merged and ceiling-bounded to max 30 candidates passed to Phase 6."""
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    mock_store = AsyncMock()
    mock_store.get_conversation.return_value = {"id": str(uuid.uuid4()), "summary": None}
    mock_store.get_recent_messages.return_value = [{"role": "user", "content": "Context"}]

    mock_rewriter = AsyncMock()
    # Rewriter returns 3 sub-queries
    mock_rewriter.rewrite_query.return_value = ["Subquery 1", "Subquery 2", "Subquery 3"]

    mock_embed = MagicMock()
    mock_embed.embed = AsyncMock(return_value=[0.1] * 1024)

    # Hybrid search returns 20 candidates per branch (total 60 candidates before merge/ceiling)
    def generate_candidates(prefix):
        return [
            {
                "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{prefix}_{i}")),
                "document_id": str(uuid.uuid4()),
                "document_version": 1,
                "text": f"Candidate text {prefix}_{i}",
                "page_number": 1,
                "section_path": "S",
                "metadata": {}
            }
            for i in range(20)
        ]

    mock_hybrid = AsyncMock()
    mock_hybrid.search.side_effect = [
        generate_candidates("b1"),
        generate_candidates("b2"),
        generate_candidates("b3")
    ]

    mock_acl = AsyncMock()
    mock_acl.get_user_role_ids.return_value = []

    mock_reranker = AsyncMock()
    mock_reranker.rerank.return_value = []

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed,
        hybrid_search=mock_hybrid,
        acl_filter=mock_acl,
        conversation_store=mock_store,
        query_rewriter=mock_rewriter,
        reranker_engine=mock_reranker
    )

    await orchestrator.handle_chat_stream(
        query="Decompose query",
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=str(uuid.uuid4()),
        enable_rewriting=True,
        enable_reranking=True
    )

    # 1. 3 hybrid search branch queries executed
    assert mock_hybrid.search.call_count == 3

    # 2. Reranker receives candidate pool strictly bounded to max 30 candidates ceiling
    candidates_passed_to_reranker = mock_reranker.rerank.call_args[1]["candidates"]
    assert len(candidates_passed_to_reranker) <= 30


@pytest.mark.asyncio
async def test_acl_enforcement_on_all_rewritten_query_branches_and_merged_pool():
    """Verify EVERY sub-query branch executes with ACL pushdown AND all candidates passed to Reranker are ACL-authorized."""
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    role_id = str(uuid.uuid4())
    auth_chunk_1 = str(uuid.uuid4())
    auth_chunk_2 = str(uuid.uuid4())

    mock_acl = AsyncMock()
    mock_acl.get_user_role_ids.return_value = [role_id]

    mock_rewriter = AsyncMock()
    mock_rewriter.rewrite_query.return_value = ["Rewritten Q1", "Rewritten Q2"]

    mock_embed = MagicMock()
    mock_embed.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_hybrid = AsyncMock()
    # Hybrid search returns strictly ACL-authorized candidates for both branches
    mock_hybrid.search.side_effect = [
        [{"chunk_id": auth_chunk_1, "document_id": str(uuid.uuid4()), "text": "Auth 1", "metadata": {}}],
        [{"chunk_id": auth_chunk_2, "document_id": str(uuid.uuid4()), "text": "Auth 2", "metadata": {}}]
    ]

    mock_reranker = AsyncMock()
    mock_reranker.rerank.return_value = []

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed,
        hybrid_search=mock_hybrid,
        acl_filter=mock_acl,
        query_rewriter=mock_rewriter,
        reranker_engine=mock_reranker
    )

    mock_store = AsyncMock()
    mock_store.get_conversation.return_value = {"id": str(uuid.uuid4()), "summary": None}
    mock_store.get_recent_messages.return_value = [{"role": "user", "content": "Context"}]
    orchestrator.conversation_store = mock_store

    await orchestrator.handle_chat_stream(
        query="Test query",
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=str(uuid.uuid4()),
        enable_rewriting=True
    )

    # 1. Assert ACL user_roles forwarded to every branch search call
    assert mock_hybrid.search.call_count == 2
    assert mock_hybrid.search.call_args_list[0][1]["user_roles"] == [role_id]
    assert mock_hybrid.search.call_args_list[1][1]["user_roles"] == [role_id]

    # 2. Assert EVERY candidate passed to Reranker is in authorized set
    candidates = mock_reranker.rerank.call_args[1]["candidates"]
    for c in candidates:
        assert str(c["chunk_id"]) in [auth_chunk_1, auth_chunk_2]


# =====================================================================
# 3. Contextual Compression & Persistence Semantics Tests
# =====================================================================

@pytest.mark.asyncio
async def test_contextual_compression_disabled_bypass():
    """Verify enable_compression=False skips ContextualCompressor completely."""
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    mock_compressor = AsyncMock()

    mock_embed = MagicMock()
    mock_embed.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_hybrid = AsyncMock()
    mock_hybrid.search.return_value = [
        {"chunk_id": str(uuid.uuid4()), "document_id": str(uuid.uuid4()), "text": "Uncompressed text", "metadata": {}}
    ]

    mock_acl = AsyncMock()
    mock_acl.get_user_role_ids.return_value = []

    mock_llm = MagicMock()
    mock_llm.stream_response.return_value = iter(["Answer."])

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed,
        hybrid_search=mock_hybrid,
        acl_filter=mock_acl,
        contextual_compressor=mock_compressor,
        llm_client=mock_llm
    )

    await orchestrator.handle_chat_stream(
        query="Compression bypass test",
        tenant_id=tenant_id,
        user_id=user_id,
        enable_compression=False
    )

    mock_compressor.compress_candidates.assert_not_called()


@pytest.mark.asyncio
async def test_pre_and_post_persistence_streaming_semantics():
    """Verify user message is persisted BEFORE streaming, and assistant completion + citations are persisted AFTER stream completes."""
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    mock_store = AsyncMock()
    mock_store.get_conversation.return_value = {"id": conv_id, "summary": None}
    mock_store.get_recent_messages.return_value = []

    mock_embed = MagicMock()
    mock_embed.embed = AsyncMock(return_value=[0.1] * 1024)

    mock_hybrid = AsyncMock()
    mock_hybrid.search.return_value = [
        {"chunk_id": chunk_id, "document_id": doc_id, "document_version": 1, "text": "Context text.", "page_number": 1, "section_path": "S1", "metadata": {}}
    ]

    mock_acl = AsyncMock()
    mock_acl.get_user_role_ids.return_value = []

    mock_llm = MagicMock()
    mock_llm.stream_response.return_value = iter(["Streamed ", "assistant ", "answer."])

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embed,
        hybrid_search=mock_hybrid,
        acl_filter=mock_acl,
        conversation_store=mock_store,
        llm_client=mock_llm
    )

    response = await orchestrator.handle_chat_stream(
        query="User pre-persist question",
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conv_id
    )

    # 1. User message pre-persisted BEFORE stream consumption completes
    mock_store.add_user_message.assert_awaited_once_with(
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conv_id,
        content="User pre-persist question",
        conn=None
    )

    # Consume SSE events to complete stream
    sse_events = []
    async for line in response.body_iterator:
        if line:
            sse_events.append(line.decode("utf-8") if isinstance(line, bytes) else line)

    # 2. Assistant message + citations post-persisted AFTER stream finishes
    mock_store.add_assistant_message.assert_awaited_once()
    kwargs = mock_store.add_assistant_message.call_args[1]
    assert kwargs["tenant_id"] == tenant_id
    assert kwargs["user_id"] == user_id
    assert kwargs["conversation_id"] == conv_id
    assert kwargs["content"] == "Streamed assistant answer."
    assert len(kwargs["citations"]) == 1
    assert kwargs["citations"][0]["chunk_id"] == chunk_id
