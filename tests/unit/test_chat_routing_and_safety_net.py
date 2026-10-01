"""
Unit Tests for ChatOrchestrator History Handling and Secondary Safety Net
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock
from backend.app.api.chat import ChatOrchestrator
from backend.app.chat.query_classifier import QueryClassifier, QueryType
from backend.app.memory.conversation_store import ConversationStore


@pytest.fixture
def mock_embedding_provider():
    provider = AsyncMock()
    provider.embed.return_value = [0.1] * 1024
    return provider


@pytest.fixture
def mock_vector_search():
    search = AsyncMock()
    search.search.return_value = []
    return search


@pytest.fixture
def mock_fts_search():
    search = AsyncMock()
    search.search.return_value = []
    return search


@pytest.fixture
def mock_hybrid_search():
    search = AsyncMock()
    search.search.return_value = []
    return search


@pytest.fixture
def mock_reranker():
    reranker = AsyncMock()
    reranker.rerank.side_effect = lambda query, candidates, top_k: candidates[:top_k]
    return reranker


@pytest.fixture
def mock_acl_filter():
    filter_mock = MagicMock()
    filter_mock.get_user_role_ids = AsyncMock(return_value=[])
    return filter_mock


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.stream_response.return_value = iter(["You like ", "baseball!"])
    client.generate_response.return_value = "YES"
    return client


@pytest.mark.asyncio
async def test_classifier_receives_history_without_current_message(
    mock_embedding_provider,
    mock_vector_search,
    mock_fts_search,
    mock_hybrid_search,
    mock_reranker,
    mock_acl_filter,
    mock_llm_client
):
    """
    Verifies that the query classifier receives prior_history strictly excluding the current user message.
    """
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    m1_id = str(uuid.uuid4())
    curr_msg_id = str(uuid.uuid4())

    mock_conv_store = MagicMock()
    mock_conv_store.get_conversation = AsyncMock(return_value={"id": conv_id, "summary": None})
    mock_conv_store.add_user_message = AsyncMock(return_value={"id": curr_msg_id, "role": "user", "content": "which sport i like"})
    mock_conv_store.get_recent_messages = AsyncMock(return_value=[
        {"id": m1_id, "conversation_id": conv_id, "role": "user", "content": "I like baseball"},
        {"id": curr_msg_id, "conversation_id": conv_id, "role": "user", "content": "which sport i like"}
    ])
    mock_conv_store.add_assistant_message = AsyncMock()

    mock_classifier = MagicMock()
    mock_classifier.classify_query = AsyncMock(return_value=QueryType.CONVERSATIONAL)

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embedding_provider,
        vector_search=mock_vector_search,
        fts_search=mock_fts_search,
        hybrid_search=mock_hybrid_search,
        reranker_engine=mock_reranker,
        acl_filter=mock_acl_filter,
        llm_client=mock_llm_client,
        query_classifier=mock_classifier,
        conversation_store=mock_conv_store
    )

    response = await orchestrator.handle_chat_stream(
        query="which sport i like",
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conv_id
    )

    # Verify classifier received history without the current message
    mock_classifier.classify_query.assert_called_once()
    passed_history = mock_classifier.classify_query.call_args[1]["history_messages"]
    assert len(passed_history) == 1
    assert passed_history[0]["id"] == m1_id
    assert passed_history[0]["content"] == "I like baseball"


@pytest.mark.asyncio
async def test_domain_insufficient_evidence_safety_net_answers_from_history(
    mock_embedding_provider,
    mock_vector_search,
    mock_fts_search,
    mock_hybrid_search,
    mock_reranker,
    mock_acl_filter,
    mock_llm_client
):
    """
    Verifies: DOMAIN route + insufficient evidence + 'I like baseball' in history + 'which sport i like'
    -> Safety net triggers and answers from history using conversation memory.
    """
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    m1_id = str(uuid.uuid4())
    curr_msg_id = str(uuid.uuid4())

    mock_conv_store = MagicMock()
    mock_conv_store.get_conversation = AsyncMock(return_value={"id": conv_id, "summary": None})
    mock_conv_store.add_user_message = AsyncMock(return_value={"id": curr_msg_id, "role": "user", "content": "which sport i like"})
    mock_conv_store.get_recent_messages = AsyncMock(return_value=[
        {"id": m1_id, "conversation_id": conv_id, "role": "user", "content": "I like baseball, it is my favorite sport"},
        {"id": curr_msg_id, "conversation_id": conv_id, "role": "user", "content": "which sport i like"}
    ])
    mock_conv_store.add_assistant_message = AsyncMock()

    # Force classifier to return DOMAIN
    mock_classifier = MagicMock()
    mock_classifier.classify_query = AsyncMock(return_value=QueryType.DOMAIN)

    # Hybrid search returns empty (no relevant document context)
    mock_hybrid_search.search.return_value = []

    mock_llm_client.stream_response.return_value = iter(["You said ", "you like baseball."])

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embedding_provider,
        vector_search=mock_vector_search,
        fts_search=mock_fts_search,
        hybrid_search=mock_hybrid_search,
        reranker_engine=mock_reranker,
        acl_filter=mock_acl_filter,
        llm_client=mock_llm_client,
        query_classifier=mock_classifier,
        conversation_store=mock_conv_store
    )

    response = await orchestrator.handle_chat_stream(
        query="which sport i like",
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conv_id
    )

    events = []
    async for chunk in response.body_iterator:
        events.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))
    full_body = "".join(events)

    # Safety net should stream conversational tokens and not produce a refusal
    assert "You said you like baseball" in full_body or "You said" in full_body
    assert '"query_type": "conversational_safety_net"' in full_body


@pytest.mark.asyncio
async def test_domain_insufficient_evidence_unrelated_history_returns_refusal(
    mock_embedding_provider,
    mock_vector_search,
    mock_fts_search,
    mock_hybrid_search,
    mock_reranker,
    mock_acl_filter,
    mock_llm_client
):
    """
    Verifies: DOMAIN route + insufficient evidence + unrelated history + 'what does my handbook say about my benefits'
    -> Safety net does NOT trigger, returns document refusal.
    """
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    m1_id = str(uuid.uuid4())
    curr_msg_id = str(uuid.uuid4())

    mock_conv_store = MagicMock()
    mock_conv_store.get_conversation = AsyncMock(return_value={"id": conv_id, "summary": None})
    mock_conv_store.add_user_message = AsyncMock(return_value={"id": curr_msg_id, "role": "user", "content": "what does my handbook say about my benefits"})
    # History only has baseball, no token overlap with handbook or benefits
    mock_conv_store.get_recent_messages = AsyncMock(return_value=[
        {"id": m1_id, "conversation_id": conv_id, "role": "user", "content": "I like baseball"},
        {"id": curr_msg_id, "conversation_id": conv_id, "role": "user", "content": "what does my handbook say about my benefits"}
    ])
    mock_conv_store.add_assistant_message = AsyncMock()

    mock_classifier = MagicMock()
    mock_classifier.classify_query = AsyncMock(return_value=QueryType.DOMAIN)

    # Empty search results
    mock_hybrid_search.search.return_value = []

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embedding_provider,
        vector_search=mock_vector_search,
        fts_search=mock_fts_search,
        hybrid_search=mock_hybrid_search,
        reranker_engine=mock_reranker,
        acl_filter=mock_acl_filter,
        llm_client=mock_llm_client,
        query_classifier=mock_classifier,
        conversation_store=mock_conv_store
    )

    response = await orchestrator.handle_chat_stream(
        query="what does my handbook say about my benefits",
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conv_id
    )

    events = []
    async for chunk in response.body_iterator:
        events.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))
    full_body = "".join(events)

    # Must return document refusal
    assert "I could not find sufficient information" in full_body or '"refusal": true' in full_body or '"refusal":true' in full_body
    assert "conversational_safety_net" not in full_body


@pytest.mark.asyncio
async def test_domain_insufficient_evidence_with_shared_stopword_returns_refusal(
    mock_embedding_provider,
    mock_vector_search,
    mock_fts_search,
    mock_hybrid_search,
    mock_reranker,
    mock_acl_filter,
    mock_llm_client
):
    """
    Verifies: query 'what does my handbook say about the benefits I like' with history 'I like baseball'
    -> Stopwords like 'like', 'say', 'about', 'my', 'what', 'i' are filtered out.
    Content words (handbook, benefits) do NOT match baseball -> document refusal (safety net does NOT fire).
    """
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    m1_id = str(uuid.uuid4())
    curr_msg_id = str(uuid.uuid4())

    mock_conv_store = MagicMock()
    mock_conv_store.get_conversation = AsyncMock(return_value={"id": conv_id, "summary": None})
    mock_conv_store.add_user_message = AsyncMock(return_value={"id": curr_msg_id, "role": "user", "content": "what does my handbook say about the benefits I like"})
    mock_conv_store.get_recent_messages = AsyncMock(return_value=[
        {"id": m1_id, "conversation_id": conv_id, "role": "user", "content": "I like baseball"},
        {"id": curr_msg_id, "conversation_id": conv_id, "role": "user", "content": "what does my handbook say about the benefits I like"}
    ])
    mock_conv_store.add_assistant_message = AsyncMock()

    mock_classifier = MagicMock()
    mock_classifier.classify_query = AsyncMock(return_value=QueryType.DOMAIN)

    # Empty search results
    mock_hybrid_search.search.return_value = []

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embedding_provider,
        vector_search=mock_vector_search,
        fts_search=mock_fts_search,
        hybrid_search=mock_hybrid_search,
        reranker_engine=mock_reranker,
        acl_filter=mock_acl_filter,
        llm_client=mock_llm_client,
        query_classifier=mock_classifier,
        conversation_store=mock_conv_store
    )

    response = await orchestrator.handle_chat_stream(
        query="what does my handbook say about the benefits I like",
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conv_id
    )

    events = []
    async for chunk in response.body_iterator:
        events.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))
    full_body = "".join(events)

    # Safety net must NOT fire, must return document refusal
    assert "I could not find sufficient information" in full_body or '"refusal": true' in full_body or '"refusal":true' in full_body
    assert "conversational_safety_net" not in full_body


@pytest.mark.asyncio
async def test_domain_sufficient_evidence_safety_net_never_runs(
    mock_embedding_provider,
    mock_vector_search,
    mock_fts_search,
    mock_hybrid_search,
    mock_reranker,
    mock_acl_filter,
    mock_llm_client
):
    """
    Verifies: DOMAIN route + sufficient document evidence -> safety net NEVER runs, document grounding prompt used.
    """
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    curr_msg_id = str(uuid.uuid4())

    mock_conv_store = MagicMock()
    mock_conv_store.get_conversation = AsyncMock(return_value={"id": conv_id, "summary": None})
    mock_conv_store.add_user_message = AsyncMock(return_value={"id": curr_msg_id, "role": "user", "content": "Explain the leave policy"})
    mock_conv_store.get_recent_messages = AsyncMock(return_value=[
        {"id": curr_msg_id, "conversation_id": conv_id, "role": "user", "content": "Explain the leave policy"}
    ])
    mock_conv_store.add_assistant_message = AsyncMock()

    mock_classifier = MagicMock()
    mock_classifier.classify_query = AsyncMock(return_value=QueryType.DOMAIN)

    # Hybrid search returns high relevance chunk
    mock_hybrid_search.search.return_value = [
        {
            "chunk_id": "c101",
            "document_id": "d101",
            "document_version": 1,
            "page_number": 1,
            "section_path": "Policy",
            "similarity_score": 0.95,
            "text": "The company offers 25 days paid vacation leave."
        }
    ]

    mock_llm_client.stream_response.return_value = iter(["According to policy, ", "25 days are offered."])

    orchestrator = ChatOrchestrator(
        embedding_provider=mock_embedding_provider,
        vector_search=mock_vector_search,
        fts_search=mock_fts_search,
        hybrid_search=mock_hybrid_search,
        reranker_engine=mock_reranker,
        acl_filter=mock_acl_filter,
        llm_client=mock_llm_client,
        query_classifier=mock_classifier,
        conversation_store=mock_conv_store
    )

    response = await orchestrator.handle_chat_stream(
        query="Explain the leave policy",
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conv_id
    )

    events = []
    async for chunk in response.body_iterator:
        events.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))
    full_body = "".join(events)

    assert "conversational_safety_net" not in full_body
    assert "According to policy" in full_body
    assert '"citations_count": 1' in full_body or '"citations_count":1' in full_body

