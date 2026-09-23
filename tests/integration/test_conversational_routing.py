"""
Integration Tests for Conversational Query Routing in ChatOrchestrator
Verifies that conversational queries bypass embedding, retrieval, RRF, and reranking,
while domain queries trigger the complete RAG pipeline.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from backend.app.api.chat import ChatOrchestrator
from backend.app.chat.query_classifier import QueryType, QueryClassifier


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
    search.search.return_value = [
        {
            "chunk_id": "chunk-101",
            "document_id": "doc-101",
            "document_version": 1,
            "page_number": 1,
            "section_path": "Sec 1",
            "similarity_score": 0.88,
            "text": "Employee leave policy states 20 days paid leave per year."
        }
    ]
    return search


@pytest.fixture
def mock_reranker():
    reranker = AsyncMock()
    reranker.rerank.side_effect = lambda query, candidates, top_k: candidates[:top_k]
    return reranker


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.stream_response.return_value = iter(["Hello! ", "How can I help you today?"])
    client.generate_response.return_value = "YES"
    return client


@pytest.fixture
def orchestrator(
    mock_embedding_provider,
    mock_vector_search,
    mock_fts_search,
    mock_hybrid_search,
    mock_reranker,
    mock_llm_client
):
    query_classifier = QueryClassifier(llm_client=mock_llm_client)
    return ChatOrchestrator(
        embedding_provider=mock_embedding_provider,
        vector_search=mock_vector_search,
        fts_search=mock_fts_search,
        hybrid_search=mock_hybrid_search,
        reranker_engine=mock_reranker,
        llm_client=mock_llm_client,
        query_classifier=query_classifier
    )


@pytest.mark.asyncio
async def test_conversational_query_skips_retrieval(
    orchestrator,
    mock_embedding_provider,
    mock_hybrid_search,
    mock_reranker
):
    response = await orchestrator.handle_chat_stream(
        query="Hello",
        tenant_id="00000000-0000-0000-0000-000000000001"
    )

    events = []
    async for chunk in response.body_iterator:
        events.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))

    full_sse = "".join(events)

    assert not mock_embedding_provider.embed.called, "Embedding generation should not be called for CONVERSATIONAL query"
    assert not mock_hybrid_search.search.called, "Hybrid search should not be called for CONVERSATIONAL query"
    assert not mock_reranker.rerank.called, "Reranker should not be called for CONVERSATIONAL query"

    assert "data: " in full_sse
    assert '"citations_count": 0' in full_sse or '"citations_count":0' in full_sse


@pytest.mark.asyncio
async def test_hi_i_am_sarah_complete_routing_boundary(
    orchestrator,
    mock_embedding_provider,
    mock_vector_search,
    mock_fts_search,
    mock_hybrid_search,
    mock_reranker
):
    """
    Integration Test: Submits 'hi i am sarah' and verifies complete conversational routing boundary.
    Pipeline flow: QueryClassifier -> CONVERSATIONAL -> Direct LLM -> SSE tokens -> done (citations_count = 0).
    Explicitly asserts that retrieval & scoring components are NEVER called.
    """
    mock_evidence_scorer = MagicMock()
    orchestrator.evidence_scorer = mock_evidence_scorer

    classification = await orchestrator.query_classifier.classify_query("hi i am sarah")
    assert classification == QueryType.CONVERSATIONAL, f"Expected CONVERSATIONAL, got {classification}"

    response = await orchestrator.handle_chat_stream(
        query="hi i am sarah",
        tenant_id="00000000-0000-0000-0000-000000000001"
    )

    events = []
    async for chunk in response.body_iterator:
        events.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))

    full_sse = "".join(events)

    assert not mock_embedding_provider.embed.called, "embedding_provider.embed MUST NOT be called"
    assert not mock_vector_search.search.called, "vector_search.search MUST NOT be called"
    assert not mock_fts_search.search.called, "fts_search.search MUST NOT be called"
    assert not mock_hybrid_search.search.called, "hybrid_search.search MUST NOT be called"
    assert not mock_reranker.rerank.called, "reranker.rerank MUST NOT be called"
    assert not mock_evidence_scorer.score.called, "EvidenceScorer MUST NOT be called"

    assert "data: " in full_sse
    assert "Hello!" in full_sse or "How can I help" in full_sse
    assert '"citations_count": 0' in full_sse or '"citations_count":0' in full_sse


@pytest.mark.asyncio
async def test_mixed_and_domain_queries_trigger_full_rag_pipeline(
    orchestrator,
    mock_embedding_provider,
    mock_hybrid_search,
    mock_reranker
):
    """
    Verifies that domain and mixed queries (greeting + domain intent) execute full RAG retrieval pipeline.
    """
    domain_queries = [
        "Hi, what is the leave policy?",
        "Hello, how do I configure authentication?",
        "Thanks, but what is the leave policy for contractors?"
    ]

    for q in domain_queries:
        mock_embedding_provider.embed.reset_mock()
        mock_hybrid_search.search.reset_mock()
        mock_reranker.rerank.reset_mock()

        response = await orchestrator.handle_chat_stream(
            query=q,
            tenant_id="00000000-0000-0000-0000-000000000001"
        )

        events = []
        async for chunk in response.body_iterator:
            events.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))

        assert mock_embedding_provider.embed.called, f"Embedding MUST be called for domain query: '{q}'"
        assert mock_hybrid_search.search.called, f"Hybrid search MUST be called for domain query: '{q}'"
        assert mock_reranker.rerank.called, f"Reranker MUST be called for domain query: '{q}'"


@pytest.mark.asyncio
async def test_memory_recall_queries_skip_retrieval_and_use_history(
    orchestrator,
    mock_embedding_provider,
    mock_vector_search,
    mock_fts_search,
    mock_hybrid_search,
    mock_reranker
):
    """
    Verifies memory recall queries ('what is my name', 'hey what's my name again?', 'who am i')
    bypass all retrieval and scoring components and output 0 citations.
    """
    recall_queries = [
        "what is my name",
        "hey what's my name again?",
        "who am i"
    ]

    for q in recall_queries:
        mock_embedding_provider.embed.reset_mock()
        mock_vector_search.search.reset_mock()
        mock_fts_search.search.reset_mock()
        mock_hybrid_search.search.reset_mock()
        mock_reranker.rerank.reset_mock()

        response = await orchestrator.handle_chat_stream(
            query=q,
            tenant_id="00000000-0000-0000-0000-000000000001"
        )

        events = []
        async for chunk in response.body_iterator:
            events.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk))

        full_sse = "".join(events)

        assert not mock_embedding_provider.embed.called, f"Embedding MUST NOT be called for recall query: '{q}'"
        assert not mock_vector_search.search.called, f"Vector search MUST NOT be called for recall query: '{q}'"
        assert not mock_fts_search.search.called, f"FTS search MUST NOT be called for recall query: '{q}'"
        assert not mock_hybrid_search.search.called, f"Hybrid search MUST NOT be called for recall query: '{q}'"
        assert not mock_reranker.rerank.called, f"Reranker MUST NOT be called for recall query: '{q}'"

        assert "data: " in full_sse
        assert '"citations_count": 0' in full_sse or '"citations_count":0' in full_sse


