"""
Unit tests for Cross-Encoder Reranker Module (Phase 6 — Checkpoint 6.1).
"""

import uuid
import math
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from backend.app.retrieval.reranker import (
    RerankerEngine,
    BGERerankerProvider,
    MockRerankerProvider,
    get_reranker_provider
)
@pytest.mark.asyncio
async def test_reranker_valid_scoring_and_ranking():
    """1. Valid scoring & ranking — candidates are re-ordered by rerank_score DESC."""
    c1 = {"chunk_id": "00000000-0000-0000-0000-000000000001", "text": "Passage 1", "rrf_score": 0.03}
    c2 = {"chunk_id": "00000000-0000-0000-0000-000000000002", "text": "Passage 2", "rrf_score": 0.05}
    candidates = [c1, c2]

    # Mock provider giving higher score to c1 (0.95) than c2 (0.12)
    mock_provider = MockRerankerProvider(scores=[0.95, 0.12])
    engine = RerankerEngine(provider=mock_provider, score_threshold=0.0, relative_threshold_factor=0.0)

    results = await engine.rerank("test query", candidates, top_k=2)

    assert len(results) == 2
    # c1 has higher rerank_score, so it should rank first despite lower rrf_score
    assert results[0]["chunk_id"] == c1["chunk_id"]
    assert results[0]["rerank_score"] == 0.95
    assert results[0]["rrf_score"] == 0.03

    assert results[1]["chunk_id"] == c2["chunk_id"]
    assert results[1]["rerank_score"] == 0.12
    assert results[1]["rrf_score"] == 0.05


@pytest.mark.asyncio
async def test_reranker_deterministic_tie_breaking_chunk_id_asc():
    """2. Deterministic tie-breaking — candidates with equal rerank_score sort by chunk_id ASC."""
    c_z = {"chunk_id": "00000000-0000-0000-0000-000000000099", "text": "Passage Z"}
    c_a = {"chunk_id": "00000000-0000-0000-0000-000000000001", "text": "Passage A"}
    candidates = [c_z, c_a]

    # Both get exact same rerank_score (0.90)
    mock_provider = MockRerankerProvider(scores=[0.90, 0.90])
    engine = RerankerEngine(provider=mock_provider)

    results = await engine.rerank("test query", candidates, top_k=2)

    assert len(results) == 2
    assert results[0]["rerank_score"] == results[1]["rerank_score"]
    # Primary tie-break chunk_id ASC puts c_a before c_z
    assert results[0]["chunk_id"] == c_a["chunk_id"]
    assert results[1]["chunk_id"] == c_z["chunk_id"]


@pytest.mark.asyncio
async def test_reranker_empty_candidate_list():
    """3. Empty candidate list — returns [] without provider invocation."""
    mock_provider = AsyncMock()
    engine = RerankerEngine(provider=mock_provider)

    results = await engine.rerank("test query", [], top_k=5)

    assert results == []
    mock_provider.score.assert_not_called()


@pytest.mark.asyncio
async def test_reranker_pool_size_bounds_top_k():
    """4. Pool size bounds — top_k > candidate_count returns available candidates without error."""
    c1 = {"chunk_id": str(uuid.uuid4()), "text": "P1"}
    c2 = {"chunk_id": str(uuid.uuid4()), "text": "P2"}
    candidates = [c1, c2]

    mock_provider = MockRerankerProvider(scores=[0.9, 0.85])
    engine = RerankerEngine(provider=mock_provider)

    # Request top_k=10 on pool of 2
    results = await engine.rerank("test query", candidates, top_k=10)

    assert len(results) == 2


@pytest.mark.asyncio
async def test_reranker_single_top_k():
    """5. Single top_k — top_k=1 returns top 1 reranked candidate."""
    c1 = {"chunk_id": "00000000-0000-0000-0000-000000000001", "text": "P1"}
    c2 = {"chunk_id": "00000000-0000-0000-0000-000000000002", "text": "P2"}
    candidates = [c1, c2]

    mock_provider = MockRerankerProvider(scores=[0.2, 0.9])
    engine = RerankerEngine(provider=mock_provider)

    results = await engine.rerank("test query", candidates, top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == c2["chunk_id"]


@pytest.mark.asyncio
async def test_reranker_uuid_chunk_id_validation():
    """6. UUID chunk_id validation — missing or non-UUID chunk_id raises ValueError."""
    engine = RerankerEngine(provider=MockRerankerProvider())

    # Missing chunk_id
    with pytest.raises(ValueError, match="missing required 'chunk_id' field"):
        await engine.rerank("q", [{"text": "P1"}])

    # Non-UUID string chunk_id
    with pytest.raises(ValueError, match="invalid UUID string 'chunk_id'"):
        await engine.rerank("q", [{"chunk_id": "invalid-chunk-id-123", "text": "P1"}])


@pytest.mark.asyncio
async def test_reranker_graceful_fallback_preserves_exact_input_rrf_order():
    """7. Graceful fallback — provider exception returns candidates in exact input RRF score order (c2 then c1)."""
    c2 = {"chunk_id": "00000000-0000-0000-0000-000000000002", "text": "P2", "rrf_score": 0.08}
    c1 = {"chunk_id": "00000000-0000-0000-0000-000000000001", "text": "P1", "rrf_score": 0.02}
    candidates = [c2, c1]

    mock_provider = AsyncMock()
    mock_provider.score.side_effect = RuntimeError("Reranker HTTP connection timeout")

    engine = RerankerEngine(provider=mock_provider)

    results = await engine.rerank("test query", candidates, top_k=2)

    # Must NOT raise exception — falls back safely to input list order (c2, then c1)
    assert len(results) == 2
    assert results[0]["chunk_id"] == c2["chunk_id"]
    assert results[1]["chunk_id"] == c1["chunk_id"]
    assert "rerank_score" not in results[0]  # Fallback preserves original RRF candidates


@pytest.mark.asyncio
async def test_reranker_mismatched_score_count_triggers_fallback():
    """8. Mismatched score count — provider returning wrong number of scores triggers RRF fallback."""
    c1 = {"chunk_id": str(uuid.uuid4()), "text": "P1", "rrf_score": 0.05}
    c2 = {"chunk_id": str(uuid.uuid4()), "text": "P2", "rrf_score": 0.03}

    # Returns 1 score for 2 candidates
    mock_provider = MockRerankerProvider(scores=[0.9])
    engine = RerankerEngine(provider=mock_provider)

    results = await engine.rerank("test query", [c1, c2], top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == c1["chunk_id"]


@pytest.mark.asyncio
async def test_reranker_non_finite_score_triggers_fallback():
    """9. Non-finite score validation — NaN or Infinity score triggers safe RRF fallback."""
    c1 = {"chunk_id": str(uuid.uuid4()), "text": "P1", "rrf_score": 0.05}
    c2 = {"chunk_id": str(uuid.uuid4()), "text": "P2", "rrf_score": 0.03}

    mock_provider = MockRerankerProvider(scores=[0.5, float("nan")])
    engine = RerankerEngine(provider=mock_provider)

    results = await engine.rerank("test query", [c1, c2], top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == c1["chunk_id"]


@pytest.mark.asyncio
async def test_reranker_provenance_non_mutation_preservation():
    """10. Provenance non-mutation — original candidate dicts are not mutated in-place and all fields remain identical."""
    doc_id = str(uuid.uuid4())
    c1 = {
        "chunk_id": "00000000-0000-0000-0000-000000000001",
        "document_id": doc_id,
        "document_version": 2,
        "text": "Original text.",
        "page_number": 3,
        "section_path": "Sec > Sub",
        "metadata": {"key": "val"},
        "rrf_score": 0.032
    }
    c1_copy = dict(c1)

    mock_provider = MockRerankerProvider(scores=[0.88])
    engine = RerankerEngine(provider=mock_provider)

    results = await engine.rerank("query", [c1], top_k=1)

    # Original object must NOT be mutated in-place
    assert "rerank_score" not in c1
    assert c1 == c1_copy

    # Output object contains all original fields plus rerank_score
    res = results[0]
    assert res["chunk_id"] == c1["chunk_id"]
    assert res["document_id"] == doc_id
    assert res["document_version"] == 2
    assert res["text"] == "Original text."
    assert res["page_number"] == 3
    assert res["section_path"] == "Sec > Sub"
    assert res["metadata"] == {"key": "val"}
    assert res["rrf_score"] == 0.032
    assert res["rerank_score"] == 0.88


@pytest.mark.asyncio
async def test_reranker_single_batch_http_invocation():
    """11. Single batch HTTP invocation — BGERerankerProvider sends 1 batch POST /rerank request for candidate pool."""
    c1 = {"chunk_id": str(uuid.uuid4()), "text": "P1"}
    c2 = {"chunk_id": str(uuid.uuid4()), "text": "P2"}

    provider = BGERerankerProvider(base_url="http://mock-reranker:8001", model_name="BAAI/bge-reranker-v2-m3")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"scores": [0.75, 0.45]}

    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=mock_resp)) as mock_post:
        scores = await provider.score("query string", ["P1", "P2"])
        assert scores == [0.75, 0.45]

        # Verify EXACTLY 1 batch HTTP POST call was made
        mock_post.assert_awaited_once()
        call_url = mock_post.call_args[0][0]
        call_json = mock_post.call_args[1]["json"]

        assert call_url == "http://mock-reranker:8001/rerank"
        assert call_json["model"] == "BAAI/bge-reranker-v2-m3"
        assert call_json["query"] == "query string"
        assert call_json["passages"] == ["P1", "P2"]


@pytest.mark.asyncio
async def test_reranker_timeout_configuration_and_factory_default():
    """12. Config & Factory tests — verifies RERANKER_TIMEOUT_SECONDS resolution and get_reranker_provider defaults."""
    # Test RERANKER_TIMEOUT_SECONDS env var
    with patch.dict("os.environ", {"RERANKER_TIMEOUT_SECONDS": "5.5"}):
        p1 = BGERerankerProvider()
        assert p1.timeout_seconds == 5.5

    # Test explicit timeout_seconds parameter override
    p2 = BGERerankerProvider(timeout_seconds=10.0)
    assert p2.timeout_seconds == 10.0

    # Test factory default (RERANKER_PROVIDER not set -> defaults to BGERerankerProvider)
    with patch.dict("os.environ", {}, clear=True):
        p_factory = get_reranker_provider()
        assert isinstance(p_factory, BGERerankerProvider)

    # Test factory mock selection
    with patch.dict("os.environ", {"RERANKER_PROVIDER": "mock"}):
        p_mock = get_reranker_provider()
        assert isinstance(p_mock, MockRerankerProvider)


@pytest.mark.asyncio
async def test_bge_reranker_provider_readiness_and_http_errors():
    """13. Provider readiness and HTTP error handling — GET /health readiness check and non-200/malformed JSON handling."""
    provider = BGERerankerProvider(base_url="http://mock-reranker:8001")

    # Readiness check success (200 OK)
    mock_health_ok = MagicMock()
    mock_health_ok.status_code = 200
    with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=mock_health_ok)):
        assert await provider.check_readiness() is True

    # Readiness check failure (503 Service Unavailable)
    mock_health_fail = MagicMock()
    mock_health_fail.status_code = 503
    with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=mock_health_fail)):
        assert await provider.check_readiness() is False

    # HTTP non-200 status on score()
    mock_500 = MagicMock()
    mock_500.status_code = 500
    mock_500.text = "Internal Server Error"
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=mock_500)):
        with pytest.raises(RuntimeError, match="returned status code 500"):
            await provider.score("query", ["passage"])

    # Malformed JSON payload on score()
    mock_bad_json = MagicMock()
    mock_bad_json.status_code = 200
    mock_bad_json.json.return_value = {"invalid_key": []}
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=mock_bad_json)):
        with pytest.raises(RuntimeError, match="missing 'scores' key"):
            await provider.score("query", ["passage"])


@pytest.mark.asyncio
async def test_reranker_dynamic_top_k_absolute_floor():
    """14. Dynamic top-K thresholding — drops candidates below absolute floor (tau = 0.80)."""
    c1 = {"chunk_id": str(uuid.uuid4()), "text": "High quality GT"}
    c2 = {"chunk_id": str(uuid.uuid4()), "text": "Topically adjacent FP"}
    c3 = {"chunk_id": str(uuid.uuid4()), "text": "FTS noise FP"}
    candidates = [c1, c2, c3]

    # c1=0.94, c2=0.72 (< 0.80 floor), c3=0.28 (< 0.80 floor)
    provider = MockRerankerProvider(scores=[0.94, 0.72, 0.28])
    engine = RerankerEngine(provider=provider, score_threshold=0.80, relative_threshold_factor=0.85)

    results = await engine.rerank("query", candidates, top_k=3)

    # Only c1 survives thresholding
    assert len(results) == 1
    assert results[0]["chunk_id"] == c1["chunk_id"]
    assert results[0]["rerank_score"] == 0.94


@pytest.mark.asyncio
async def test_reranker_dynamic_top_k_relative_dropoff():
    """15. Dynamic top-K thresholding — drops candidates below relative factor (>= 0.85 * top_score)."""
    c1 = {"chunk_id": str(uuid.uuid4()), "text": "GT 1"}
    c2 = {"chunk_id": str(uuid.uuid4()), "text": "GT 2"}
    c3 = {"chunk_id": str(uuid.uuid4()), "text": "Marginal FP"}
    candidates = [c1, c2, c3]

    # top_score=0.92, 0.85 * 0.92 = 0.782.
    # c1=0.92 (keep), c2=0.86 (keep: >= 0.80 and >= 0.782), c3=0.75 (drop: < 0.782 even though 0.75 relative)
    provider = MockRerankerProvider(scores=[0.92, 0.86, 0.75])
    engine = RerankerEngine(provider=provider, score_threshold=0.80, relative_threshold_factor=0.85)

    results = await engine.rerank("query", candidates, top_k=3)

    assert len(results) == 2
    assert results[0]["chunk_id"] == c1["chunk_id"]
    assert results[1]["chunk_id"] == c2["chunk_id"]


@pytest.mark.asyncio
async def test_reranker_dynamic_top_k_top1_fallback_guarantee():
    """16. Top-1 fallback guarantee — retains top-scored chunk when all candidates score below threshold."""
    c1 = {"chunk_id": str(uuid.uuid4()), "text": "Low score chunk 1"}
    c2 = {"chunk_id": str(uuid.uuid4()), "text": "Low score chunk 2"}
    candidates = [c1, c2]

    # Both score below floor (0.35 and 0.20 < 0.80)
    provider = MockRerankerProvider(scores=[0.35, 0.20])
    engine = RerankerEngine(provider=provider, score_threshold=0.80, relative_threshold_factor=0.85)

    results = await engine.rerank("query", candidates, top_k=3)

    # Must return top 1 candidate c1
    assert len(results) == 1
    assert results[0]["chunk_id"] == c1["chunk_id"]
    assert results[0]["rerank_score"] == 0.35


