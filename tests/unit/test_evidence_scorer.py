"""
Unit tests for Checkpoint 8.2 — EvidenceScorer & Composite Refusal.
"""

import pytest
from backend.app.generation.prompt_guard import (
    EvidenceScorer,
    GuardrailResult,
    MIN_EVIDENCE_THRESHOLD,
    INSUFFICIENT_EVIDENCE_RESPONSE
)


def test_evidence_scorer_empty_candidates_returns_insufficient_evidence():
    """Verify empty candidate pool triggers immediate insufficient evidence refusal."""
    scorer = EvidenceScorer()
    res = scorer.evaluate_evidence([])
    assert res.action == "INSUFFICIENT_EVIDENCE"
    assert res.reason == "NO_CANDIDATES_RETRIEVED"
    assert res.detail["response_text"] == INSUFFICIENT_EVIDENCE_RESPONSE
    assert res.detail["evidence_score"] == 0.0


def test_evidence_scorer_high_relevance_candidates_returns_allow():
    """Verify candidates with high rerank scores yield score >= threshold and ALLOW."""
    scorer = EvidenceScorer()
    candidates = [
        {"chunk_id": "c1", "rerank_score": 3.0},
        {"chunk_id": "c2", "rerank_score": 2.0},
        {"chunk_id": "c3", "rerank_score": 1.0},
    ]

    res = scorer.evaluate_evidence(candidates)
    assert res.action == "ALLOW"
    assert res.detail["evidence_score"] > 0.7


def test_evidence_scorer_low_relevance_candidates_triggers_refusal():
    """Verify low rerank scores (below 0.35 composite evidence score) trigger INSUFFICIENT_EVIDENCE."""
    scorer = EvidenceScorer()
    candidates = [
        {"chunk_id": "c1", "rerank_score": -3.0},
        {"chunk_id": "c2", "rerank_score": -5.0},
    ]
    res = scorer.evaluate_evidence(candidates)
    assert res.action == "INSUFFICIENT_EVIDENCE"
    assert res.reason == "EVIDENCE_SCORE_BELOW_THRESHOLD"
    assert res.detail["evidence_score"] < MIN_EVIDENCE_THRESHOLD


def test_evidence_scorer_reranking_disabled_rrf_fallback():
    """Verify when enable_reranking=False evidence scorer uses RRF fallback score."""
    scorer = EvidenceScorer()
    candidates = [
        {"chunk_id": "c1", "rrf_score": 0.01639},  # ~1/(60+1)
    ]
    res = scorer.evaluate_evidence(candidates, enable_reranking=False)
    assert res.action == "ALLOW"
    assert res.detail["evidence_score"] >= 0.35
