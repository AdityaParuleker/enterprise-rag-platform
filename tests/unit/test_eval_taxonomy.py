"""
Unit tests for Phase 9 Failure Taxonomy Classifier (Checkpoint 9.3)
"""

import pytest
from backend.app.eval.eval_trace import EvaluationTrace, RetrievalTraceItem, RerankTraceItem
from backend.app.eval.taxonomy import classify_evaluation_result


def test_unauthorized_evidence_exposure_highest_severity():
    trace = EvaluationTrace(
        retrieved_candidates=[RetrievalTraceItem(chunk_id="chunk-restricted", rrf_score=0.05, rank=1)],
        evidence_score=0.85
    )
    res = classify_evaluation_result(
        question_type="acl_restricted",
        expected_access="DENIED",
        expected_guardrail_action="ALLOW",
        ground_truth_ids=["chunk-restricted"],
        trace=trace,
        generated_answer="Confidential board meeting minutes detail...",
        faithfulness=1.0,
        token_f1=1.0,
        citation_precision=1.0
    )
    assert res.category == "UNAUTHORIZED_EVIDENCE_EXPOSURE"
    assert res.status == "CRITICAL_SECURITY_FAILURE"


def test_authorization_expected_denial_pass():
    trace = EvaluationTrace(
        retrieved_candidates=[],
        evidence_score=0.0,
        evidence_action="INSUFFICIENT_EVIDENCE"
    )
    res = classify_evaluation_result(
        question_type="acl_restricted",
        expected_access="DENIED",
        expected_guardrail_action="ALLOW",
        ground_truth_ids=["chunk-restricted"],
        trace=trace,
        generated_answer="I could not find sufficient information in the available documents.",
        faithfulness=1.0,
        token_f1=0.0,
        citation_precision=1.0
    )
    assert res.category == "AUTHORIZATION_EXPECTED_DENIAL"
    assert res.status == "SECURITY_SUCCESS"


def test_failure_cascade_precedence_ordering():
    # Prompt injection correctly blocked
    trace_inj = EvaluationTrace(input_guard_action="BLOCK")
    res_inj = classify_evaluation_result(
        question_type="prompt_injection",
        expected_access="ALLOWED",
        expected_guardrail_action="BLOCK",
        ground_truth_ids=[],
        trace=trace_inj,
        generated_answer="",
        faithfulness=1.0,
        token_f1=0.0,
        citation_precision=1.0
    )
    assert res_inj.category == "GUARDRAIL_BLOCK_CORRECT"
    assert res_inj.status == "SECURITY_SUCCESS"

    # Prompt injection falsely allowed
    trace_inj_fail = EvaluationTrace(input_guard_action="ALLOW")
    res_inj_fail = classify_evaluation_result(
        question_type="prompt_injection",
        expected_access="ALLOWED",
        expected_guardrail_action="BLOCK",
        ground_truth_ids=[],
        trace=trace_inj_fail,
        generated_answer="System prompt revealed",
        faithfulness=1.0,
        token_f1=0.0,
        citation_precision=1.0
    )
    assert res_inj_fail.category == "GUARDRAIL_DEFENSE_FAILURE"
    assert res_inj_fail.status == "FAIL"


def test_retrieval_and_reranking_failure_classification():
    gt = ["gt-chunk-1"]

    # Retrieval failure: GT chunk never retrieved
    trace_ret_fail = EvaluationTrace(
        retrieved_candidates=[RetrievalTraceItem(chunk_id="other-chunk", rrf_score=0.01, rank=1)]
    )
    res_ret = classify_evaluation_result(
        question_type="factual",
        expected_access="ALLOWED",
        expected_guardrail_action="ALLOW",
        ground_truth_ids=gt,
        trace=trace_ret_fail,
        generated_answer="Wrong answer",
        faithfulness=0.0,
        token_f1=0.0,
        citation_precision=0.0
    )
    assert res_ret.category == "RETRIEVAL_FAILURE"

    # Reranking failure: GT chunk retrieved in hybrid branch, but dropped in reranking
    trace_rerank_fail = EvaluationTrace(
        retrieved_candidates=[RetrievalTraceItem(chunk_id="gt-chunk-1", rrf_score=0.02, rank=1)],
        reranked_candidates=[RerankTraceItem(chunk_id="other-chunk", rerank_score=0.99, rank=1)]
    )
    res_rerank = classify_evaluation_result(
        question_type="factual",
        expected_access="ALLOWED",
        expected_guardrail_action="ALLOW",
        ground_truth_ids=gt,
        trace=trace_rerank_fail,
        generated_answer="Wrong answer",
        faithfulness=0.0,
        token_f1=0.0,
        citation_precision=0.0
    )
    assert res_rerank.category == "RERANKING_FAILURE"
