"""
Unit tests for Phase 9 Evaluation Metric Engine (Checkpoint 9.2)
"""

import pytest
from backend.app.eval.metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_mrr,
    calculate_faithfulness,
    calculate_answer_correctness_token_f1,
    calculate_answer_correctness_semantic,
    calculate_citation_precision_and_recall,
)


def test_recall_at_k_calculation():
    gt = ["chunk-1", "chunk-2", "chunk-3"]
    ret = ["chunk-1", "chunk-4", "chunk-2", "chunk-5"]
    assert calculate_recall_at_k(gt, ret, k=3) == pytest.approx(2 / 3, 0.001)  # chunk-1 and chunk-2 in top 3
    assert calculate_recall_at_k(gt, ret, k=1) == pytest.approx(1 / 3, 0.001)

    # Empty GT (unanswerable / prompt injection)
    assert calculate_recall_at_k([], [], k=3) == 1.0
    assert calculate_recall_at_k([], ["chunk-1"], k=3) == 0.0


def test_precision_at_k_calculation():
    gt = ["chunk-1", "chunk-2"]
    ret = ["chunk-1", "chunk-4", "chunk-2", "chunk-5"]
    assert calculate_precision_at_k(gt, ret, k=2) == 0.5  # 1 match out of 2
    assert calculate_precision_at_k(gt, ret, k=3) == pytest.approx(2 / 3, 0.001)  # 2 matches out of 3
    assert calculate_precision_at_k(gt, [], k=3) == 0.0


def test_mrr_calculation_and_null_edge_cases():
    gt = ["chunk-3", "chunk-4"]
    ret = ["chunk-1", "chunk-2", "chunk-3", "chunk-4"]
    assert calculate_mrr(gt, ret) == pytest.approx(1 / 3, 0.001)  # first match at rank 3

    # No match
    assert calculate_mrr(gt, ["chunk-1", "chunk-2"]) == 0.0

    # Empty GT -> NULL (None)
    assert calculate_mrr([], ret) is None


def test_faithfulness_verifier_scoring():
    context = ["Component 1 operates at standard baseline parameters with high efficiency."]
    supported_ans = "Component 1 operates at standard baseline parameters."
    assert calculate_faithfulness(supported_ans, context) == 1.0

    unsupported_ans = "Component 1 produces nuclear energy and flies to mars."
    assert calculate_faithfulness(unsupported_ans, context) == 0.0

    refusal_ans = "I could not find sufficient information in the available documents."
    assert calculate_faithfulness(refusal_ans, []) == 1.0


def test_answer_correctness_token_f1_normalization():
    ans1 = "The component operates at 100% efficiency!"
    ans2 = "the component operates at 100 efficiency"
    assert calculate_answer_correctness_token_f1(ans1, ans2) == 1.0

    partial = "The component operates slowly."
    f1 = calculate_answer_correctness_token_f1(ans1, partial)
    assert 0.0 < f1 < 1.0


def test_answer_correctness_semantic_similarity():
    ans1 = "The component operates at standard baseline parameters."
    ans2 = "Component 1 operates at standard baseline parameters."
    sim = calculate_answer_correctness_semantic(ans1, ans2)
    assert sim > 0.80

    diff = "Pizza is delicious."
    diff_sim = calculate_answer_correctness_semantic(ans1, diff)
    assert diff_sim < 0.30


def test_citation_precision_and_recall():
    gt = ["chunk-1", "chunk-2"]
    cited = ["chunk-1", "chunk-3"]
    prec, rec = calculate_citation_precision_and_recall(cited, gt)
    assert prec == 0.5  # 1 valid citation out of 2 cited
    assert rec == 0.5   # 1 cited out of 2 GT

    # Empty citations and empty GT
    prec_empty, rec_empty = calculate_citation_precision_and_recall([], [])
    assert prec_empty == 1.0
    assert rec_empty == 1.0
