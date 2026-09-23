"""
Evaluation Metric Engine (Phase 9)
Implements precise mathematical evaluation functions for retrieval, generation, faithfulness, answer correctness, and citations.
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple


def calculate_recall_at_k(ground_truth_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    """
    Recall@k = |set(GT) ∩ set(R[:k])| / |set(GT)|
    For expected no-evidence queries (|GT| == 0): returns 1.0 if |R[:k]| == 0 else 0.0.
    """
    gt_set = set(ground_truth_ids or [])
    ret_set = set((retrieved_ids or [])[:k])

    if not gt_set:
        return 1.0 if not ret_set else 0.0

    match_count = len(gt_set.intersection(ret_set))
    return round(match_count / len(gt_set), 4)


def calculate_precision_at_k(ground_truth_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    """
    Precision@k = |set(GT) ∩ set(R[:k])| / min(k, |R|)
    Returns 0.0 if retrieved set R is empty.
    """
    gt_set = set(ground_truth_ids or [])
    sub_ret = (retrieved_ids or [])[:k]
    ret_set = set(sub_ret)

    if not sub_ret:
        return 0.0

    match_count = len(gt_set.intersection(ret_set))
    denominator = min(k, len(sub_ret))
    return round(match_count / denominator, 4)


def calculate_mrr(ground_truth_ids: List[str], retrieved_ids: List[str]) -> Optional[float]:
    """
    MRR = 1 / rank(first ground-truth chunk in R).
    Returns 0.0 when no ground-truth chunk appears in retrieved results.
    Returns None (NULL) for expected no-evidence questions (|GT| == 0).
    """
    gt_set = set(ground_truth_ids or [])
    if not gt_set:
        return None

    for rank_idx, chunk_id in enumerate(retrieved_ids or [], start=1):
        if chunk_id in gt_set:
            return round(1.0 / rank_idx, 4)

    return 0.0


def calculate_faithfulness(
    answer: str,
    context_chunks: List[str],
    verifier: Optional[Any] = None
) -> float:
    """
    Model-based OutputGuard bounded entailment verifier (T=0.0, max 10 claims, 10s timeout).
    Faithfulness = (supported claims) / (total extracted claims).
    Returns 1.0 if zero claims extracted (e.g. refusal answer).
    """
    if not answer or not answer.strip():
        return 1.0

    # Extract sentences / claims
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip() and len(s.strip()) > 10]
    claims = sentences[:10]  # Max 10 claims limit

    if not claims:
        return 1.0

    if not context_chunks:
        # If there is an answer but zero context chunks, claims are ungrounded unless answer is a refusal
        if "could not find sufficient information" in answer.lower() or "insufficient information" in answer.lower():
            return 1.0
        return 0.0

    combined_context = " ".join(context_chunks).lower()

    if verifier and hasattr(verifier, "evaluate_entailment"):
        supported = 0
        for claim in claims:
            res = verifier.evaluate_entailment(claim, context_chunks)
            if getattr(res, "action", "") == "ALLOW" or getattr(res, "reason", "") == "CLAIM_SUPPORTED":
                supported += 1
        return round(supported / len(claims), 4)

    # Deterministic fallback heuristic using n-gram containment for test harness repeatability
    supported = 0
    for claim in claims:
        claim_words = set(re.findall(r"\w+", claim.lower()))
        if not claim_words:
            continue
        # Count matched content words in combined context
        matched = sum(1 for w in claim_words if len(w) > 3 and w in combined_context)
        if matched / max(1, len([w for w in claim_words if len(w) > 3])) >= 0.5:
            supported += 1
        elif "could not find" in claim.lower() or "insufficient" in claim.lower():
            supported += 1

    return round(supported / len(claims), 4)


def _tokenize_text(text: str) -> List[str]:
    """Normalizes text by lowercasing, stripping punctuation, and splitting into tokens."""
    if not text:
        return []
    clean = re.sub(r"[^\w\s]", "", text.lower())
    return [t for t in clean.split() if t]


def calculate_answer_correctness_token_f1(generated_answer: str, expected_answer: str) -> float:
    """
    Calculates token-level F1 score after lowercasing, stripping punctuation, and normalizing whitespace.
    """
    gen_tokens = _tokenize_text(generated_answer)
    exp_tokens = _tokenize_text(expected_answer)

    if not gen_tokens and not exp_tokens:
        return 1.0
    if not gen_tokens or not exp_tokens:
        return 0.0

    gen_counts: Dict[str, int] = {}
    for t in gen_tokens:
        gen_counts[t] = gen_counts.get(t, 0) + 1

    exp_counts: Dict[str, int] = {}
    for t in exp_tokens:
        exp_counts[t] = exp_counts.get(t, 0) + 1

    overlap = 0
    for t, count in gen_counts.items():
        if t in exp_counts:
            overlap += min(count, exp_counts[t])

    precision = overlap / len(gen_tokens)
    recall = overlap / len(exp_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


def calculate_answer_correctness_semantic(
    generated_answer: str,
    expected_answer: str,
    embedding_provider: Optional[Any] = None
) -> float:
    """
    Calculates cosine similarity between embeddings of generated answer and expected answer using bge-large-en-v1.5.
    Falls back to token vector cosine similarity if embedding provider is not supplied.
    """
    if not generated_answer and not expected_answer:
        return 1.0
    if not generated_answer or not expected_answer:
        return 0.0

    if embedding_provider and hasattr(embedding_provider, "embed_texts"):
        vecs = embedding_provider.embed_texts([generated_answer, expected_answer])
        v1, v2 = vecs[0], vecs[1]
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return round(max(0.0, min(1.0, dot / (norm1 * norm2))), 4)

    # Token vector fallback cosine similarity
    gen_tokens = _tokenize_text(generated_answer)
    exp_tokens = _tokenize_text(expected_answer)

    vocab = sorted(list(set(gen_tokens + exp_tokens)))
    if not vocab:
        return 1.0

    v1 = [gen_tokens.count(w) for w in vocab]
    v2 = [exp_tokens.count(w) for w in vocab]

    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return round(max(0.0, min(1.0, dot / (norm1 * norm2))), 4)


def calculate_citation_precision_and_recall(
    cited_chunk_ids: List[str],
    ground_truth_ids: List[str]
) -> Tuple[float, float]:
    """
    Calculates Citation Precision and Citation Recall.
    Citation Precision = |cited ∩ GT| / |cited|  (if |cited| == 0: 1.0 if |GT| == 0 else 0.0)
    Citation Recall    = |cited ∩ GT| / |GT|     (if |GT| == 0: 1.0 if |cited| == 0 else 0.0)
    """
    cited_set = set(cited_chunk_ids or [])
    gt_set = set(ground_truth_ids or [])

    match_count = len(cited_set.intersection(gt_set))

    if not cited_set:
        prec = 1.0 if not gt_set else 0.0
    else:
        prec = match_count / len(cited_set)

    if not gt_set:
        rec = 1.0 if not cited_set else 0.0
    else:
        rec = match_count / len(gt_set)

    return round(prec, 4), round(rec, 4)
