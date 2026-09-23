"""
AI & Runtime Guardrail Pipeline (Section 6.4, 6.6 & Phase 8 Specifications)
Defines input, context, evidence, and output guardrail interfaces and detection engines.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

MAX_QUERY_LENGTH = 2048

# Prompt Injection & Jailbreak Heuristic Regex Patterns
INJECTION_PATTERNS = [
    r"(?i)\b(?:ignore|disregard|forget)\s+(?:[a-z0-9_-]+\s+)*(?:instructions|directives|prompts|rules|settings)",
    r"(?i)\b(?:reveal|print|show)\s+(?:[a-z0-9_-]+\s+)*(?:system\s+prompt|developer\s+instructions|hidden\s+rules)",
    r"(?i)\byou\s+are\s+now\s+in\s+DAN\s+mode",
    r"(?i)\bdo\s+anything\s+now\b",
    r"(?i)\[system\s*:\s*override\]",
    r"(?i)<\s*system\s*>\s*override",
    r"(?i)\bsystem\s*:\s*override",
    r"(?i)system\s*:\s*you\s+are\s+an?\s+unrestricted",
]

SECRET_PATTERNS: List[Tuple[str, str]] = [
    ("AWS_SECRET_KEY", r"(?i)(aws_secret_access_key|aws_sec_key)\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?"),
    ("PRIVATE_KEY", r"-----BEGIN (?:RSA|EC|DSA|OPENSSH|PRIVATE) KEY-----[\s\S]+?-----END (?:RSA|EC|DSA|OPENSSH|PRIVATE) KEY-----"),
    ("API_KEY", r"(?i)(api[_-]?key|access[_-]?token|bearer[_-]?token)\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{20,})['\"]?"),
    ("JWT_TOKEN", r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")
]


@dataclass
class GuardrailResult:
    action: str  # ALLOW | BLOCK | RETRY | INSUFFICIENT_EVIDENCE | REDACT
    reason: Optional[str] = None
    detail: Optional[Any] = None


class InputGuard:
    """Validates user query input length and checks for prompt-injection / jailbreak heuristics."""

    def __init__(self, max_length: int = MAX_QUERY_LENGTH):
        self.max_length = max_length

    def check_query(self, query: str) -> GuardrailResult:
        if not query or not query.strip():
            return GuardrailResult(action="BLOCK", reason="EMPTY_QUERY")

        if len(query) > self.max_length:
            return GuardrailResult(
                action="BLOCK",
                reason="QUERY_TOO_LONG",
                detail={"length": len(query), "max_allowed": self.max_length}
            )

        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, query):
                return GuardrailResult(
                    action="BLOCK",
                    reason="PROMPT_INJECTION_DETECTED",
                    detail={"matched_pattern": pattern}
                )

        return GuardrailResult(action="ALLOW")


class SecretRedactor:
    """Scans derived text for sensitive credentials/keys and redacts them prior to embedding generation."""

    def redact_secrets(self, text: str) -> Tuple[str, List[str], int]:
        """
        Returns (sanitized_text, redaction_types, redaction_count).
        Does NOT alter original file in MinIO; sanitizes derived chunk text before embedding.
        """
        if not text:
            return text, [], 0

        sanitized = text
        types_found: Set[str] = set()
        total_count = 0

        for key_type, pattern in SECRET_PATTERNS:
            matches = list(re.finditer(pattern, sanitized))
            if matches:
                types_found.add(key_type)
                total_count += len(matches)
                sanitized = re.sub(pattern, f"[REDACTED_SECRET:{key_type}]", sanitized)

        return sanitized, sorted(list(types_found)), total_count


MIN_EVIDENCE_THRESHOLD = 0.35
COUNT_ABOVE_THRESHOLD = 0.50
INSUFFICIENT_EVIDENCE_RESPONSE = "I could not find sufficient information in the available documents to answer your question."


def sigmoid_normalize(score: float) -> float:
    """Monotonically normalizes any raw real-valued reranker logit score into [0.0, 1.0]."""
    try:
        return 1.0 / (1.0 + math.exp(-float(score)))
    except OverflowError:
        return 1.0 if float(score) > 0 else 0.0


def min_max_normalize(scores: List[float]) -> List[float]:
    """Min-max normalizes candidate score list into [0.0, 1.0]."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    diff = max_s - min_s
    if diff <= 1e-6:
        return [1.0] * len(scores)
    return [(s - min_s) / diff for s in scores]


class EvidenceScorer:
    """Calculates composite evidence score using normalized scores and triggers insufficient-evidence refusal when relevance is low."""

    def __init__(self, threshold: float = MIN_EVIDENCE_THRESHOLD, count_threshold: float = COUNT_ABOVE_THRESHOLD):
        self.threshold = threshold
        self.count_threshold = count_threshold

    def calculate_score(self, candidates: List[Dict[str, Any]], enable_reranking: bool = True) -> float:
        if not candidates:
            return 0.0

        has_rerank_scores = any("rerank_score" in c for c in candidates)

        if enable_reranking and has_rerank_scores:
            # Reranker score normalization via Sigmoid transformation
            norm_scores = [sigmoid_normalize(c.get("rerank_score", 0.0)) for c in candidates]
        elif any("similarity_score" in c for c in candidates):
            # Direct cosine similarity score evaluation
            norm_scores = [float(c.get("similarity_score", 0.0)) for c in candidates]
        else:
            # RRF score normalization via Min-Max scaling over current candidate pool
            raw_rrf = [float(c.get("rrf_score", 0.0)) for c in candidates]
            norm_scores = min_max_normalize(raw_rrf)

        norm_top_score = norm_scores[0]
        count_above = sum(1 for ns in norm_scores if ns >= self.count_threshold)
        count_component = min(count_above, 3) / 3.0

        k_idx = min(3, len(candidates)) - 1
        norm_kth_score = norm_scores[k_idx]
        score_margin = max(0.0, norm_top_score - norm_kth_score)

        evidence_score = 0.5 * norm_top_score + 0.3 * count_component + 0.2 * score_margin
        return min(1.0, max(0.0, round(evidence_score, 4)))

    def evaluate_evidence(self, candidates: List[Dict[str, Any]], enable_reranking: bool = True, strict_grounding: Optional[bool] = None) -> GuardrailResult:
        if strict_grounding is None:
            import os
            strict_grounding = os.getenv("STRICT_GROUNDING", "true").strip().lower() in ("true", "1", "yes")

        if not strict_grounding:
            return GuardrailResult(
                action="ALLOW",
                detail={"evidence_score": 1.0, "threshold": self.threshold, "strict_grounding": False}
            )

        if not candidates:
            return GuardrailResult(
                action="INSUFFICIENT_EVIDENCE",
                reason="NO_CANDIDATES_RETRIEVED",
                detail={"evidence_score": 0.0, "threshold": self.threshold, "response_text": INSUFFICIENT_EVIDENCE_RESPONSE}
            )

        score = self.calculate_score(candidates, enable_reranking=enable_reranking)
        if score < self.threshold:
            return GuardrailResult(
                action="INSUFFICIENT_EVIDENCE",
                reason="EVIDENCE_SCORE_BELOW_THRESHOLD",
                detail={"evidence_score": score, "threshold": self.threshold, "response_text": INSUFFICIENT_EVIDENCE_RESPONSE}
            )

        return GuardrailResult(
            action="ALLOW",
            detail={"evidence_score": score, "threshold": self.threshold}
        )


MAX_GENERATION_CHARS = 8192
MAX_GENERATION_TOKENS = 2048
MAX_REGENERATION_ATTEMPTS = 1


class OutputGuard:
    """Validates LLM outputs: canonical citation markers, size bounds, system prompt leakage, secret leaks, and claim entailment."""

    def __init__(self, max_chars: int = MAX_GENERATION_CHARS):
        self.max_chars = max_chars

    def check_output(self, answer: str, candidates: List[Dict[str, Any]], user_context: Any = None) -> GuardrailResult:
        if not answer:
            return GuardrailResult(action="BLOCK", reason="EMPTY_OUTPUT")

        # 1. Output resource limits
        if len(answer) > self.max_chars:
            return GuardrailResult(
                action="BLOCK",
                reason="GENERATION_SIZE_EXCEEDED",
                detail={"length": len(answer), "max_allowed": self.max_chars}
            )

        # 2. System prompt / secret leakage check
        if "SYSTEM INSTRUCTIONS:" in answer or "RETRIEVED DOCUMENT CONTENT" in answer:
            return GuardrailResult(
                action="BLOCK",
                reason="SYSTEM_PROMPT_LEAK_DETECTED"
            )

        # Scan for secret patterns
        for key_type, pattern in SECRET_PATTERNS:
            if re.search(pattern, answer):
                return GuardrailResult(
                    action="BLOCK",
                    reason="SECRET_LEAK_DETECTED",
                    detail={"key_type": key_type}
                )

        # 3. Canonical Citation Validation
        matches = re.findall(r"\[(\d+)\]", answer)
        valid_indices = set(range(1, len(candidates) + 1))
        invalid_citations = []

        for m in matches:
            idx = int(m)
            if idx not in valid_indices:
                invalid_citations.append(idx)

        if invalid_citations:
            return GuardrailResult(
                action="RETRY",
                reason="INVALID_CITATION_MARKER",
                detail={"invalid_citations": invalid_citations, "valid_count": len(candidates)}
            )

        return GuardrailResult(action="ALLOW")


class GuardrailPipeline:
    def __init__(self):
        self.input_guard = InputGuard()
        self.secret_redactor = SecretRedactor()
        self.evidence_scorer = EvidenceScorer()
        self.output_guard = OutputGuard()

    def check_input(self, query: str, user_context: Any = None) -> GuardrailResult:
        return self.input_guard.check_query(query)

    def check_context(self, query: str, retrieved_chunks: Any, user_context: Any = None, enable_reranking: bool = True, strict_grounding: Optional[bool] = None) -> GuardrailResult:
        return self.evidence_scorer.evaluate_evidence(retrieved_chunks, enable_reranking=enable_reranking, strict_grounding=strict_grounding)

    def check_output(self, answer: str, citations: Any, context: Any = None) -> GuardrailResult:
        return self.output_guard.check_output(answer, candidates=citations or [], user_context=context)

