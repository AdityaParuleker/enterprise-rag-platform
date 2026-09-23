"""
Secure Metadata-Only Evaluation Tracing Model (Phase 9)
Defines EvaluationTrace, RetrievalTraceItem, and RerankTraceItem.
MUST NOT store raw prompt text, document text, compressed context, secrets, or generated answer content by default.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class RetrievalTraceItem:
    chunk_id: str
    rrf_score: float
    rank: int
    included_in_context: bool = False


@dataclass
class RerankTraceItem:
    chunk_id: str
    rerank_score: float
    rank: int
    included_in_context: bool = False


@dataclass
class EvaluationTrace:
    """Metadata-only execution trace for evaluation pipeline analysis."""
    input_guard_action: Optional[str] = None
    input_guard_reason: Optional[str] = None
    retrieved_candidates: List[RetrievalTraceItem] = field(default_factory=list)
    reranked_candidates: List[RerankTraceItem] = field(default_factory=list)
    compressed_candidate_ids: List[str] = field(default_factory=list)
    evidence_score: Optional[float] = None
    evidence_action: Optional[str] = None
    generation_status: Optional[str] = None
    citation_validation_status: Optional[str] = None
    reranker_min_ratio: Optional[float] = None
    reranker_drift_alert: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Returns safe metadata dictionary for audit and evaluation recording without sensitive text."""
        return {
            "input_guard_action": self.input_guard_action,
            "input_guard_reason": self.input_guard_reason,
            "retrieved_count": len(self.retrieved_candidates),
            "reranked_count": len(self.reranked_candidates),
            "compressed_count": len(self.compressed_candidate_ids),
            "retrieved_chunk_ids": [r.chunk_id for r in self.retrieved_candidates],
            "reranked_chunk_ids": [r.chunk_id for r in self.reranked_candidates],
            "evidence_score": self.evidence_score,
            "evidence_action": self.evidence_action,
            "generation_status": self.generation_status,
            "citation_validation_status": self.citation_validation_status,
            "reranker_min_ratio": self.reranker_min_ratio,
            "reranker_drift_alert": self.reranker_drift_alert,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
        }
