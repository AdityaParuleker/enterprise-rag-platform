"""
Cross-Encoder Reranker Module (Phase 6 — Checkpoint 6.1)
Implements provider interface, BGERerankerProvider HTTP client, MockRerankerProvider,
and RerankerEngine with validation, length budget truncation, graceful RRF fallback, and deterministic sorting.
"""

import os
import math
import logging
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import httpx

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MAX_PASSAGE_CHARS = 2048
DEFAULT_RERANKER_TIMEOUT_SECONDS = 3.0
DEFAULT_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class RerankerProvider(ABC):
    """Abstract base class interface for cross-encoder reranker inference providers."""

    @abstractmethod
    async def score(self, query: str, passages: List[str]) -> List[float]:
        """
        Score a list of passage strings against a single search query string.

        Args:
            query: The user search/question query string.
            passages: List of passage text strings to score against the query.

        Returns:
            List of raw float relevance scores corresponding 1-to-1 with input passages.
        """
        pass

    @abstractmethod
    async def check_readiness(self) -> bool:
        """Check if the reranker inference backend service is online and ready."""
        pass


class BGERerankerProvider(RerankerProvider):
    """
    Production HTTP provider for BAAI/bge-reranker-v2-m3 cross-encoder inference service.
    Sends single-batch HTTP requests to POST {base_url}/rerank.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None
    ):
        self.base_url = (base_url or os.getenv("RERANKER_BASE_URL", "http://localhost:8001")).rstrip("/")
        self.model_name = model_name or os.getenv("RERANKER_MODEL_NAME", DEFAULT_RERANKER_MODEL_NAME)

        if timeout_seconds is None:
            env_val = os.getenv("RERANKER_TIMEOUT_SECONDS")
            try:
                self.timeout_seconds = float(env_val) if env_val else DEFAULT_RERANKER_TIMEOUT_SECONDS
                if self.timeout_seconds <= 0:
                    self.timeout_seconds = DEFAULT_RERANKER_TIMEOUT_SECONDS
            except ValueError:
                self.timeout_seconds = DEFAULT_RERANKER_TIMEOUT_SECONDS
        else:
            self.timeout_seconds = float(timeout_seconds) if timeout_seconds > 0 else DEFAULT_RERANKER_TIMEOUT_SECONDS

    async def score(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []

        url = f"{self.base_url}/rerank"
        payload = {
            "model": self.model_name,
            "query": query,
            "passages": passages
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, json=payload)

            if response.status_code != 200:
                raise RuntimeError(
                    f"Reranker HTTP provider returned status code {response.status_code}: {response.text}"
                )

            data = response.json()
            if not isinstance(data, dict) or "scores" not in data:
                raise RuntimeError("Invalid reranker HTTP response: missing 'scores' key in JSON payload.")

            raw_scores = data["scores"]
            if not isinstance(raw_scores, list):
                raise RuntimeError("Invalid reranker HTTP response: 'scores' is not a JSON list.")

            if len(raw_scores) != len(passages):
                raise RuntimeError(
                    f"Reranker provider score length mismatch: expected {len(passages)}, got {len(raw_scores)}"
                )

            scores = []
            for s in raw_scores:
                val = float(s)
                if not math.isfinite(val):
                    raise ValueError(f"Reranker provider returned non-finite score: {s}")
                scores.append(val)

            return scores

        except Exception as e:
            logger.warning(f"BGERerankerProvider HTTP score request failed: {e}")
            raise RuntimeError(f"BGERerankerProvider execution failed: {e}") from e

    async def check_readiness(self) -> bool:
        url = f"{self.base_url}/health"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.get(url)
                return resp.status_code == 200
        except Exception:
            return False


class MockRerankerProvider(RerankerProvider):
    """
    Deterministic mock reranker provider for test/offline environments.
    """

    def __init__(self, scores: Optional[List[float]] = None):
        self.preset_scores = scores

    async def score(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
        if self.preset_scores is not None:
            if len(self.preset_scores) != len(passages):
                raise RuntimeError(
                    f"Mock provider score count mismatch: expected {len(passages)}, got {len(self.preset_scores)}"
                )
            return list(self.preset_scores)
        # Default deterministic score: 1.0 / (idx + 1)
        return [round(1.0 / (i + 1), 4) for i in range(len(passages))]

    async def check_readiness(self) -> bool:
        return True


def get_reranker_provider() -> RerankerProvider:
    """Factory function for instantiating the configured RerankerProvider."""
    provider_type = os.getenv("RERANKER_PROVIDER", "http").lower().strip()
    if provider_type == "mock":
        return MockRerankerProvider()
    return BGERerankerProvider()


class RerankerEngine:
    """
    Orchestrates cross-encoder candidate re-scoring, UUID validation, passage character budget truncation,
    graceful fallback to RRF candidates on provider errors, and deterministic sorting.
    """

    def __init__(
        self,
        provider: Optional[RerankerProvider] = None,
        max_passage_chars: int = DEFAULT_RERANKER_MAX_PASSAGE_CHARS,
        score_threshold: float = 0.75,
        relative_threshold_factor: float = 0.75,
        score_floor_min: float = 0.30,
        drift_alert_threshold: float = 0.78
    ):
        self.provider = provider or get_reranker_provider()
        self.max_passage_chars = max_passage_chars
        self.score_threshold = score_threshold
        self.relative_threshold_factor = relative_threshold_factor
        self.score_floor_min = score_floor_min
        self.drift_alert_threshold = drift_alert_threshold

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-score candidate chunks using cross-encoder provider and apply dynamic thresholding.

        Args:
            query: User search/question string.
            candidates: List of ACL-authorized candidate chunk dicts from HybridSearchEngine.
            top_k: Number of reranked candidates to return (default 5).

        Returns:
            List of candidate chunk dicts ordered by rerank_score DESC, chunk_id ASC.
            Applies absolute score floor (0.80) and relative drop-off (>= 0.75 * top_score).
            Top-1 fallback applies only if top_score >= score_floor_min (0.30) to prevent unanswerable hallucination.
            Logs telemetry warnings if candidate score ratios drop below drift_alert_threshold (0.78).
            On provider error, falls back safely to original RRF candidate order.
        """
        if not candidates:
            return []

        # 1. Validate chunk_id UUID format on all candidates
        for idx, candidate in enumerate(candidates):
            cid = candidate.get("chunk_id")
            if not cid:
                raise ValueError(f"Candidate at index {idx} is missing required 'chunk_id' field.")
            try:
                uuid.UUID(str(cid))
            except (ValueError, TypeError, AttributeError):
                raise ValueError(f"Candidate at index {idx} has invalid UUID string 'chunk_id': '{cid}'")

        target_k = min(max(1, top_k), len(candidates))

        # 2. Extract passage text truncated to application character safety budget
        passages = [str(c.get("text", ""))[:self.max_passage_chars] for c in candidates]

        # 3. Invoke provider in ONE single batch call with fallback handling
        try:
            scores = await self.provider.score(query, passages)
            if len(scores) != len(candidates):
                raise RuntimeError(
                    f"Reranker provider returned {len(scores)} scores for {len(candidates)} candidates."
                )

            # Construct new shallow-copy candidate dicts (preventing in-place object mutation)
            reranked = []
            for i, candidate in enumerate(candidates):
                s = float(scores[i])
                if not math.isfinite(s):
                    raise ValueError(f"Provider returned non-finite rerank_score: {scores[i]}")

                new_candidate = dict(candidate)
                new_candidate["rerank_score"] = s
                reranked.append(new_candidate)

            # Deterministic sort: primary rerank_score DESC, secondary chunk_id ASC
            reranked.sort(key=lambda x: (-x["rerank_score"], str(x["chunk_id"])))

            # Candidate pool capped at target_k
            pool = reranked[:target_k]
            top_score = pool[0]["rerank_score"]

            # Dynamic top-K thresholding & Telemetry Audit:
            # a. Keep any chunk scoring above absolute floor (self.score_threshold)
            # b. Keep a chunk only if score >= self.relative_threshold_factor * top_score
            filtered = []
            for c in pool:
                s = c["rerank_score"]
                ratio = (s / top_score) if top_score > 0 else 0.0

                # Telemetry Alert: Log if a secondary candidate with significant raw relevance (s >= 0.70) drops below drift alert threshold
                if s >= 0.70 and ratio < self.drift_alert_threshold:
                    logger.warning(
                        f"[RERANKER_DRIFT_ALERT] Candidate '{c['chunk_id']}' ratio ({ratio:.4f}) is below alert threshold ({self.drift_alert_threshold:.2f}). "
                        f"Top score = {top_score:.4f}, Candidate score = {s:.4f}."
                    )

                if s >= self.score_threshold and ratio >= self.relative_threshold_factor:
                    filtered.append(c)

            # c. Fallback guarantee: retain top-1 chunk ONLY if top_score >= score_floor_min (0.30)
            # Below 0.30, return [] so EvidenceScorer & prompt builder abstain safely
            if not filtered and pool:
                if top_score >= self.score_floor_min:
                    filtered = [pool[0]]
                else:
                    filtered = []

            return filtered

        except Exception as e:
            logger.warning(
                f"[RERANKER_FALLBACK] Reranker provider score execution failed: {e}. "
                f"Falling back safely to original authorized RRF candidate order."
            )
            # Safe Fallback: Return original candidates in RRF score order without mutating candidates
            return [dict(c) for c in candidates[:target_k]]
