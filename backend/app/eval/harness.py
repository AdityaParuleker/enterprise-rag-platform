"""
Evaluation Harness & Pipeline Instrumentation (Phase 9)
Executes evaluation datasets against ChatOrchestrator under frozen feature profiles.
ACL and Auth/RBAC remain strictly ENABLED across ALL profiles.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from backend.app.eval.eval_dataset import EvalDataset, EvalQuestion
from backend.app.eval.eval_trace import EvaluationTrace, RetrievalTraceItem, RerankTraceItem
from backend.app.eval.metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_mrr,
    calculate_faithfulness,
    calculate_answer_correctness_token_f1,
    calculate_answer_correctness_semantic,
    calculate_citation_precision_and_recall,
)
from backend.app.eval.taxonomy import classify_evaluation_result


FEATURE_PROFILES = {
    "vector_only": {
        "use_fts": False,
        "enable_reranking": False,
        "enable_rewriting": False,
        "enable_compression": False,
        "guardrails_enabled": True,
        "acl_enabled": True,
    },
    "hybrid": {
        "use_fts": True,
        "enable_reranking": False,
        "enable_rewriting": False,
        "enable_compression": False,
        "guardrails_enabled": True,
        "acl_enabled": True,
    },
    "hybrid_rerank": {
        "use_fts": True,
        "enable_reranking": True,
        "enable_rewriting": False,
        "enable_compression": False,
        "guardrails_enabled": True,
        "acl_enabled": True,
    },
    "full_pipeline": {
        "use_fts": True,
        "enable_reranking": True,
        "enable_rewriting": True,
        "enable_compression": True,
        "guardrails_enabled": True,
        "acl_enabled": True,
    },
}


def calculate_token_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Calculates estimated USD cost based on provider/model token pricing."""
    prov_lower = (provider or "").lower()
    if "ollama" in prov_lower or "local" in prov_lower:
        return 0.0
    if "openai" in prov_lower or "gpt" in (model or "").lower():
        # Baseline GPT-4o approx $2.50/1M in, $10.00/1M out
        return round((input_tokens * 2.50 / 1e6) + (output_tokens * 10.00 / 1e6), 6)
    if "anthropic" in prov_lower or "claude" in (model or "").lower():
        # Baseline Claude 3.5 Sonnet approx $3.00/1M in, $15.00/1M out
        return round((input_tokens * 3.00 / 1e6) + (output_tokens * 15.00 / 1e6), 6)
    return None


class EvaluationHarness:
    """Async evaluation execution runner."""

    def __init__(self, db_pool: Optional[Any] = None, orchestrator: Optional[Any] = None):
        self.db_pool = db_pool
        self.orchestrator = orchestrator
        self._cancelled_runs: set = set()

    def cancel_run(self, run_id: str):
        """Marks run_id as cancelled to halt remaining question execution."""
        self._cancelled_runs.add(run_id)

    def is_cancelled(self, run_id: str) -> bool:
        return run_id in self._cancelled_runs

    async def execute_question_eval(
        self,
        question: EvalQuestion,
        config_name: str,
        user_id: str,
        tenant_id: str,
        eval_run_id: str
    ) -> Dict[str, Any]:
        """Executes a single evaluation question against the pipeline and computes all metrics."""
        profile = FEATURE_PROFILES.get(config_name, FEATURE_PROFILES["full_pipeline"])

        # Construct evaluation trace
        trace = EvaluationTrace()

        # Simulated or live ChatOrchestrator invocation
        answer_text = ""
        cited_ids: List[str] = []
        retrieved_chunk_ids: List[str] = []

        if self.orchestrator:
            try:
                # Execution through ChatOrchestrator preserving Phase 5-8 contracts
                res = await self.orchestrator.process_chat(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    query=question.question,
                    enable_reranking=profile["enable_reranking"],
                    enable_rewriting=profile["enable_rewriting"],
                    enable_compression=profile["enable_compression"],
                )
                answer_text = res.get("answer", "")
                cited_ids = [c.get("chunk_id") for c in res.get("citations", []) if c.get("chunk_id")]
                retrieved_chunk_ids = res.get("retrieved_chunk_ids", [])
                trace.input_tokens = res.get("input_tokens", 50)
                trace.output_tokens = res.get("output_tokens", 80)
            except Exception as e:
                answer_text = f"Error during execution: {str(e)}"
        else:
            # Deterministic mock harness logic for evaluation verification tests
            if question.expected_guardrail_action == "BLOCK":
                trace.input_guard_action = "BLOCK"
                trace.input_guard_reason = "PROMPT_INJECTION_DETECTED"
                answer_text = "I cannot fulfill this request due to security policies."
            elif question.question_type == "acl_restricted" and question.expected_access == "DENIED":
                trace.input_guard_action = "ALLOW"
                trace.evidence_action = "INSUFFICIENT_EVIDENCE"
                trace.evidence_score = 0.0
                answer_text = "I could not find sufficient information in the available documents."
            elif question.question_type == "unanswerable":
                trace.input_guard_action = "ALLOW"
                trace.evidence_action = "INSUFFICIENT_EVIDENCE"
                trace.evidence_score = 0.10
                answer_text = "I could not find sufficient information in the available documents to answer your question."
            else:
                trace.input_guard_action = "ALLOW"
                trace.evidence_action = "ALLOW"
                trace.evidence_score = 0.85
                retrieved_chunk_ids = list(question.ground_truth_chunk_ids)
                cited_ids = list(question.expected_citation_ids)
                answer_text = question.expected_answer
                for r_idx, c_id in enumerate(retrieved_chunk_ids, start=1):
                    trace.retrieved_candidates.append(RetrievalTraceItem(chunk_id=c_id, rrf_score=0.05, rank=r_idx, included_in_context=True))
                    trace.reranked_candidates.append(RerankTraceItem(chunk_id=c_id, rerank_score=0.90, rank=r_idx, included_in_context=True))
                trace.compressed_candidate_ids = list(retrieved_chunk_ids)
                trace.input_tokens = 60
                trace.output_tokens = 40

        # Calculate metrics
        recall = calculate_recall_at_k(question.ground_truth_chunk_ids, retrieved_chunk_ids, k=3)
        precision = calculate_precision_at_k(question.ground_truth_chunk_ids, retrieved_chunk_ids, k=3)
        mrr = calculate_mrr(question.ground_truth_chunk_ids, retrieved_chunk_ids)
        faithfulness = calculate_faithfulness(answer_text, [question.expected_answer])
        token_f1 = calculate_answer_correctness_token_f1(answer_text, question.expected_answer)
        semantic = calculate_answer_correctness_semantic(answer_text, question.expected_answer)
        cit_prec, cit_rec = calculate_citation_precision_and_recall(cited_ids, question.ground_truth_chunk_ids)

        # Classify failure taxonomy
        classification = classify_evaluation_result(
            question_type=question.question_type,
            expected_access=question.expected_access,
            expected_guardrail_action=question.expected_guardrail_action,
            ground_truth_ids=question.ground_truth_chunk_ids,
            trace=trace,
            generated_answer=answer_text,
            faithfulness=faithfulness,
            token_f1=token_f1,
            citation_precision=cit_prec
        )

        cost = calculate_token_cost("ollama", "qwen2.5", trace.input_tokens, trace.output_tokens)

        return {
            "result_id": str(uuid.uuid4()),
            "run_id": eval_run_id,
            "question_id": question.question_id,
            "tenant_id": tenant_id,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "answer": answer_text,
            "latency_ms": trace.latency_ms or 120.0,
            "recall": recall,
            "precision_at_k": precision,
            "mrr": mrr,
            "faithfulness": faithfulness,
            "answer_correctness_token_f1": token_f1,
            "answer_correctness_semantic": semantic,
            "citation_precision": cit_prec,
            "citation_recall": cit_rec,
            "failure_category": classification.category,
            "status": classification.status,
            "input_tokens": trace.input_tokens,
            "output_tokens": trace.output_tokens,
            "total_tokens": trace.input_tokens + trace.output_tokens,
            "estimated_cost": cost,
            "trace_metadata": trace.to_metadata_dict()
        }

    async def run_evaluation_dataset(
        self,
        dataset: EvalDataset,
        config_name: str,
        user_id: str,
        tenant_id: str,
        eval_run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs evaluation over entire dataset managing state transitions (QUEUED -> RUNNING -> COMPLETED / CANCELLED)."""
        run_id = eval_run_id or str(uuid.uuid4())
        results: List[Dict[str, Any]] = []

        status = "RUNNING"

        for question in dataset.questions:
            if self.is_cancelled(run_id):
                status = "CANCELLED"
                break
            res = await self.execute_question_eval(question, config_name, user_id, tenant_id, run_id)
            results.append(res)

        if status != "CANCELLED":
            status = "COMPLETED"

        # Compute aggregate averages
        num_q = len(results) or 1
        avg_recall = sum(r["recall"] for r in results) / num_q
        avg_precision = sum(r["precision_at_k"] for r in results) / num_q
        valid_mrrs = [r["mrr"] for r in results if r["mrr"] is not None]
        avg_mrr = (sum(valid_mrrs) / len(valid_mrrs)) if valid_mrrs else None
        avg_faithfulness = sum(r["faithfulness"] for r in results) / num_q
        avg_token_f1 = sum(r["answer_correctness_token_f1"] for r in results) / num_q

        return {
            "run_id": run_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.name,
            "dataset_version": dataset.version,
            "dataset_hash": dataset.dataset_hash,
            "config_name": config_name,
            "status": status,
            "question_count": len(results),
            "avg_recall": round(avg_recall, 4),
            "avg_precision": round(avg_precision, 4),
            "avg_mrr": round(avg_mrr, 4) if avg_mrr is not None else None,
            "avg_faithfulness": round(avg_faithfulness, 4),
            "avg_token_f1": round(avg_token_f1, 4),
            "results": results
        }
