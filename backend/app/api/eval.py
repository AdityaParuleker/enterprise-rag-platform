"""
Evaluation API Endpoints (Phase 9 Specifications)
Implements /run, /results, /results/compare, and /cancel endpoints with strict RBAC and tenant isolation.
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.app.auth.rbac import get_current_user, require_permission
from backend.app.eval.eval_dataset import generate_standard_benchmark_dataset
from backend.app.eval.harness import EvaluationHarness, FEATURE_PROFILES

router = APIRouter(prefix="/api/v1/eval", tags=["Evaluation"])

# In-memory store for evaluation runs & results for test harness and active jobs
_EVAL_RUNS_STORE: Dict[str, Dict[str, Any]] = {}
_EVAL_RESULTS_STORE: Dict[str, List[Dict[str, Any]]] = {}
_HARNESS_INSTANCE = EvaluationHarness()


class EvalRunRequest(BaseModel):
    config_name: str = Field(default="full_pipeline", description="Configuration profile name")
    dataset_name: Optional[str] = Field(default="standard_rag_benchmark", description="Dataset name")


class EvalCancelRequest(BaseModel):
    run_id: str = Field(..., description="Run ID to cancel")


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
async def run_evaluation(
    req: EvalRunRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permission("eval:run")),
):
    """POST /api/v1/eval/run — Initiates async evaluation run under tenant isolation."""
    if req.config_name not in FEATURE_PROFILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid config_name '{req.config_name}'. Valid choices: {list(FEATURE_PROFILES.keys())}"
        )

    tenant_id = current_user.get("tenant_id")
    user_id = current_user.get("sub")
    run_id = str(uuid.uuid4())

    dataset = generate_standard_benchmark_dataset(tenant_id=tenant_id)

    # Initial state record
    _EVAL_RUNS_STORE[run_id] = {
        "run_id": run_id,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "config_name": req.config_name,
        "dataset_name": dataset.name,
        "dataset_version": dataset.version,
        "dataset_hash": dataset.dataset_hash,
        "status": "QUEUED",
        "question_count": len(dataset.questions),
        "created_at": "2026-09-11T12:00:00Z"
    }
    _EVAL_RESULTS_STORE[run_id] = []

    # Execute dataset run asynchronously / inline for API handler
    async def _bg_run():
        _EVAL_RUNS_STORE[run_id]["status"] = "RUNNING"
        res = await _HARNESS_INSTANCE.run_evaluation_dataset(
            dataset=dataset,
            config_name=req.config_name,
            user_id=user_id,
            tenant_id=tenant_id,
            eval_run_id=run_id
        )
        _EVAL_RUNS_STORE[run_id].update(res)
        _EVAL_RESULTS_STORE[run_id] = res.get("results", [])

    asyncio.create_task(_bg_run())

    return {
        "run_id": run_id,
        "tenant_id": tenant_id,
        "status": "QUEUED",
        "config_name": req.config_name,
        "dataset_hash": dataset.dataset_hash,
        "message": "Evaluation run accepted and queued."
    }


@router.get("/results")
async def get_eval_results(
    run_id: str = Query(..., description="Evaluation Run ID"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permission("eval:read")),
):
    """GET /api/v1/eval/results?run_id=... — Retrieves evaluation run metrics and detailed question results under tenant isolation."""
    run_meta = _EVAL_RUNS_STORE.get(run_id)
    if not run_meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Evaluation run '{run_id}' not found.")

    tenant_id = current_user.get("tenant_id")
    if run_meta.get("tenant_id") != tenant_id:
        # Tenant boundary isolation
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Evaluation run '{run_id}' not found.")

    results = _EVAL_RESULTS_STORE.get(run_id, [])
    return {
        "run_id": run_id,
        "status": run_meta.get("status"),
        "config_name": run_meta.get("config_name"),
        "dataset_name": run_meta.get("dataset_name"),
        "dataset_version": run_meta.get("dataset_version"),
        "dataset_hash": run_meta.get("dataset_hash"),
        "avg_recall": run_meta.get("avg_recall"),
        "avg_precision": run_meta.get("avg_precision"),
        "avg_mrr": run_meta.get("avg_mrr"),
        "avg_faithfulness": run_meta.get("avg_faithfulness"),
        "avg_token_f1": run_meta.get("avg_token_f1"),
        "question_count": run_meta.get("question_count"),
        "results": results
    }


@router.get("/results/compare")
async def compare_eval_results(
    run_ids: str = Query(..., description="Comma-separated run IDs to compare"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permission("eval:compare")),
):
    """GET /api/v1/eval/results/compare?run_ids=id1,id2 — Compares metric deltas across multiple evaluation runs."""
    ids = [i.strip() for i in run_ids.split(",") if i.strip()]
    if not ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No run_ids provided.")

    tenant_id = current_user.get("tenant_id")
    comparisons = []

    for r_id in ids:
        run_meta = _EVAL_RUNS_STORE.get(r_id)
        if not run_meta or run_meta.get("tenant_id") != tenant_id:
            continue
        comparisons.append({
            "run_id": r_id,
            "config_name": run_meta.get("config_name"),
            "status": run_meta.get("status"),
            "dataset_hash": run_meta.get("dataset_hash"),
            "avg_recall": run_meta.get("avg_recall"),
            "avg_precision": run_meta.get("avg_precision"),
            "avg_mrr": run_meta.get("avg_mrr"),
            "avg_faithfulness": run_meta.get("avg_faithfulness"),
            "avg_token_f1": run_meta.get("avg_token_f1"),
        })

    return {
        "tenant_id": tenant_id,
        "run_count": len(comparisons),
        "comparisons": comparisons
    }


@router.post("/cancel")
async def cancel_eval_run(
    req: EvalCancelRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_permission("eval:run")),
):
    """POST /api/v1/eval/cancel — Cancels an in-progress or queued evaluation run."""
    run_meta = _EVAL_RUNS_STORE.get(req.run_id)
    if not run_meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Evaluation run '{req.run_id}' not found.")

    tenant_id = current_user.get("tenant_id")
    if run_meta.get("tenant_id") != tenant_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Evaluation run '{req.run_id}' not found.")

    _HARNESS_INSTANCE.cancel_run(req.run_id)
    run_meta["status"] = "CANCELLED"

    return {
        "run_id": req.run_id,
        "status": "CANCELLED",
        "message": "Evaluation run cancellation requested."
    }
