"""
Administrative API Module (Section 15 Repo Structure)
"""

import os
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from backend.app.auth.rbac import require_permission

router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])


class StrictGroundingConfigRequest(BaseModel):
    strict_grounding: bool


@router.get("/config")
async def get_system_config(current_user: Dict = Depends(require_permission("chat"))):
    raw_env = os.getenv("STRICT_GROUNDING", "true").strip().lower()
    is_strict = raw_env in ("true", "1", "yes")
    reranker_provider = os.getenv("RERANKER_PROVIDER", "http").strip().lower()

    reranker_status = "offline_fallback"
    if reranker_provider == "mock":
        reranker_status = "active_mock"
    else:
        try:
            from backend.app.retrieval.reranker import BGERerankerProvider
            is_ready = await BGERerankerProvider().check_readiness()
            if is_ready:
                reranker_status = "active_http"
        except Exception:
            pass

    return {
        "data": {
            "strict_grounding": is_strict,
            "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
            "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            "reranker_provider": reranker_provider,
            "reranker_status": reranker_status
        },
        "error": None
    }


@router.post("/config/strict-grounding")
async def set_strict_grounding_config(
    req: StrictGroundingConfigRequest,
    current_user: Dict = Depends(require_permission("chat"))
):
    val_str = "true" if req.strict_grounding else "false"
    os.environ["STRICT_GROUNDING"] = val_str
    return {
        "data": {
            "strict_grounding": req.strict_grounding,
            "message": f"STRICT_GROUNDING environment setting dynamically set to '{val_str}'"
        },
        "error": None
    }
