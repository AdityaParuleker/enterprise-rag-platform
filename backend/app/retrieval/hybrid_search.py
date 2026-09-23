"""
Hybrid Search & Reciprocal Rank Fusion (RRF) Engine (Phase 5 — Checkpoint 5.5)
Combines pgvector similarity search and PostgreSQL Full-Text Search via RRF (k=60).
Enforces strict fail-closed error policy on retrieval branch failures while preserving HTTP 400 client validation errors.
"""

import asyncio
from typing import List, Dict, Any, Optional
import asyncpg
from fastapi import HTTPException

from backend.app.retrieval.vector_search import VectorSearchEngine
from backend.app.retrieval.fts_search import FTSSearchEngine

RRF_K_CONSTANT = 60


class RetrievalFailureError(RuntimeError):
    """Exception raised when a retrieval branch fails during hybrid search execution."""
    pass


class HybridSearchEngine:
    """
    Executes dual-branch hybrid retrieval (Vector + FTS) and merges results using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_search_engine: Optional[VectorSearchEngine] = None,
        fts_search_engine: Optional[FTSSearchEngine] = None
    ):
        self.vector_search_engine = vector_search_engine or VectorSearchEngine()
        self.fts_search_engine = fts_search_engine or FTSSearchEngine()

    async def search(
        self,
        query: str,
        query_vector: List[float],
        tenant_id: str,
        limit: int = 30,
        metadata_filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        user_roles: Optional[List[str]] = None,
        conn: Optional[asyncpg.Connection] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute dual-branch hybrid retrieval and merge candidates via RRF.

        Args:
            query: User natural language query string.
            query_vector: 1024-dimensional embedding vector.
            tenant_id: Requesting tenant UUID string.
            limit: Final Top-K result limit (default 30).
            metadata_filters: Optional metadata filters.
            user_id: Optional requesting user UUID string for ACL.
            user_roles: Optional list of active user role UUID strings.
            conn: Optional database connection override.

        Returns:
            List of candidate chunk dicts merged and ordered by rrf_score DESC, chunk_id ASC.

        Raises:
            HTTPException: If metadata filter validation fails (HTTP 400 client error).
            ValueError: If input arguments are malformed (client error).
            RetrievalFailureError: If either retrieval branch encounters a DB/execution error (fail-closed).
        """
        if not tenant_id or not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id must be a non-empty string")

        if limit is None or limit <= 0:
            return []

        # Execute vector and FTS branches concurrently using independent DB connections
        # Vector branch uses the passed connection (or pool if None); FTS branch acquires from pool
        # This prevents asyncpg InterfaceError ("another operation is in progress") on shared connections
        vector_task = self._execute_vector_branch(
            query_vector=query_vector,
            tenant_id=tenant_id,
            limit=limit,
            metadata_filters=metadata_filters,
            user_id=user_id,
            user_roles=user_roles,
            conn=conn
        )

        fts_task = self._execute_fts_branch(
            query=query,
            tenant_id=tenant_id,
            limit=limit,
            metadata_filters=metadata_filters,
            user_id=user_id,
            user_roles=user_roles,
            conn=None
        )

        results = await asyncio.gather(vector_task, fts_task, return_exceptions=True)
        vec_res, fts_res = results[0], results[1]

        # 1. Handle validation exceptions (HTTPException / ValueError) -> propagate client error
        if isinstance(vec_res, (HTTPException, ValueError)):
            raise vec_res
        if isinstance(fts_res, (HTTPException, ValueError)):
            raise fts_res

        # 2. Fail-closed branch failure check: If either branch threw an exception, fail retrieval
        if isinstance(vec_res, Exception):
            raise RetrievalFailureError(f"Vector search branch execution failed: {vec_res}") from vec_res
        if isinstance(fts_res, Exception):
            raise RetrievalFailureError(f"FTS search branch execution failed: {fts_res}") from fts_res

        # 3. Both branches succeeded cleanly -> merge via RRF
        return self.compute_rrf(vec_results=vec_res, fts_results=fts_res, limit=limit)

    async def _execute_vector_branch(
        self,
        query_vector: List[float],
        tenant_id: str,
        limit: int,
        metadata_filters: Optional[Dict[str, Any]],
        user_id: Optional[str],
        user_roles: Optional[List[str]],
        conn: Optional[asyncpg.Connection]
    ) -> List[Dict[str, Any]]:
        return await self.vector_search_engine.search(
            query_vector=query_vector,
            tenant_id=tenant_id,
            limit=limit,
            metadata_filters=metadata_filters,
            user_id=user_id,
            user_roles=user_roles,
            conn=conn
        )


    async def _execute_fts_branch(
        self,
        query: str,
        tenant_id: str,
        limit: int,
        metadata_filters: Optional[Dict[str, Any]],
        user_id: Optional[str],
        user_roles: Optional[List[str]],
        conn: Optional[asyncpg.Connection]
    ) -> List[Dict[str, Any]]:
        return await self.fts_search_engine.search(
            query=query,
            tenant_id=tenant_id,
            limit=limit,
            metadata_filters=metadata_filters,
            user_id=user_id,
            user_roles=user_roles,
            conn=conn
        )

    def compute_rrf(
        self,
        vec_results: List[Dict[str, Any]],
        fts_results: List[Dict[str, Any]],
        limit: int = 30,
        k: int = RRF_K_CONSTANT
    ) -> List[Dict[str, Any]]:
        """
        Merge vector and FTS candidate lists using Reciprocal Rank Fusion (RRF).
        Primary sort: rrf_score DESC
        Secondary tie-break: chunk_id ASC (lexicographical string UUID sort)
        """
        chunk_map: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        # 1. Process vector candidates
        for rank, chunk in enumerate(vec_results, start=1):
            cid = str(chunk["chunk_id"])
            if cid not in chunk_map:
                chunk_map[cid] = dict(chunk)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

        # 2. Process FTS candidates
        for rank, chunk in enumerate(fts_results, start=1):
            cid = str(chunk["chunk_id"])
            if cid not in chunk_map:
                chunk_map[cid] = dict(chunk)
            else:
                if "fts_score" in chunk:
                    chunk_map[cid]["fts_score"] = chunk["fts_score"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

        # 3. Populate final rrf_score into chunk dicts
        merged_chunks = []
        for cid, chunk_item in chunk_map.items():
            chunk_item["rrf_score"] = round(rrf_scores[cid], 6)
            merged_chunks.append(chunk_item)

        # 4. Deterministic sort: primary rrf_score DESC, secondary chunk_id ASC
        merged_chunks.sort(key=lambda x: (-x["rrf_score"], str(x["chunk_id"])))

        return merged_chunks[:limit]
