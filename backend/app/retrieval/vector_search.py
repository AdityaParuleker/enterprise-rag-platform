"""
Vector Search Module (Phase 4 & Phase 5 — Basic RAG & Tenant-Isolated Vector Retrieval)
Implements tenant-bounded pgvector similarity retrieval with metadata filtering and ACL pushdown.
"""

import json
import uuid
from typing import List, Dict, Any, Optional
import asyncpg

from backend.app.db.connection import get_db_pool
from backend.app.retrieval.metadata_filter import MetadataFilterEngine
from backend.app.retrieval.acl_filter import DocumentACLFilter

VECTOR_DIMENSION = 1024


class VectorSearchEngine:
    """
    Vector search engine for retrieving context chunks using pgvector cosine distance.
    Supports tenant scoping, metadata filtering, and document-level ACL pushdown authorization.
    """

    def __init__(
        self,
        db_pool: Optional[asyncpg.Pool] = None,
        metadata_filter_engine: Optional[MetadataFilterEngine] = None,
        acl_filter: Optional[DocumentACLFilter] = None
    ):
        self.db_pool = db_pool
        self.metadata_filter_engine = metadata_filter_engine or MetadataFilterEngine()
        self.acl_filter = acl_filter or DocumentACLFilter()

    async def search(
        self,
        query_vector: List[float],
        tenant_id: str,
        limit: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        user_roles: Optional[List[str]] = None,
        conn: Optional[asyncpg.Connection] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute tenant-bounded vector similarity search.

        Args:
            query_vector: 1024-dimensional embedding vector.
            tenant_id: Tenant UUID string.
            limit: Top-K result limit (default 10).
            metadata_filters: Optional dict of raw metadata filters.
            user_id: Optional requesting user UUID string for ACL pushdown.
            user_roles: Optional list of resolved user role UUID strings for ACL pushdown.
            conn: Optional database connection override (e.g. for transactions/tests).

        Returns:
            List of result dicts containing chunk/document details, distance, and similarity score.
        """
        if not isinstance(query_vector, (list, tuple)):
            raise ValueError(f"query_vector must be a list or tuple of {VECTOR_DIMENSION} floats")

        if len(query_vector) != VECTOR_DIMENSION:
            raise ValueError(
                f"Query vector dimension mismatch: expected {VECTOR_DIMENSION}, got {len(query_vector)}"
            )

        if not tenant_id or not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id must be a non-empty string")

        if limit is None or limit <= 0:
            return []

        clean_tenant_id = tenant_id.strip()

        # Vector string representation for pgvector $1::vector
        try:
            vector_str = f"[{','.join(str(float(v)) for v in query_vector)}]"
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid elements in query_vector: {e}")

        # Parameter tracking: $1=vector_str, $2=tenant_id
        sql_params: List[Any] = [vector_str, clean_tenant_id]
        where_conditions: List[str] = [
            "c.tenant_id = $2::uuid",
            "d.is_latest = TRUE",
            "d.status = 'INDEXED'"
        ]
        curr_offset = 3

        # 1. Append metadata filter conditions if provided
        if metadata_filters:
            validated = self.metadata_filter_engine.parse_and_validate(metadata_filters)
            meta_conds, meta_params = self.metadata_filter_engine.build_where_clauses(
                validated, param_offset=curr_offset
            )
            where_conditions.extend(meta_conds)
            sql_params.extend(meta_params)
            curr_offset += len(meta_params)

        # 2. Append ACL authorization condition if user_id provided
        if user_id:
            acl_clause, acl_params = self.acl_filter.build_acl_where_clause(
                user_id=user_id,
                tenant_id=clean_tenant_id,
                user_roles=user_roles or [],
                param_offset=curr_offset
            )
            where_conditions.append(acl_clause)
            sql_params.extend(acl_params)
            curr_offset += len(acl_params)

        # Append limit parameter
        sql_params.append(limit)
        limit_param_idx = curr_offset

        where_clause_str = " AND\n  ".join(where_conditions)

        search_query = f"""SELECT
    c.id AS chunk_id,
    c.document_id,
    d.version AS document_version,
    c.text,
    c.page_number,
    c.section_path,
    c.metadata,
    (c.embedding <=> $1::vector) AS distance,
    (1 - (c.embedding <=> $1::vector)) AS similarity_score
FROM chunks c
JOIN documents d ON c.document_id = d.id
WHERE {where_clause_str}
ORDER BY distance ASC, c.id ASC
LIMIT ${limit_param_idx};"""

        if conn is not None:
            rows = await conn.fetch(search_query, *sql_params)
        else:
            pool = self.db_pool if self.db_pool is not None else await get_db_pool()
            rows = await pool.fetch(search_query, *sql_params)

        results = []
        for row in rows:
            raw_meta = row["metadata"]
            if isinstance(raw_meta, str):
                try:
                    metadata = json.loads(raw_meta)
                except Exception:
                    metadata = {}
            elif isinstance(raw_meta, dict):
                metadata = raw_meta
            else:
                metadata = {}

            results.append({
                "chunk_id": str(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "document_version": row["document_version"],
                "text": row["text"],
                "page_number": row["page_number"],
                "section_path": row["section_path"],
                "metadata": metadata,
                "distance": float(row["distance"]),
                "similarity_score": float(row["similarity_score"])
            })

        return results


