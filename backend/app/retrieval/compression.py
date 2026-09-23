"""
Contextual Compression & ACL-Protected Neighbor Stitching Module (Phase 7 — Checkpoint 7.3)
Filters reranked passages to query-relevant sentences, enforces ACL-protected prev_chunk_id/next_chunk_id neighbor stitching,
preserves candidate text non-mutation contract via 'compressed_text', and provides safe fallback.
"""

import re
import uuid
import logging
from typing import List, Dict, Any, Optional
import asyncpg

from backend.app.db.connection import get_db_pool
from backend.app.retrieval.acl_filter import DocumentACLFilter

logger = logging.getLogger(__name__)


class ContextualCompressor:
    """
    Extracts query-relevant sentences from candidate chunks and stitches neighboring sentence context
    only when neighbor chunks pass strict ACL authorization checks.
    """

    def __init__(
        self,
        db_pool: Optional[asyncpg.Pool] = None,
        acl_filter: Optional[DocumentACLFilter] = None
    ):
        self.db_pool = db_pool
        self.acl_filter = acl_filter or DocumentACLFilter()

    async def compress_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        tenant_id: str,
        user_id: str,
        user_roles: Optional[List[str]] = None,
        conn: Optional[asyncpg.Connection] = None
    ) -> List[Dict[str, Any]]:
        """
        Compress candidate chunk text into query-relevant sentences with ACL-protected neighbor stitching.

        Args:
            query: User query string.
            candidates: List of candidate chunk dicts from RerankerEngine.
            tenant_id: Requesting tenant UUID.
            user_id: Requesting user UUID.
            user_roles: Resolved user role UUIDs.
            conn: Optional database connection.

        Returns:
            List of new shallow-copy candidate dicts containing 'compressed_text' field.
        """
        if not candidates:
            return []

        compressed_results = []

        for candidate in candidates:
            try:
                new_cand = dict(candidate)
                original_text = candidate.get("text", "")

                # 1. Extract query terms for sentence relevance matching
                query_terms = set(re.findall(r"\w+", query.lower()))
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", original_text) if s.strip()]

                relevant_sentences = []
                for idx, sentence in enumerate(sentences):
                    sentence_terms = set(re.findall(r"\w+", sentence.lower()))
                    overlap = query_terms.intersection(sentence_terms)
                    # Heuristic: sentence is relevant if it shares query terms or is first sentence
                    if overlap or idx == 0:
                        relevant_sentences.append(sentence)

                # 2. Check if neighbor stitching is needed (if relevant sentences < full passage)
                prev_id = candidate.get("prev_chunk_id")
                next_id = candidate.get("next_chunk_id")

                neighbor_prev_text = ""
                neighbor_next_text = ""

                # ACL-protected neighbor stitching
                if prev_id:
                    neighbor_prev_text = await self._fetch_acl_authorized_neighbor(
                        neighbor_chunk_id=str(prev_id),
                        tenant_id=tenant_id,
                        user_id=user_id,
                        user_roles=user_roles,
                        conn=conn
                    )

                if next_id:
                    neighbor_next_text = await self._fetch_acl_authorized_neighbor(
                        neighbor_chunk_id=str(next_id),
                        tenant_id=tenant_id,
                        user_id=user_id,
                        user_roles=user_roles,
                        conn=conn
                    )

                # Assemble compressed passage
                if relevant_sentences:
                    core_text = " ".join(relevant_sentences)
                    parts = []
                    if neighbor_prev_text:
                        parts.append(neighbor_prev_text)
                    parts.append(core_text)
                    if neighbor_next_text:
                        parts.append(neighbor_next_text)

                    new_cand["compressed_text"] = " ".join(parts)
                else:
                    # Fallback to original text if no sentence passed relevance threshold
                    new_cand["compressed_text"] = original_text

                compressed_results.append(new_cand)

            except Exception as e:
                logger.warning(
                    f"[COMPRESSOR_FALLBACK] Contextual compression failed for chunk {candidate.get('chunk_id')}: {e}. "
                    f"Falling back safely to original passage text."
                )
                fallback_cand = dict(candidate)
                fallback_cand["compressed_text"] = candidate.get("text", "")
                compressed_results.append(fallback_cand)

        return compressed_results

    async def _fetch_acl_authorized_neighbor(
        self,
        neighbor_chunk_id: str,
        tenant_id: str,
        user_id: str,
        user_roles: Optional[List[str]] = None,
        conn: Optional[asyncpg.Connection] = None
    ) -> str:
        """
        Fetch neighbor chunk text strictly enforcing ACL authorization for neighbor chunk's parent document.
        Returns empty string if neighbor chunk is unauthorized or missing.
        """
        try:
            n_uuid = uuid.UUID(neighbor_chunk_id.strip())
            t_uuid = uuid.UUID(tenant_id.strip())
        except (ValueError, TypeError, AttributeError):
            return ""

        acl_sql, acl_params = self.acl_filter.build_acl_where_clause(
            user_id=user_id,
            tenant_id=tenant_id,
            user_roles=user_roles,
            param_offset=3
        )

        query = f"""
        SELECT c.text
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.id = $1
          AND c.tenant_id = $2
          AND {acl_sql};
        """

        params = [n_uuid, t_uuid] + acl_params

        try:
            if conn is not None:
                row = await conn.fetchrow(query, *params)
            else:
                pool = self.db_pool if self.db_pool is not None else await get_db_pool()
                row = await pool.fetchrow(query, *params)

            return row["text"] if row else ""
        except Exception as e:
            logger.warning(f"ACL neighbor chunk lookup failed for {neighbor_chunk_id}: {e}")
            return ""
