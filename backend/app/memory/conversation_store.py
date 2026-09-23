"""
Conversation Store & Multi-Turn History Persistence Module (Phase 7 — Checkpoint 7.1)
Implements DB-authoritative conversation memory, SQL-level (tenant_id, user_id) authorization isolation,
dual-tier Redis turn caching, user pre-persistence, assistant post-stream persistence, and rolling LLM history summary.
"""

import json
import uuid
import logging
from typing import List, Dict, Any, Optional
import asyncpg

from backend.app.db.connection import get_db_pool
from backend.app.cache.redis_client import RedisManager

logger = logging.getLogger(__name__)

REDIS_TTL_SECONDS = 86400  # 24 hours


class ConversationStore:
    """
    Manages multi-turn conversation sessions, message history, citations, and caching.
    Enforces strict SQL-level authorization filtering (WHERE tenant_id = :tenant_id AND user_id = :user_id).
    """

    def __init__(self, db_pool: Optional[asyncpg.Pool] = None, redis_manager: Optional[RedisManager] = None):
        self.db_pool = db_pool
        self.redis_manager = redis_manager or RedisManager()

    def _get_redis_key(self, tenant_id: str, user_id: str, conversation_id: str) -> str:
        return f"recent_messages:{tenant_id}:{user_id}:{conversation_id}"

    async def _validate_uuids(self, tenant_id: str, user_id: str, conversation_id: Optional[str] = None):
        if not tenant_id or not isinstance(tenant_id, str):
            raise ValueError("tenant_id must be a non-empty UUID string")
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty UUID string")
        try:
            t_uuid = uuid.UUID(tenant_id.strip())
            u_uuid = uuid.UUID(user_id.strip())
            c_uuid = uuid.UUID(conversation_id.strip()) if conversation_id else None
            return t_uuid, u_uuid, c_uuid
        except (ValueError, TypeError, AttributeError) as e:
            raise ValueError(f"Invalid UUID format: {e}")

    async def create_conversation(
        self,
        tenant_id: str,
        user_id: str,
        title: str = "New Conversation",
        conn: Optional[asyncpg.Connection] = None
    ) -> Dict[str, Any]:
        """Create a new conversation record constrained by tenant_id and user_id."""
        t_uuid, u_uuid, _ = await self._validate_uuids(tenant_id, user_id)
        conv_id = uuid.uuid4()
        clean_title = (title or "New Conversation").strip()[:255]

        query = """
        INSERT INTO conversations (id, tenant_id, user_id, title, summary, created_at)
        VALUES ($1, $2, $3, $4, NULL, CURRENT_TIMESTAMP)
        RETURNING id, tenant_id, user_id, title, summary, created_at;
        """

        if conn is not None:
            row = await conn.fetchrow(query, conv_id, t_uuid, u_uuid, clean_title)
        else:
            pool = self.db_pool if self.db_pool is not None else await get_db_pool()
            row = await pool.fetchrow(query, conv_id, t_uuid, u_uuid, clean_title)

        return {
            "id": str(row["id"]),
            "tenant_id": str(row["tenant_id"]),
            "user_id": str(row["user_id"]),
            "title": row["title"],
            "summary": row["summary"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None
        }

    async def get_conversation(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        conn: Optional[asyncpg.Connection] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch conversation by ID strictly enforcing WHERE tenant_id AND user_id at the DB layer."""
        t_uuid, u_uuid, c_uuid = await self._validate_uuids(tenant_id, user_id, conversation_id)

        query = """
        SELECT id, tenant_id, user_id, title, summary, created_at
        FROM conversations
        WHERE id = $1 AND tenant_id = $2 AND user_id = $3;
        """

        if conn is not None:
            row = await conn.fetchrow(query, c_uuid, t_uuid, u_uuid)
        else:
            pool = self.db_pool if self.db_pool is not None else await get_db_pool()
            row = await pool.fetchrow(query, c_uuid, t_uuid, u_uuid)

        if not row:
            return None

        return {
            "id": str(row["id"]),
            "tenant_id": str(row["tenant_id"]),
            "user_id": str(row["user_id"]),
            "title": row["title"],
            "summary": row["summary"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None
        }

    async def list_conversations(
        self,
        tenant_id: str,
        user_id: str,
        limit: int = 50,
        conn: Optional[asyncpg.Connection] = None
    ) -> List[Dict[str, Any]]:
        """List user conversations ordered by created_at DESC strictly filtered by tenant_id and user_id."""
        t_uuid, u_uuid, _ = await self._validate_uuids(tenant_id, user_id)
        target_limit = min(max(1, limit), 100)

        query = """
        SELECT id, tenant_id, user_id, title, summary, created_at
        FROM conversations
        WHERE tenant_id = $1 AND user_id = $2
        ORDER BY created_at DESC
        LIMIT $3;
        """

        if conn is not None:
            rows = await conn.fetch(query, t_uuid, u_uuid, target_limit)
        else:
            pool = self.db_pool if self.db_pool is not None else await get_db_pool()
            rows = await pool.fetch(query, t_uuid, u_uuid, target_limit)

        return [
            {
                "id": str(row["id"]),
                "tenant_id": str(row["tenant_id"]),
                "user_id": str(row["user_id"]),
                "title": row["title"],
                "summary": row["summary"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None
            }
            for row in rows
        ]

    async def delete_conversation(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        conn: Optional[asyncpg.Connection] = None
    ) -> bool:
        """Delete conversation and associated messages/citations if authorized."""
        t_uuid, u_uuid, c_uuid = await self._validate_uuids(tenant_id, user_id, conversation_id)

        # 1. Check ownership in SQL
        conv = await self.get_conversation(tenant_id, user_id, conversation_id, conn=conn)
        if not conv:
            return False

        query_citations = """
        DELETE FROM message_citations
        WHERE message_id IN (
            SELECT id FROM messages WHERE conversation_id = $1
        );
        """
        query_messages = "DELETE FROM messages WHERE conversation_id = $1;"
        query_conv = "DELETE FROM conversations WHERE id = $1 AND tenant_id = $2 AND user_id = $3;"

        if conn is not None:
            await conn.execute(query_citations, c_uuid)
            await conn.execute(query_messages, c_uuid)
            res = await conn.execute(query_conv, c_uuid, t_uuid, u_uuid)
        else:
            pool = self.db_pool if self.db_pool is not None else await get_db_pool()
            async with pool.acquire() as connection:
                async with connection.transaction():
                    await connection.execute(query_citations, c_uuid)
                    await connection.execute(query_messages, c_uuid)
                    res = await connection.execute(query_conv, c_uuid, t_uuid, u_uuid)

        # Invalidate Redis cache
        try:
            r = self.redis_manager.get_client()
            await r.delete(self._get_redis_key(tenant_id, user_id, conversation_id))
        except Exception as e:
            logger.warning(f"Redis cache delete failed for conversation {conversation_id}: {e}")

        return "DELETE 1" in res

    async def add_user_message(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        content: str,
        conn: Optional[asyncpg.Connection] = None
    ) -> Dict[str, Any]:
        """Pre-persist user message before retrieval/generation starts."""
        t_uuid, u_uuid, c_uuid = await self._validate_uuids(tenant_id, user_id, conversation_id)

        # Verify conversation ownership in DB
        conv = await self.get_conversation(tenant_id, user_id, conversation_id, conn=conn)
        if not conv:
            raise ValueError(f"Conversation {conversation_id} not found or unauthorized for tenant {tenant_id} and user {user_id}")

        msg_id = uuid.uuid4()
        clean_content = str(content or "").strip()

        query = """
        INSERT INTO messages (id, conversation_id, role, content, created_at)
        VALUES ($1, $2, 'user', $3, CURRENT_TIMESTAMP)
        RETURNING id, conversation_id, role, content, created_at;
        """

        if conn is not None:
            row = await conn.fetchrow(query, msg_id, c_uuid, clean_content)
        else:
            pool = self.db_pool if self.db_pool is not None else await get_db_pool()
            row = await pool.fetchrow(query, msg_id, c_uuid, clean_content)

        # Invalidate Redis cache
        try:
            r = self.redis_manager.get_client()
            await r.delete(self._get_redis_key(tenant_id, user_id, conversation_id))
        except Exception as e:
            logger.warning(f"Redis cache invalidation failed: {e}")

        return {
            "id": str(row["id"]),
            "conversation_id": str(row["conversation_id"]),
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None
        }

    async def add_assistant_message(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        conn: Optional[asyncpg.Connection] = None
    ) -> Dict[str, Any]:
        """Post-persist assistant completion and citations after successful stream completion."""
        t_uuid, u_uuid, c_uuid = await self._validate_uuids(tenant_id, user_id, conversation_id)

        conv = await self.get_conversation(tenant_id, user_id, conversation_id, conn=conn)
        if not conv:
            raise ValueError(f"Conversation {conversation_id} not found or unauthorized for tenant {tenant_id} and user {user_id}")

        msg_id = uuid.uuid4()
        clean_content = str(content or "").strip()

        query_msg = """
        INSERT INTO messages (id, conversation_id, role, content, created_at)
        VALUES ($1, $2, 'assistant', $3, CURRENT_TIMESTAMP)
        RETURNING id, conversation_id, role, content, created_at;
        """

        query_cit = """
        INSERT INTO message_citations (id, message_id, citation_index, chunk_id, document_id, page_number, section_path)
        VALUES ($1, $2, $3, $4, $5, $6, $7);
        """

        if conn is not None:
            row = await conn.fetchrow(query_msg, msg_id, c_uuid, clean_content)
            if citations:
                for idx, cit in enumerate(citations):
                    await conn.execute(
                        query_cit,
                        uuid.uuid4(),
                        msg_id,
                        idx + 1,
                        uuid.UUID(str(cit["chunk_id"])),
                        uuid.UUID(str(cit["document_id"])),
                        cit.get("page_number"),
                        cit.get("section_path")
                    )
        else:
            pool = self.db_pool if self.db_pool is not None else await get_db_pool()
            async with pool.acquire() as connection:
                async with connection.transaction():
                    row = await connection.fetchrow(query_msg, msg_id, c_uuid, clean_content)
                    if citations:
                        for idx, cit in enumerate(citations):
                            await connection.execute(
                                query_cit,
                                uuid.uuid4(),
                                msg_id,
                                idx + 1,
                                uuid.UUID(str(cit["chunk_id"])),
                                uuid.UUID(str(cit["document_id"])),
                                cit.get("page_number"),
                                cit.get("section_path")
                            )

        # Invalidate Redis cache
        try:
            r = self.redis_manager.get_client()
            await r.delete(self._get_redis_key(tenant_id, user_id, conversation_id))
        except Exception as e:
            logger.warning(f"Redis cache invalidation failed: {e}")

        return {
            "id": str(row["id"]),
            "conversation_id": str(row["conversation_id"]),
            "role": row["role"],
            "content": row["content"],
            "citations": citations or [],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None
        }

    async def get_recent_messages(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        limit: int = 10,
        conn: Optional[asyncpg.Connection] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent conversation turns with dual-tier Redis caching.
        Enforces tenant and user authorization boundary at SQL layer on cache miss.
        """
        t_uuid, u_uuid, c_uuid = await self._validate_uuids(tenant_id, user_id, conversation_id)
        target_limit = min(max(1, limit), 50)
        redis_key = self._get_redis_key(tenant_id, user_id, conversation_id)

        # 1. Try Redis cache first
        try:
            r = self.redis_manager.get_client()
            cached_data = await r.get(redis_key)
            if cached_data:
                parsed = json.loads(cached_data)
                if isinstance(parsed, list):
                    return parsed[-target_limit:]
        except Exception as e:
            logger.warning(f"Redis cache read failed for conversation {conversation_id}: {e}")

        # 2. Redis miss / error -> Query PostgreSQL with explicit JOIN authorization check
        query = """
        SELECT m.id, m.conversation_id, m.role, m.content, m.created_at
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE c.id = $1 AND c.tenant_id = $2 AND c.user_id = $3
        ORDER BY m.created_at ASC;
        """

        if conn is not None:
            rows = await conn.fetch(query, c_uuid, t_uuid, u_uuid)
        else:
            pool = self.db_pool if self.db_pool is not None else await get_db_pool()
            rows = await pool.fetch(query, c_uuid, t_uuid, u_uuid)

        messages = [
            {
                "id": str(row["id"]),
                "conversation_id": str(row["conversation_id"]),
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None
            }
            for row in rows
        ]

        # 3. Populate Redis cache (best-effort)
        try:
            r = self.redis_manager.get_client()
            await r.set(redis_key, json.dumps(messages), ex=REDIS_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"Redis cache write failed for conversation {conversation_id}: {e}")

        return messages[-target_limit:]

    async def summarize_history_if_needed(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        llm_client: Optional[Any] = None,
        max_turns: int = 10,
        conn: Optional[asyncpg.Connection] = None
    ) -> Optional[str]:
        """
        Rolling history summarization when turn count exceeds threshold.
        Updates conversations.summary in DB. Summary is treated as untrusted conversational context.
        """
        t_uuid, u_uuid, c_uuid = await self._validate_uuids(tenant_id, user_id, conversation_id)
        messages = await self.get_recent_messages(tenant_id, user_id, conversation_id, limit=50, conn=conn)

        if len(messages) <= max_turns or not llm_client:
            conv = await self.get_conversation(tenant_id, user_id, conversation_id, conn=conn)
            return conv.get("summary") if conv else None

        # Extract older turns to summarize
        older_turns = messages[:-max_turns]
        formatted_history = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in older_turns)

        summary_prompt = (
            f"Summarize the following past conversation history concisely in 2-3 sentences. "
            f"Focus strictly on key facts, user goals, and context mentioned.\n\n"
            f"=== BEGIN CONVERSATION HISTORY ===\n{formatted_history}\n=== END CONVERSATION HISTORY ==="
        )

        try:
            summary_response = ""
            raw_gen = llm_client.stream_response(prompt=summary_prompt, system="You are a helpful conversation summarizer.")
            for token in raw_gen:
                summary_response += token

            clean_summary = summary_response.strip()

            query_update = """
            UPDATE conversations
            SET summary = $1
            WHERE id = $2 AND tenant_id = $3 AND user_id = $4;
            """

            if conn is not None:
                await conn.execute(query_update, clean_summary, c_uuid, t_uuid, u_uuid)
            else:
                pool = self.db_pool if self.db_pool is not None else await get_db_pool()
                await pool.execute(query_update, clean_summary, c_uuid, t_uuid, u_uuid)

            return clean_summary
        except Exception as e:
            logger.warning(f"Rolling history summarization failed: {e}")
            conv = await self.get_conversation(tenant_id, user_id, conversation_id, conn=conn)
            return conv.get("summary") if conv else None
