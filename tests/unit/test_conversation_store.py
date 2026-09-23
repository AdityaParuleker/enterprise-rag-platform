"""
Unit tests for ConversationStore & DB Authorization (Checkpoint 7.1).
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.app.memory.conversation_store import ConversationStore


@pytest.mark.asyncio
async def test_create_and_get_conversation_success():
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    conv_id = uuid.uuid4()

    mock_pool = MagicMock()
    mock_row = {
        "id": conv_id,
        "tenant_id": uuid.UUID(tenant_id),
        "user_id": uuid.UUID(user_id),
        "title": "Security Discussion",
        "summary": None,
        "created_at": None
    }
    mock_pool.fetchrow = AsyncMock(return_value=mock_row)

    store = ConversationStore(db_pool=mock_pool)

    # 1. Create conversation
    res = await store.create_conversation(tenant_id=tenant_id, user_id=user_id, title="Security Discussion")
    assert res["id"] == str(conv_id)
    assert res["tenant_id"] == tenant_id
    assert res["user_id"] == user_id
    assert res["title"] == "Security Discussion"

    # 2. Get conversation
    conv = await store.get_conversation(tenant_id=tenant_id, user_id=user_id, conversation_id=str(conv_id))
    assert conv["id"] == str(conv_id)
    assert mock_pool.fetchrow.call_args[0][1] == conv_id
    assert mock_pool.fetchrow.call_args[0][2] == uuid.UUID(tenant_id)
    assert mock_pool.fetchrow.call_args[0][3] == uuid.UUID(user_id)


@pytest.mark.asyncio
async def test_conversation_unauthorized_user_or_tenant_returns_none():
    tenant_id_a = str(uuid.uuid4())
    user_id_a = str(uuid.uuid4())
    tenant_id_b = str(uuid.uuid4())
    user_id_b = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())

    mock_pool = MagicMock()
    mock_pool.fetchrow = AsyncMock(return_value=None)  # DB query returns no rows for unauthorized tenant/user

    store = ConversationStore(db_pool=mock_pool)

    # Attempt to access User A's conversation using User B's credentials
    res = await store.get_conversation(tenant_id=tenant_id_b, user_id=user_id_b, conversation_id=conv_id)
    assert res is None
    mock_pool.fetchrow.assert_called_once()
    sql_args = mock_pool.fetchrow.call_args[0]
    assert sql_args[1] == uuid.UUID(conv_id)
    assert sql_args[2] == uuid.UUID(tenant_id_b)
    assert sql_args[3] == uuid.UUID(user_id_b)


@pytest.mark.asyncio
async def test_list_conversations_filtered_by_tenant_and_user():
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    c1 = uuid.uuid4()
    c2 = uuid.uuid4()

    mock_pool = MagicMock()
    mock_rows = [
        {"id": c1, "tenant_id": uuid.UUID(tenant_id), "user_id": uuid.UUID(user_id), "title": "Conv 1", "summary": None, "created_at": None},
        {"id": c2, "tenant_id": uuid.UUID(tenant_id), "user_id": uuid.UUID(user_id), "title": "Conv 2", "summary": None, "created_at": None}
    ]
    mock_pool.fetch = AsyncMock(return_value=mock_rows)

    store = ConversationStore(db_pool=mock_pool)

    conversations = await store.list_conversations(tenant_id=tenant_id, user_id=user_id, limit=10)
    assert len(conversations) == 2
    assert conversations[0]["id"] == str(c1)
    assert conversations[1]["id"] == str(c2)

    mock_pool.fetch.assert_called_once()
    sql_args = mock_pool.fetch.call_args[0]
    assert sql_args[1] == uuid.UUID(tenant_id)
    assert sql_args[2] == uuid.UUID(user_id)


@pytest.mark.asyncio
async def test_delete_conversation_unauthorized_returns_false():
    tenant_id_owner = str(uuid.uuid4())
    user_id_owner = str(uuid.uuid4())
    tenant_id_attacker = str(uuid.uuid4())
    user_id_attacker = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())

    mock_pool = MagicMock()
    mock_pool.fetchrow = AsyncMock(return_value=None)

    store = ConversationStore(db_pool=mock_pool)

    deleted = await store.delete_conversation(tenant_id=tenant_id_attacker, user_id=user_id_attacker, conversation_id=conv_id)
    assert deleted is False


@pytest.mark.asyncio
async def test_add_messages_and_citations_persistence():
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    mock_pool = MagicMock()

    # Ownership lookup succeeds
    mock_conv_row = {
        "id": uuid.UUID(conv_id),
        "tenant_id": uuid.UUID(tenant_id),
        "user_id": uuid.UUID(user_id),
        "title": "Title",
        "summary": None,
        "created_at": None
    }

    mock_msg_row = {
        "id": uuid.uuid4(),
        "conversation_id": uuid.UUID(conv_id),
        "role": "user",
        "content": "User question",
        "created_at": None
    }

    mock_pool.fetchrow = AsyncMock(side_effect=[mock_conv_row, mock_msg_row])

    mock_redis_manager = MagicMock()
    mock_redis_client = MagicMock()
    mock_redis_client.delete = AsyncMock()
    mock_redis_manager.get_client.return_value = mock_redis_client

    store = ConversationStore(db_pool=mock_pool, redis_manager=mock_redis_manager)

    # Pre-persist user message
    user_msg = await store.add_user_message(tenant_id=tenant_id, user_id=user_id, conversation_id=conv_id, content="User question")
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "User question"
    mock_redis_client.delete.assert_called_once()


@pytest.mark.asyncio
async def test_redis_cache_miss_fallback_to_postgres():
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    m1_id = uuid.uuid4()

    mock_pool = MagicMock()
    mock_rows = [
        {"id": m1_id, "conversation_id": uuid.UUID(conv_id), "role": "user", "content": "Hello", "created_at": None}
    ]
    mock_pool.fetch = AsyncMock(return_value=mock_rows)

    # Redis client raises connection error
    mock_redis_manager = MagicMock()
    mock_redis_client = MagicMock()
    mock_redis_client.get = AsyncMock(side_effect=RuntimeError("Redis connection refused"))
    mock_redis_manager.get_client.return_value = mock_redis_client

    store = ConversationStore(db_pool=mock_pool, redis_manager=mock_redis_manager)

    # Request messages -> should fall back gracefully to Postgres query
    messages = await store.get_recent_messages(tenant_id=tenant_id, user_id=user_id, conversation_id=conv_id, limit=5)
    assert len(messages) == 1
    assert messages[0]["content"] == "Hello"
    mock_pool.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_redis_cache_key_isolation():
    tenant_id = "tenant-123"
    user_id = "user-456"
    conv_id = "conv-789"

    store = ConversationStore()
    key = store._get_redis_key(tenant_id, user_id, conv_id)
    assert key == "recent_messages:tenant-123:user-456:conv-789"
