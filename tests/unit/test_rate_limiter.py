"""
Unit tests for Checkpoint 8.4 — RateLimiter & Redis Fail-Closed policy.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException
from backend.app.auth.rate_limiter import RateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_redis_outage_fails_closed_503():
    """Verify rate limiter raises HTTP 503 Service Unavailable when Redis is unreachable."""
    mock_manager = MagicMock()
    mock_manager.get_client.side_effect = RuntimeError("Redis offline")

    limiter = RateLimiter(route_class="chat", redis_manager=mock_manager)
    mock_request = MagicMock()
    mock_request.client.host = "127.0.0.1"

    with pytest.raises(HTTPException) as exc_info:
        await limiter.check_rate_limit(mock_request, tenant_id="t1", user_id="u1")

    assert exc_info.value.status_code == 503
    assert "Rate limiter service unavailable" in exc_info.value.detail


@pytest.mark.asyncio
async def test_rate_limiter_allows_under_limit():
    """Verify requests under the token limit are allowed."""
    mock_redis = MagicMock()
    mock_pipe = AsyncMock()
    mock_pipe.execute.return_value = [0, 1, 5, True]
    mock_redis.pipeline.return_value = mock_pipe

    mock_manager = MagicMock()
    mock_manager.get_client.return_value = mock_redis

    limiter = RateLimiter(route_class="chat", limit_per_minute=10, redis_manager=mock_manager)
    mock_request = MagicMock()

    res = await limiter.check_rate_limit(mock_request, tenant_id="t1", user_id="u1")
    assert res is True


@pytest.mark.asyncio
async def test_rate_limiter_exceeds_limit_raises_429():
    """Verify requests exceeding the token limit raise HTTP 429 Too Many Requests."""
    mock_redis = MagicMock()
    mock_pipe = AsyncMock()
    mock_pipe.execute.return_value = [0, 1, 15, True]
    mock_redis.pipeline.return_value = mock_pipe

    mock_manager = MagicMock()
    mock_manager.get_client.return_value = mock_redis

    limiter = RateLimiter(route_class="chat", limit_per_minute=10, redis_manager=mock_manager)
    mock_request = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await limiter.check_rate_limit(mock_request, tenant_id="t1", user_id="u1")

    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in exc_info.value.detail
    assert exc_info.value.headers["Retry-After"] == "60"
