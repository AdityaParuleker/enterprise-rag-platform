"""
Redis Client Manager Module (Section 7.17 & Section 15)
Provides async Redis client instance with connection pooling and graceful error handling.
"""

import os
import logging
from typing import Optional
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class RedisManager:
    """Manager for async Redis connections."""

    def __init__(self, host: Optional[str] = None, port: Optional[str] = None, password: Optional[str] = None):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or os.getenv("REDIS_PORT", "6379")
        self.password = password or os.getenv("REDIS_PASSWORD", None)
        self._client: Optional[Redis] = None

    def get_client(self) -> Redis:
        if self._client is None:
            url = f"redis://{self.host}:{self.port}"
            self._client = Redis.from_url(url, password=self.password, socket_timeout=3.0, decode_responses=True)
        return self._client

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None
