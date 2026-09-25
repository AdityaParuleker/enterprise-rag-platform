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
    """Manager for async Redis connections with graceful cloud degradation."""

    def __init__(self, host: Optional[str] = None, port: Optional[str] = None, password: Optional[str] = None):
        raw_host = host or os.getenv("REDIS_HOST")
        is_cloud = bool(
            os.getenv("RENDER")
            or os.getenv("VERCEL")
            or os.getenv("ENVIRONMENT") == "production"
            or os.getenv("ENV") == "production"
        )

        if is_cloud and (not raw_host or raw_host in ("localhost", "127.0.0.1")):
            logger.warning(
                "Redis not configured for cloud deployment (REDIS_HOST is unset or localhost). "
                "Redis features (rate-limiting, turn caching) will degrade gracefully."
            )
            self.host = None
            self.port = None
            self.password = None
            self.is_available = False
        else:
            self.host = raw_host or "localhost"
            self.port = port or os.getenv("REDIS_PORT", "6379")
            self.password = password or os.getenv("REDIS_PASSWORD", None)
            self.is_available = True

        self._client: Optional[Redis] = None

    def get_client(self) -> Optional[Redis]:
        if not self.is_available or not self.host:
            return None
        if self._client is None:
            url = f"redis://{self.host}:{self.port}"
            self._client = Redis.from_url(url, password=self.password, socket_timeout=3.0, decode_responses=True)
        return self._client

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

