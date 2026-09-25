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
    """Manager for async Redis connections with graceful cloud degradation and REDIS_URL/TLS support."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        password: Optional[str] = None,
    ):
        env_url = redis_url or os.getenv("REDIS_URL")
        raw_host = host or os.getenv("REDIS_HOST")

        is_cloud = bool(
            os.getenv("RENDER")
            or os.getenv("VERCEL")
            or os.getenv("ENVIRONMENT") == "production"
            or os.getenv("ENV") == "production"
        )

        if env_url:
            self.redis_url = env_url
            self.is_available = True
        elif is_cloud and (not raw_host or raw_host in ("localhost", "127.0.0.1")):
            logger.warning(
                "Redis not configured for cloud deployment (REDIS_URL and REDIS_HOST are unset or localhost). "
                "Redis features (rate-limiting, turn caching) will degrade gracefully."
            )
            self.redis_url = None
            self.is_available = False
        else:
            host_val = raw_host or "localhost"
            port_val = port or os.getenv("REDIS_PORT", "6379")
            pass_val = password or os.getenv("REDIS_PASSWORD", "")
            auth_str = f":{pass_val}@" if pass_val else ""
            self.redis_url = f"redis://{auth_str}{host_val}:{port_val}"
            self.is_available = True

        self._client: Optional[Redis] = None

    def get_client(self) -> Optional[Redis]:
        if not self.is_available or not self.redis_url:
            return None
        if self._client is None:
            self._client = Redis.from_url(
                self.redis_url,
                socket_timeout=3.0,
                decode_responses=True,
            )
        return self._client

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None


