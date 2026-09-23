"""
Redis-Backed Sliding Token Bucket Rate Limiter (Section 6.6 & Phase 8 Specifications)
Implements fail-closed token-bucket rate limiting with distinct pre-auth IP keys vs post-auth tenant/user keys.
"""

import time
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status
from backend.app.cache.redis_client import RedisManager

# Default Rate Limits (requests per minute)
DEFAULT_LIMITS = {
    "auth": 10,
    "chat": 60,
    "documents": 30,
    "eval": 20
}


class RateLimiter:
    """Sliding window token-bucket rate limiter backed by Redis. Fails closed (HTTP 503) on Redis outage."""

    def __init__(self, route_class: str = "chat", limit_per_minute: Optional[int] = None, redis_manager: Optional[RedisManager] = None):
        self.route_class = route_class
        self.limit = limit_per_minute or DEFAULT_LIMITS.get(route_class, 60)
        self.redis_manager = redis_manager or RedisManager()

    async def check_rate_limit(
        self,
        request: Request,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        try:
            redis = self.redis_manager.get_client()
            if not redis:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Rate limiter service unavailable."
                )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rate limiter service unavailable."
            )

        client_ip = request.client.host if request.client else "127.0.0.1"

        if tenant_id and user_id:
            key = f"rate_limit:{tenant_id}:{user_id}:{self.route_class}"
        else:
            key = f"rate_limit:preauth:{client_ip}:{self.route_class}"

        try:
            now = time.time()
            window_start = now - 60.0

            # Atomic sliding window in Redis using pipeline / zset
            pipe = redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, 60)
            results = await pipe.execute()

            request_count = results[2]

            if request_count > self.limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded for {self.route_class}. Maximum {self.limit} requests per minute.",
                    headers={"Retry-After": "60"}
                )

            return True
        except HTTPException:
            raise
        except Exception as exc:
            # Redis operational error -> Fail closed per Phase 8 invariant
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Rate limiter error: {str(exc)}"
            )
