"""
Health & Dynamic Readiness Check Routes (Section 5, Section 7 & Section 11)
"""

import asyncio
import os
import uuid
import asyncpg
import httpx
from fastapi import APIRouter, Request, Response
from redis.asyncio import Redis

from backend.app.generation.providers.factory import get_llm_provider

router = APIRouter(tags=["Health"])


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid.uuid4()))


@router.get("/health")
async def health_check(request: Request):
    """Process liveness check — confirms FastAPI process is alive (0 external I/O)."""
    return {
        "data": {"status": "ok"},
        "error": None,
        "request_id": _get_request_id(request)
    }


async def _check_postgres() -> bool:
    try:
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        user = os.getenv("POSTGRES_USER", "ekp_user")
        password = os.getenv("POSTGRES_PASSWORD", "ekp_password")
        database = os.getenv("POSTGRES_DB", "ekp_db")

        ssl_env = os.getenv("POSTGRES_SSL")
        ssl_val = ssl_env if ssl_env is not None else ("require" if host not in ("localhost", "127.0.0.1") else None)

        kwargs = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database
        }
        if ssl_val:
            kwargs["ssl"] = ssl_val

        conn = await asyncio.wait_for(
            asyncpg.connect(**kwargs),
            timeout=3.0
        )
        try:
            val = await asyncio.wait_for(conn.fetchval("SELECT 1;"), timeout=3.0)
            return val == 1
        finally:
            await conn.close()
    except Exception:
        return False


async def _check_redis() -> bool:
    try:
        host = os.getenv("REDIS_HOST", "localhost")
        port = os.getenv("REDIS_PORT", "6379")
        password = os.getenv("REDIS_PASSWORD", None)

        url = f"redis://{host}:{port}"
        redis_client = Redis.from_url(url, password=password, socket_timeout=3.0)
        try:
            res = await redis_client.ping()
            return res is True
        finally:
            await redis_client.aclose()
    except Exception:
        return False


async def _check_minio() -> bool:
    try:
        endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        url = f"{endpoint.rstrip('/')}/minio/health/live"
        async with httpx.AsyncClient(timeout=3.0) as client:
            res = await client.get(url)
            return res.status_code == 200
    except Exception:
        return False


async def _check_llm_provider() -> bool:
    try:
        provider = get_llm_provider()
        return await provider.check_readiness()
    except Exception:
        return False


@router.get("/ready")
async def readiness_check(request: Request, response: Response):
    """Dynamic readiness check verifying PostgreSQL, Redis, MinIO, and LLM Provider."""
    req_id = _get_request_id(request)

    pg_ok, redis_ok, minio_ok, llm_ok = await asyncio.gather(
        _check_postgres(),
        _check_redis(),
        _check_minio(),
        _check_llm_provider()
    )

    checks = {
        "postgres": "ok" if pg_ok else "failed",
        "redis": "ok" if redis_ok else "failed",
        "minio": "ok" if minio_ok else "failed",
        "llm_provider": "ok" if llm_ok else "failed"
    }

    all_ready = pg_ok and redis_ok and minio_ok and llm_ok

    if not all_ready:
        response.status_code = 503

    return {
        "data": {
            "status": "ready" if all_ready else "not_ready",
            "checks": checks
        },
        "error": None if all_ready else {"code": "SERVICE_UNAVAILABLE", "message": "One or more dependencies are unavailable"},
        "request_id": req_id
    }
