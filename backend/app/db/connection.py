"""
Database Connection Manager (Section 4 & Phase 2)
Provides asyncpg database connection acquisition using configured environment variables.
"""

import os
import asyncpg
from typing import Optional

_pool: Optional[asyncpg.Pool] = None


async def get_db_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = int(os.getenv("POSTGRES_PORT", "5432"))
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
            "database": database,
            "min_size": 1,
            "max_size": 10,
            "timeout": 5.0
        }
        if ssl_val:
            kwargs["ssl"] = ssl_val

        _pool = await asyncpg.create_pool(**kwargs)
    return _pool


async def get_db_connection() -> asyncpg.Connection:
    """Acquire a single connection to PostgreSQL."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
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
        "database": database,
        "timeout": 5.0
    }
    if ssl_val:
        kwargs["ssl"] = ssl_val

    return await asyncpg.connect(**kwargs)
