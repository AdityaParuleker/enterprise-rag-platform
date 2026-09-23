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

        _pool = await asyncpg.create_pool(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            min_size=1,
            max_size=10,
            timeout=5.0
        )
    return _pool


async def get_db_connection() -> asyncpg.Connection:
    """Acquire a single connection to PostgreSQL."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    user = os.getenv("POSTGRES_USER", "ekp_user")
    password = os.getenv("POSTGRES_PASSWORD", "ekp_password")
    database = os.getenv("POSTGRES_DB", "ekp_db")

    return await asyncpg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        timeout=5.0
    )
