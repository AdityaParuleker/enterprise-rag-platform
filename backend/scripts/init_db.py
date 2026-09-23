"""
Database Initialization & Migration Runner
Applies migrations 001-004 to PostgreSQL (e.g., Neon.tech) and seeds default admin user.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

import asyncpg
from backend.app.auth.password import hash_password

MIGRATIONS_DIR = backend_dir / "backend" / "app" / "db" / "migrations"


async def run_migrations():
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
        "timeout": 10.0
    }
    if ssl_val:
        kwargs["ssl"] = ssl_val

    print(f" Connecting to PostgreSQL host: {host}, db: {database} (ssl={ssl_val})...")
    conn = await asyncpg.connect(**kwargs)

    try:
        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        for sql_file in migration_files:
            print(f" Executing migration: {sql_file.name}...")
            with open(sql_file, "r", encoding="utf-8") as f:
                sql_content = f.read()
            await conn.execute(sql_content)
            print(f" Migration {sql_file.name} applied successfully.")

        # Seed default admin user if not exists
        admin_email = "demo_admin@enterprise.com"
        admin_pass = "password123"

        existing_user = await conn.fetchrow("SELECT id FROM users WHERE email = $1", admin_email)
        if not existing_user:
            print(f" Seeding default admin user ({admin_email})...")
            # 1. Create or get Demo Tenant
            tenant = await conn.fetchrow("SELECT id FROM tenants WHERE name = 'Enterprise Demo Tenant'")
            if not tenant:
                tenant = await conn.fetchrow(
                    "INSERT INTO tenants (name, plan, is_active) VALUES ('Enterprise Demo Tenant', 'enterprise', true) RETURNING id"
                )
            tenant_id = tenant["id"]

            # 2. Insert admin user
            pwd_hash = hash_password(admin_pass)
            user_row = await conn.fetchrow(
                """
                INSERT INTO users (tenant_id, email, password_hash, is_active)
                VALUES ($1, $2, $3, true)
                RETURNING id
                """,
                tenant_id,
                admin_email,
                pwd_hash
            )
            user_id = user_row["id"]

            # 3. Assign TENANT_ADMIN role
            role_row = await conn.fetchrow("SELECT id FROM roles WHERE name = 'TENANT_ADMIN'")
            if role_row:
                await conn.execute(
                    """
                    INSERT INTO user_roles (user_id, role_id, tenant_id)
                    VALUES ($1, $2, $3)
                    ON CONFLICT DO NOTHING
                    """,
                    user_id,
                    role_row["id"],
                    tenant_id
                )
            print(f" Default admin created!\n   Email: {admin_email}\n   Password: {admin_pass}\n   Tenant ID: {tenant_id}")
        else:
            print(f" Admin user {admin_email} already exists.")

    finally:
        await conn.close()
        print(" Database initialization complete.")


if __name__ == "__main__":
    asyncio.run(run_migrations())
