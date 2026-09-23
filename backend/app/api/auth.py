"""
Authentication API Endpoints (Section 5, Section 6.1)
Implements Register, Login, Refresh (with rotation), Logout, and Password Reset.
"""

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, EmailStr

from backend.app.auth.jwt import create_access_token, get_access_token_expire_minutes
from backend.app.auth.password import hash_password, verify_password
from backend.app.db.connection import get_db_connection

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid.uuid4()))


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


# Request Schemas
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    tenant_name: Optional[str] = None
    tenant_id: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: Optional[str] = None


class PasswordResetRequest(BaseModel):
    email: EmailStr
    old_password: str
    new_password: str


@router.post("/register")
async def register(body: RegisterRequest, request: Request):
    """Register a new user and tenant atomically within a database transaction."""
    req_id = _get_request_id(request)
    
    if not body.password or len(body.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters"
        )

    conn = await get_db_connection()
    try:
        async with conn.transaction():
            # Check if user already exists
            existing = await conn.fetchrow("SELECT id FROM users WHERE email = $1", body.email)
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="User with this email already exists"
                )

            tenant_id = None
            is_new_tenant = False

            if body.tenant_id:
                try:
                    tenant_uuid = UUID(body.tenant_id)
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid tenant_id format")

                tenant_row = await conn.fetchrow("SELECT id FROM tenants WHERE id = $1 AND is_active = true", tenant_uuid)
                if not tenant_row:
                    raise HTTPException(status_code=404, detail="Tenant not found or inactive")
                tenant_id = tenant_row["id"]
            else:
                tenant_name = body.tenant_name or f"{body.email.split('@')[0]}'s Organization"
                tenant_row = await conn.fetchrow(
                    "INSERT INTO tenants (name, plan, is_active) VALUES ($1, 'free', true) RETURNING id",
                    tenant_name
                )
                tenant_id = tenant_row["id"]
                is_new_tenant = True

            # Hash password
            pwd_hash = hash_password(body.password)

            # Insert User
            user_row = await conn.fetchrow(
                """
                INSERT INTO users (tenant_id, email, password_hash, is_active)
                VALUES ($1, $2, $3, true)
                RETURNING id, email, tenant_id, created_at
                """,
                tenant_id,
                body.email,
                pwd_hash
            )
            user_id = user_row["id"]

            # Assign Role: TENANT_ADMIN for new tenant creator, USER for existing tenant additions
            role_name = "TENANT_ADMIN" if is_new_tenant else "USER"
            role_row = await conn.fetchrow("SELECT id FROM roles WHERE name = $1", role_name)
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

            return {
                "data": {
                    "user_id": str(user_id),
                    "tenant_id": str(tenant_id),
                    "email": user_row["email"],
                    "role": role_name
                },
                "error": None,
                "request_id": req_id
            }
    finally:
        await conn.close()


@router.post("/login")
async def login(body: LoginRequest, request: Request):
    """Authenticate user credentials and return access_token + refresh_token."""
    req_id = _get_request_id(request)

    conn = await get_db_connection()
    try:
        user_row = await conn.fetchrow(
            "SELECT id, tenant_id, email, password_hash, is_active FROM users WHERE email = $1",
            body.email
        )

        if not user_row or not user_row["is_active"] or not verify_password(body.password, user_row["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        user_id = user_row["id"]
        tenant_id = user_row["tenant_id"]

        # Update last_login_at
        await conn.execute("UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = $1", user_id)

        # Generate access token
        access_token = create_access_token({
            "sub": str(user_id),
            "tenant_id": str(tenant_id),
            "email": user_row["email"]
        })

        # Generate refresh token & store hash
        raw_refresh_token = secrets.token_urlsafe(48)
        refresh_hash = _hash_token(raw_refresh_token)
        expires_at = datetime.now(timezone.utc) + timedelta(days=7)

        await conn.execute(
            """
            INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
            VALUES ($1, $2, $3)
            """,
            user_id,
            refresh_hash,
            expires_at
        )

        return {
            "data": {
                "access_token": access_token,
                "refresh_token": raw_refresh_token,
                "token_type": "bearer",
                "expires_in_minutes": get_access_token_expire_minutes(),
                "user_id": str(user_id),
                "tenant_id": str(tenant_id)
            },
            "error": None,
            "request_id": req_id
        }
    finally:
        await conn.close()


@router.post("/refresh")
async def refresh_token(body: RefreshRequest, request: Request):
    """Atomic Refresh Token Rotation: Revokes presented token and issues new token pair."""
    req_id = _get_request_id(request)

    if not body.refresh_token:
        raise HTTPException(status_code=400, detail="refresh_token is required")

    present_hash = _hash_token(body.refresh_token)
    conn = await get_db_connection()

    try:
        async with conn.transaction():
            # Select and lock refresh token record
            token_row = await conn.fetchrow(
                """
                SELECT id, user_id, expires_at, revoked_at
                FROM refresh_tokens
                WHERE token_hash = $1
                FOR UPDATE
                """,
                present_hash
            )

            if not token_row:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )

            if token_row["revoked_at"] is not None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Refresh token has been revoked"
                )

            now_utc = datetime.now(timezone.utc)
            token_exp = token_row["expires_at"]
            if token_exp.tzinfo is None:
                token_exp = token_exp.replace(tzinfo=timezone.utc)

            if token_exp < now_utc:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Refresh token has expired"
                )

            # Atomic Revocation of presented token
            await conn.execute(
                "UPDATE refresh_tokens SET revoked_at = CURRENT_TIMESTAMP WHERE id = $1",
                token_row["id"]
            )

            # Fetch user details
            user_row = await conn.fetchrow(
                "SELECT id, tenant_id, email, is_active FROM users WHERE id = $1",
                token_row["user_id"]
            )

            if not user_row or not user_row["is_active"]:
                raise HTTPException(status_code=401, detail="User account is inactive or disabled")

            # Generate NEW Access Token
            new_access_token = create_access_token({
                "sub": str(user_row["id"]),
                "tenant_id": str(user_row["tenant_id"]),
                "email": user_row["email"]
            })

            # Generate NEW Refresh Token & Insert Hash
            new_raw_refresh_token = secrets.token_urlsafe(48)
            new_refresh_hash = _hash_token(new_raw_refresh_token)
            new_expires_at = datetime.now(timezone.utc) + timedelta(days=7)

            await conn.execute(
                """
                INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
                VALUES ($1, $2, $3)
                """,
                user_row["id"],
                new_refresh_hash,
                new_expires_at
            )

            return {
                "data": {
                    "access_token": new_access_token,
                    "refresh_token": new_raw_refresh_token,
                    "token_type": "bearer"
                },
                "error": None,
                "request_id": req_id
            }
    finally:
        await conn.close()


@router.post("/logout")
async def logout(body: LogoutRequest, request: Request):
    """Revoke refresh token."""
    req_id = _get_request_id(request)

    if body and body.refresh_token:
        present_hash = _hash_token(body.refresh_token)
        conn = await get_db_connection()
        try:
            await conn.execute(
                "UPDATE refresh_tokens SET revoked_at = CURRENT_TIMESTAMP WHERE token_hash = $1 AND revoked_at IS NULL",
                present_hash
            )
        finally:
            await conn.close()

    return {
        "data": {"status": "logged_out"},
        "error": None,
        "request_id": req_id
    }


@router.post("/password-reset")
async def password_reset(body: PasswordResetRequest, request: Request):
    """Authenticated credential verification and password update flow."""
    req_id = _get_request_id(request)

    if not body.new_password or len(body.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")

    conn = await get_db_connection()
    try:
        async with conn.transaction():
            user_row = await conn.fetchrow(
                "SELECT id, password_hash FROM users WHERE email = $1",
                body.email
            )

            if not user_row or not verify_password(body.old_password, user_row["password_hash"]):
                raise HTTPException(status_code=401, detail="Invalid current password")

            new_pwd_hash = hash_password(body.new_password)
            user_id = user_row["id"]

            # Update password
            await conn.execute(
                "UPDATE users SET password_hash = $1 WHERE id = $2",
                new_pwd_hash,
                user_id
            )

            # Revoke all active refresh tokens for user
            await conn.execute(
                "UPDATE refresh_tokens SET revoked_at = CURRENT_TIMESTAMP WHERE user_id = $1 AND revoked_at IS NULL",
                user_id
            )

            return {
                "data": {"status": "password_reset_success"},
                "error": None,
                "request_id": req_id
            }
    finally:
        await conn.close()
