"""
JWT Access Token Module (Section 6.1)
Handles issuance and signature verification of short-lived JWT access tokens.
Enforces non-empty JWT_SECRET environment variable.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import jwt
from fastapi import HTTPException, status


def get_jwt_secret() -> str:
    """Read JWT_SECRET from environment. Raises ValueError if missing or empty."""
    secret = os.getenv("JWT_SECRET")
    if not secret or not secret.strip():
        raise ValueError("JWT_SECRET environment variable is missing or empty.")
    return secret.strip()


def get_jwt_algorithm() -> str:
    return os.getenv("JWT_ALGORITHM", "HS256")


def get_access_token_expire_minutes() -> int:
    try:
        return int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    except ValueError:
        return 30



def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT access token containing required claims: sub, tenant_id, email, exp, iat, jti."""
    secret = get_jwt_secret()
    algorithm = get_jwt_algorithm()
    
    to_encode = data.copy()

    # Enforce string types for sub and tenant_id
    if "sub" in to_encode:
        to_encode["sub"] = str(to_encode["sub"])
    if "tenant_id" in to_encode:
        to_encode["tenant_id"] = str(to_encode["tenant_id"])

    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=get_access_token_expire_minutes())

    to_encode.update({
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "jti": str(uuid.uuid4())
    })

    return jwt.encode(to_encode, secret, algorithm=algorithm)


def verify_token(token: str) -> Dict:
    """Verify JWT signature, algorithm, expiration, and required claims (sub, tenant_id)."""
    secret = get_jwt_secret()
    algorithm = get_jwt_algorithm()

    try:
        payload = jwt.decode(token, secret, algorithms=[algorithm])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Validate required claims
    if "sub" not in payload or not payload["sub"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload missing required 'sub' claim",
            headers={"WWW-Authenticate": "Bearer"}
        )
    if "tenant_id" not in payload or not payload["tenant_id"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload missing required 'tenant_id' claim",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return payload
