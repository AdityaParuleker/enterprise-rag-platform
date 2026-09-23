"""
Password Hashing & Verification Module (Section 6.1)
Uses OpenBSD Blowfish bcrypt hashing algorithm.
"""

import bcrypt


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt with salt work factor 12."""
    if not password:
        raise ValueError("Password cannot be empty")
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash string."""
    if not plain_password or not hashed_password:
        return False
    try:
        plain_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(plain_bytes, hashed_bytes)
    except Exception:
        return False
