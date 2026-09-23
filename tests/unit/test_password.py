"""
Phase 2 Unit Tests: Password Hashing & Verification
"""

import pytest
from backend.app.auth.password import hash_password, verify_password


def test_password_hashing_and_verification():
    raw_password = "SecurePassword123!"
    hashed = hash_password(raw_password)

    assert hashed != raw_password
    assert hashed.startswith("$2b$")
    assert verify_password(raw_password, hashed) is True
    assert verify_password("WrongPassword", hashed) is False


def test_password_empty_input_handling():
    with pytest.raises(ValueError):
        hash_password("")

    assert verify_password("", "$2b$12$somehash") is False
    assert verify_password("pass", "") is False


def test_repeated_hashing_produces_distinct_salts():
    password = "SamePassword"
    hash1 = hash_password(password)
    hash2 = hash_password(password)

    assert hash1 != hash2
    assert verify_password(password, hash1) is True
    assert verify_password(password, hash2) is True
