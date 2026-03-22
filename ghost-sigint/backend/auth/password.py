"""
Password hashing and verification using bcrypt.
Plain-text passwords are NEVER stored or logged.
"""
from __future__ import annotations

import bcrypt

from backend.config import get_settings

settings = get_settings()


def hash_password(plain: str) -> str:
    """Return bcrypt hash of *plain* password. Never store or log *plain*."""
    rounds = settings.BCRYPT_ROUNDS
    salt = bcrypt.gensalt(rounds=rounds)
    hashed = bcrypt.hashpw(plain.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Return True iff *plain* matches *hashed* bcrypt digest."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False
