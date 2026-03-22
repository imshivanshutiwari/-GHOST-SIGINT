"""
JWT creation, verification, rotation, and revocation.
Tokens are HS256, stored user_id+role in payload.
Refresh tokens are single-use; used tokens stored in _USED_REFRESH_TOKENS.
"""
from __future__ import annotations

import time
from threading import Lock
from typing import Any

from fastapi import HTTPException, status
from jose import JWTError, jwt

from backend.config import get_settings

settings = get_settings()

_USED_REFRESH_TOKENS: set[str] = set()
_REVOKED_USERS: set[str] = set()
_lock = Lock()


def _unix_now() -> int:
    return int(time.time())


def create_access_token(sub: str, role: str) -> str:
    expire = _unix_now() + settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    payload: dict[str, Any] = {
        "sub": sub,
        "role": role,
        "type": "access",
        "exp": expire,
        "iat": _unix_now(),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(sub: str) -> str:
    expire = _unix_now() + settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400
    payload: dict[str, Any] = {
        "sub": sub,
        "type": "refresh",
        "exp": expire,
        "iat": _unix_now(),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def verify_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    sub: str | None = payload.get("sub")
    if sub is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    with _lock:
        if sub in _REVOKED_USERS:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked")

    return payload


def rotate_refresh_token(old_token: str) -> tuple[str, str]:
    payload = verify_token(old_token)

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not a refresh token")

    with _lock:
        if old_token in _USED_REFRESH_TOKENS:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token already used")
        _USED_REFRESH_TOKENS.add(old_token)

    sub: str = payload["sub"]
    role: str = payload.get("role", "OPERATOR")
    access = create_access_token(sub=sub, role=role)
    refresh = create_refresh_token(sub=sub)
    return access, refresh


def revoke_all_tokens(user_id: str) -> None:
    with _lock:
        _REVOKED_USERS.add(user_id)
