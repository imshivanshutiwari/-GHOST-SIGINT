"""
Role-Based Access Control (RBAC) for GHOST-SIGINT.

Role hierarchy: OPERATOR (0) < ANALYST (1) < COMMANDER (2)

FastAPI dependency require_role(min_role) extracts JWT from
httpOnly cookie or Authorization header and enforces hierarchy.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Annotated

from fastapi import Cookie, Depends, Header, HTTPException, status

from backend.auth.jwt_handler import verify_token


class Role(IntEnum):
    OPERATOR = 0
    ANALYST = 1
    COMMANDER = 2


_ROLE_MAP: dict[str, Role] = {
    "OPERATOR": Role.OPERATOR,
    "ANALYST": Role.ANALYST,
    "COMMANDER": Role.COMMANDER,
}


def _extract_token(
    authorization: Annotated[str | None, Header()] = None,
    access_token: Annotated[str | None, Cookie()] = None,
) -> str:
    if authorization and authorization.startswith("Bearer "):
        return authorization.split(" ", 1)[1]
    if access_token:
        return access_token
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(min_role: Role):
    """
    FastAPI dependency factory.
    Usage: Depends(require_role(Role.ANALYST))
    Returns the verified JWT payload dict.
    """
    def dependency(token: Annotated[str, Depends(_extract_token)]) -> dict:
        payload = verify_token(token)
        role_str: str = payload.get("role", "OPERATOR")
        user_role = _ROLE_MAP.get(role_str, Role.OPERATOR)
        if user_role < min_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role_str}' insufficient; requires '{min_role.name}'",
            )
        return payload
    return dependency
