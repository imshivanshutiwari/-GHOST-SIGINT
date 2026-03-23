"""Authentication routes: /login /logout /refresh"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request, Response, Depends
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from auth.jwt_handler import create_access_token, create_refresh_token, rotate_refresh_token, verify_token
from auth.password import verify_password, hash_password
from auth.audit import log_action
from auth.rbac import require_role, Role
from constants import LOGIN_LOCKOUT_ATTEMPTS, LOGIN_LOCKOUT_MINUTES

router = APIRouter(prefix="/auth", tags=["auth"])

_USERS: dict = {
    "operator":  {"hashed_password": None, "role": "OPERATOR"},
    "analyst":   {"hashed_password": None, "role": "ANALYST"},
    "commander": {"hashed_password": None, "role": "COMMANDER"},
}

def _init():
    for u, d in _USERS.items():
        if d["hashed_password"] is None:
            d["hashed_password"] = hash_password(u + "_ghost2024!")
_init()

_fails: dict = {}
_locks: dict = {}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


@router.post("/login", response_model=TokenResponse)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    request: Request,
    response: Response,
):
    username = form.username.lower()
    ip = request.client.host if request.client else "unknown"
    ua = request.headers.get("user-agent", "")

    # Check lockout
    if username in _locks:
        if datetime.utcnow() < _locks[username]:
            raise HTTPException(429, "Account locked. Try later.")
        del _locks[username]
        _fails.pop(username, None)

    user = _USERS.get(username)
    if not user or not verify_password(form.password, user["hashed_password"]):
        _fails.setdefault(username, []).append(datetime.utcnow())
        recent = [t for t in _fails[username] if (datetime.utcnow()-t).total_seconds() < 900]
        _fails[username] = recent
        if len(recent) >= LOGIN_LOCKOUT_ATTEMPTS:
            _locks[username] = datetime.utcnow() + timedelta(minutes=LOGIN_LOCKOUT_MINUTES)
        log_action(username, "unknown", "LOGIN_FAIL", "/auth/login", "FAILURE", ip, ua)
        raise HTTPException(401, "AUTHENTICATION FAILED")

    role = user["role"]
    access_token = create_access_token(sub=username, role=role)
    refresh_token = create_refresh_token(sub=username)
    _fails.pop(username, None)

    response.set_cookie("access_token",  access_token,  httponly=True, samesite="strict", secure=False)
    response.set_cookie("refresh_token", refresh_token, httponly=True, samesite="strict", secure=False)
    log_action(username, role, "LOGIN_SUCCESS", "/auth/login", "SUCCESS", ip, ua)
    return TokenResponse(access_token=access_token, role=role)


@router.post("/logout")
async def logout(response: Response, _=Depends(require_role(Role.OPERATOR))):
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")
    return {"message": "Logged out"}


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: Request, response: Response):
    rt = request.cookies.get("refresh_token")
    if not rt:
        raise HTTPException(401, "No refresh token")
    try:
        access, new_rt = rotate_refresh_token(rt)
    except Exception:
        raise HTTPException(401, "Invalid refresh token")
    response.set_cookie("access_token",  access,  httponly=True, samesite="strict", secure=False)
    response.set_cookie("refresh_token", new_rt, httponly=True, samesite="strict", secure=False)
    payload = verify_token(access)
    return TokenResponse(access_token=access, role=payload.get("role", "OPERATOR"))
