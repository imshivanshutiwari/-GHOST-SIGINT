from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
from auth.jwt_handler import (create_access_token, create_refresh_token,
                               verify_token, rotate_refresh_token)
from auth.password import hash_password, verify_password


def test_access_token_roundtrip():
    t = create_access_token(sub="u1", role="ANALYST")
    p = verify_token(t)
    assert p["sub"] == "u1" and p["role"] == "ANALYST"

def test_refresh_token_rotation():
    rt = create_refresh_token(sub="u1")
    access, new_rt = rotate_refresh_token(rt)
    assert isinstance(access, str) and new_rt != rt

def test_invalid_token_raises():
    with pytest.raises(Exception):
        verify_token("not.a.valid.token")

def test_bcrypt_verify():
    h = hash_password("SecureP@ss!")
    assert verify_password("SecureP@ss!", h) is True
    assert verify_password("wrong", h) is False

def test_bcrypt_unique_salts():
    assert hash_password("same") != hash_password("same")

def test_rotate_invalidates_old():
    rt = create_refresh_token(sub="u2")
    rotate_refresh_token(rt)
    with pytest.raises(Exception):
        rotate_refresh_token(rt)
