"""
Application configuration via Pydantic BaseSettings.
All secrets loaded from environment variables or .env file.
"""
from __future__ import annotations

import secrets
from functools import lru_cache
from typing import Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Security
    SECRET_KEY: str = secrets.token_hex(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    BCRYPT_ROUNDS: int = 12

    # Database / Cache
    DB_URL: str = "sqlite+aiosqlite:///./ghost_sigint.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Rate limiting / connections
    MAX_CONNECTIONS_PER_USER: int = 5
    RATE_LIMIT_PER_MINUTE: int = 100
    LOGIN_LOCKOUT_ATTEMPTS: int = 5
    LOGIN_LOCKOUT_MINUTES: int = 15

    # RF / DSP
    FS: float = 1e6          # Sample rate (Hz)
    C: float = 3e8           # Speed of light
    TARGET_POWER: float = -20.0  # dBFS

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8080"]

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    @field_validator("SECRET_KEY")
    @classmethod
    def secret_key_entropy(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters")
        return v

    @model_validator(mode="after")
    def production_checks(self) -> "Settings":
        if self.ENVIRONMENT == "production":
            if self.DEBUG:
                raise ValueError("DEBUG must be False in production")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
