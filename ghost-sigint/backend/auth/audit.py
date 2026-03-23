"""
Append-only audit log stored in SQLite.
Sensitive fields (passwords, tokens, IQ data) are NEVER logged.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Optional

_DB_PATH: Path = Path("ghost_sigint_audit.db")
_lock = Lock()

# Tokens/passwords/IQ field names that must never appear in logs
_FORBIDDEN_FIELDS = frozenset({
    "password", "token", "secret", "iq_data", "raw_signal",
    "access_token", "refresh_token",
})


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_ns  INTEGER NOT NULL,
            user_id       TEXT    NOT NULL,
            role          TEXT    NOT NULL,
            action        TEXT    NOT NULL,
            endpoint      TEXT    NOT NULL,
            result        TEXT    NOT NULL,
            ip            TEXT,
            user_agent    TEXT
        )
    """)
    # Deny DELETE and UPDATE at SQL level by not exposing those operations.
    conn.commit()
    return conn


_conn: sqlite3.Connection | None = None


def _ensure_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = _get_connection()
    return _conn


def log_action(
    user_id: str,
    role: str,
    action: str,
    endpoint: str,
    result: str,
    ip: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> None:
    """
    Append an audit record. Sensitive fields are silently redacted.
    DELETE on the underlying table is intentionally not exposed.
    """
    # Redact any forbidden substring in action/endpoint/result
    for field in _FORBIDDEN_FIELDS:
        action = _redact(action, field)
        endpoint = _redact(endpoint, field)
        result = _redact(result, field)

    timestamp_ns = time.time_ns()

    with _lock:
        conn = _ensure_conn()
        conn.execute(
            """
            INSERT INTO audit_log
                (timestamp_ns, user_id, role, action, endpoint, result, ip, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp_ns, user_id, role, action, endpoint, result, ip, user_agent),
        )
        conn.commit()


def _redact(text: str, keyword: str) -> str:
    lower = text.lower()
    if keyword in lower:
        return "[REDACTED]"
    return text


def get_recent_logs(limit: int = 100) -> list[dict]:
    """Return up to *limit* most-recent audit records (read-only query)."""
    with _lock:
        conn = _ensure_conn()
        cursor = conn.execute(
            """
            SELECT id, timestamp_ns, user_id, role, action, endpoint, result, ip, user_agent
            FROM audit_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]
