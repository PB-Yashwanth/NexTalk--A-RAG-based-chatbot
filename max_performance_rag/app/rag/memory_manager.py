# app/rag/memory_manager.py

from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from typing import Optional, List, Dict

from app.config import MEMORY_DB_PATH, MAX_MEMORY_MESSAGES


def _connect() -> sqlite3.Connection:
    MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(MEMORY_DB_PATH, check_same_thread=False)


def init_db() -> None:
    with _connect() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id)")
        con.commit()


def add_message(user_id: str, role: str, content: str) -> None:
    now = datetime.now(timezone.utc).isoformat()

    with _connect() as con:
        con.execute(
            "INSERT INTO messages(user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, now),
        )

        # Keep only last N messages per user
        con.execute(
            """
            DELETE FROM messages
            WHERE user_id = ?
              AND id NOT IN (
                  SELECT id FROM messages
                  WHERE user_id = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (user_id, user_id, MAX_MEMORY_MESSAGES),
        )
        con.commit()


def add_exchange(user_id: str, user_message: str, assistant_message: str) -> None:
    add_message(user_id, "user", user_message)
    add_message(user_id, "assistant", assistant_message)


def get_chat_history(user_id: str, limit: Optional[int] = None) -> List[Dict]:
    limit = limit or MAX_MEMORY_MESSAGES
    with _connect() as con:
        rows = con.execute(
            """
            SELECT role, content
            FROM messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    rows.reverse()  # chronological order
    return [{"role": r[0], "content": r[1]} for r in rows]


def clear_history(user_id: str) -> None:
    with _connect() as con:
        con.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        con.commit()