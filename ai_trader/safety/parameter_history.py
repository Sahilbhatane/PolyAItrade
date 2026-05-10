"""Append-only SQLite log for parameter / RL proposals."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


class ParameterHistoryStore:
    def __init__(self, db_path: str | Path = "ai_trader_parameter_history.db"):
        self._path = Path(db_path)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS params (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT DEFAULT CURRENT_TIMESTAMP,
                payload TEXT NOT NULL,
                hash TEXT NOT NULL
            )"""
        )
        self._conn.commit()

    def append(self, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True)
        h = hashlib.sha256(raw.encode()).hexdigest()[:16]
        self._conn.execute("INSERT INTO params (payload, hash) VALUES (?, ?)", (raw, h))
        self._conn.commit()
        return h

    def close(self) -> None:
        self._conn.close()
