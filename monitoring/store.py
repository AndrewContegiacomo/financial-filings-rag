"""
Event log for the RAG application.

Backend is chosen at runtime: Postgres when DATABASE_URL is set (the
docker-compose setup), SQLite otherwise (local development and the
Streamlit Cloud demo, where the filesystem is ephemeral so events live
as long as the container).

The two dialects differ in three ways that matter here: placeholder
syntax (? vs %s), the autoincrement declaration, and how the inserted
row id is returned. Everything else is plain SQL.
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL")
USE_POSTGRES = bool(DATABASE_URL)

DB_PATH = Path(__file__).resolve().parent.parent / "data/monitoring.db"

_SERIAL = "SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"
_PH = "%s" if USE_POSTGRES else "?"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS queries (
    id {_SERIAL},
    ts TEXT NOT NULL,
    question TEXT NOT NULL,
    path TEXT NOT NULL,
    filters TEXT,
    n_filters INTEGER,
    retrieval_ms INTEGER,
    generation_ms INTEGER,
    cold_start INTEGER DEFAULT 0,
    n_chunks INTEGER,
    refused INTEGER DEFAULT 0,
    error TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id {_SERIAL},
    query_id INTEGER NOT NULL,
    ts TEXT NOT NULL,
    rating INTEGER NOT NULL
);
"""


@contextmanager
def _conn():
    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row
        conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        try:
            conn.execute(SCHEMA)
            yield conn
            conn.commit()
        finally:
            conn.close()
    else:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(SCHEMA)
            yield conn
            conn.commit()
        finally:
            conn.close()

def log_query(question: str, path: str, filters: dict | None,
              retrieval_ms: int, generation_ms: int, cold_start: bool,
              n_chunks: int, refused: bool = False,
              error: str | None = None) -> int:
    sql = f"""INSERT INTO queries
              (ts, question, path, filters, n_filters, retrieval_ms,
               generation_ms, cold_start, n_chunks, refused, error)
              VALUES ({', '.join([_PH] * 11)})"""
    params = (
        datetime.now(timezone.utc).isoformat(),
        question, path,
        json.dumps(filters or {}), len(filters or {}),
        retrieval_ms, generation_ms, int(cold_start),
        n_chunks, int(refused), error,
    )
    with _conn() as conn:
        if USE_POSTGRES:
            cur = conn.execute(sql + " RETURNING id", params)
            return cur.fetchone()["id"]
        return conn.execute(sql, params).lastrowid


def log_feedback(query_id: int, rating: int) -> None:
    sql = f"INSERT INTO feedback (query_id, ts, rating) VALUES ({_PH}, {_PH}, {_PH})"
    with _conn() as conn:
        conn.execute(sql, (query_id,
                           datetime.now(timezone.utc).isoformat(), rating))


def fetch_queries() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM queries ORDER BY ts").fetchall()
        return [dict(r) for r in rows]


def fetch_feedback() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM feedback ORDER BY ts").fetchall()
        return [dict(r) for r in rows]
    