"""SQLite によるジョブ / 実験の記録。

分散基盤は使わない（1台構成）。WAL モードにして、API プロセスからの読みと
スケジューラからの書きが競合しないようにする。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from . import config

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    status        TEXT NOT NULL,            -- queued|running|completed|failed|canceled
    stage         TEXT,
    progress      INTEGER NOT NULL DEFAULT 0,
    gpu           INTEGER,
    style         TEXT NOT NULL,
    quality       TEXT NOT NULL,
    output_format TEXT NOT NULL DEFAULT 'vrm',
    params        TEXT NOT NULL DEFAULT '{}',
    input_path    TEXT,
    input_sha256  TEXT,
    error         TEXT,
    created_at    REAL NOT NULL,
    started_at    REAL,
    completed_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id  TEXT PRIMARY KEY,
    job_id        TEXT NOT NULL,
    model         TEXT NOT NULL,
    model_version TEXT,
    seed          INTEGER,
    parameters    TEXT NOT NULL DEFAULT '{}',
    gpu           INTEGER,
    status        TEXT NOT NULL,            -- ok|failed
    error         TEXT,
    output_path   TEXT,
    polygon_count INTEGER,
    texture_res   INTEGER,
    duration_sec  REAL,
    vram_peak_gb  REAL,
    created_at    REAL NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
);
CREATE INDEX IF NOT EXISTS idx_cand_job ON candidates(job_id);

CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id    TEXT NOT NULL,
    event     TEXT NOT NULL,
    payload   TEXT NOT NULL DEFAULT '{}',
    at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id, id);
"""


def db_path() -> Path:
    return config.data_dir() / "jobs.db"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init() -> None:
    with _LOCK, connect() as conn:
        conn.executescript(SCHEMA)


def new_job_id() -> str:
    # 短くて人が読める ID（Web UI で "Job #A81F" のように出す）
    return uuid.uuid4().hex[:12]


def create_job(
    *, style: str, quality: str, output_format: str, params: dict[str, Any],
    input_path: str, input_sha256: str, job_id: str | None = None,
) -> str:
    job_id = job_id or new_job_id()
    with _LOCK, connect() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id,status,stage,progress,style,quality,output_format,"
            "params,input_path,input_sha256,created_at) "
            "VALUES (?,'queued',NULL,0,?,?,?,?,?,?,?)",
            (job_id, style, quality, output_format, json.dumps(params),
             input_path, input_sha256, time.time()),
        )
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["params"] = json.loads(d.get("params") or "{}")
    return d


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["params"] = json.loads(d.get("params") or "{}")
        out.append(d)
    return out


def claim_next_queued() -> dict[str, Any] | None:
    """キュー先頭を running にして返す（原子的に1件だけ取る）。"""
    with _LOCK, connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE jobs SET status='running', started_at=?, progress=1 WHERE job_id=?",
            (time.time(), row["job_id"]),
        )
    return get_job(row["job_id"])


def update_job(job_id: str, **fields: Any) -> None:
    """指定されたカラムだけ更新する。値が None のフィールドは「変更しない」意味。

    None を素通しすると progress のような NOT NULL 列で IntegrityError になり、
    UPDATE 全体がロールバックされる。実際それで failed 状態の書き込みが失われ、
    ジョブが永久に running のままになる不具合が出た。
    """
    fields = {k: v for k, v in fields.items() if v is not None}
    if not fields:
        return
    if "params" in fields and isinstance(fields["params"], dict):
        fields["params"] = json.dumps(fields["params"])
    cols = ",".join(f"{k}=?" for k in fields)
    with _LOCK, connect() as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE job_id=?",
                     (*fields.values(), job_id))


def finish_job(job_id: str, status: str, error: str | None = None) -> None:
    update_job(job_id, status=status, error=error, completed_at=time.time(),
               progress=100 if status == "completed" else None)


def add_candidate(**f: Any) -> str:
    cid = f.get("candidate_id") or uuid.uuid4().hex[:12]
    f["candidate_id"] = cid
    f.setdefault("created_at", time.time())
    f["parameters"] = json.dumps(f.get("parameters") or {})
    cols = ",".join(f.keys())
    marks = ",".join("?" for _ in f)
    with _LOCK, connect() as conn:
        conn.execute(f"INSERT INTO candidates ({cols}) VALUES ({marks})",
                     tuple(f.values()))
    return cid


def list_candidates(job_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM candidates WHERE job_id=? ORDER BY created_at", (job_id,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["parameters"] = json.loads(d.get("parameters") or "{}")
        out.append(d)
    return out


def add_event(job_id: str, event: str, payload: dict[str, Any] | None = None) -> None:
    with _LOCK, connect() as conn:
        conn.execute(
            "INSERT INTO events (job_id,event,payload,at) VALUES (?,?,?,?)",
            (job_id, event, json.dumps(payload or {}), time.time()),
        )


def list_events(job_id: str, after_id: int = 0) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE job_id=? AND id>? ORDER BY id", (job_id, after_id)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["payload"] = json.loads(d.get("payload") or "{}")
        out.append(d)
    return out


def recover_stale_running() -> int:
    """起動時、前回のクラッシュで running のまま残ったジョブを failed にする。"""
    with _LOCK, connect() as conn:
        cur = conn.execute(
            "UPDATE jobs SET status='failed', error='orchestrator restarted', "
            "completed_at=? WHERE status='running'", (time.time(),)
        )
        return cur.rowcount
