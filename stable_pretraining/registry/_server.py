#!/usr/bin/env python
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone registry server — fast startup, no torch/lightning imports.

This module deliberately avoids importing anything from
``stable_pretraining`` to keep startup under 1 second.  The SQLite
backend is inlined here (same logic as ``_local.py``).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Inline SQLite backend (avoids importing _local.py → stable_pretraining → torch)
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    status          TEXT DEFAULT 'running',
    created_at      REAL,
    updated_at      REAL,
    run_dir         TEXT,
    checkpoint_path TEXT,
    config          TEXT DEFAULT '{}',
    hparams         TEXT DEFAULT '{}',
    summary         TEXT DEFAULT '{}',
    tags            TEXT DEFAULT '[]',
    notes           TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
"""


class _ServerDB:
    """Minimal thread-safe SQLite wrapper for the server process."""

    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path).resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._check_and_repair()

    def _check_and_repair(self) -> None:
        """Verify DB integrity on startup.

        If corrupt (e.g. stale NFS WAL), remove the file and let SQLite
        recreate it.
        """
        p = Path(self.db_path)
        if not p.exists():
            return
        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            conn.execute("PRAGMA integrity_check")
            conn.execute("SELECT count(*) FROM runs")
            conn.close()
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
            import sys

            print(
                f"[spt-registry] DB corrupt ({exc}), removing and starting fresh.",
                file=sys.stderr,
                flush=True,
            )
            for suffix in ("", "-wal", "-shm"):
                try:
                    Path(self.db_path + suffix).unlink()
                except FileNotFoundError:
                    pass

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.row_factory = sqlite3.Row
            conn.executescript(_SCHEMA_SQL)
            self._local.conn = conn
        return conn

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        conn = self._conn()
        delay = 0.1
        for attempt in range(10):
            try:
                cur = conn.execute(sql, params)
                conn.commit()
                return cur
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() and attempt < 9:
                    time.sleep(delay)
                    delay = min(delay * 2, 5.0)
                else:
                    raise
        raise sqlite3.OperationalError("max retries")  # pragma: no cover

    def insert_run(
        self,
        run_id,
        *,
        status="running",
        run_dir=None,
        config=None,
        hparams=None,
        tags=None,
        notes=None,
    ):
        now = time.time()
        self._exec(
            "INSERT INTO runs "
            "(run_id,status,created_at,updated_at,run_dir,config,hparams,summary,tags,notes) "
            "VALUES (?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(run_id) DO UPDATE SET "
            "status=excluded.status,updated_at=excluded.updated_at,"
            "run_dir=excluded.run_dir,config=excluded.config,"
            "hparams=excluded.hparams,summary=excluded.summary,"
            "tags=excluded.tags,notes=excluded.notes",
            (
                run_id,
                status,
                now,
                now,
                run_dir,
                json.dumps(config or {}),
                json.dumps(hparams or {}),
                json.dumps({}),
                json.dumps(tags or []),
                notes or "",
            ),
        )

    def update_run(self, run_id: str, **fields):
        if not fields:
            return
        json_dict = {"config", "hparams", "summary"}
        json_list = {"tags"}
        proc = {}
        for k, v in fields.items():
            if k in json_dict and isinstance(v, dict):
                proc[k] = json.dumps(v)
            elif k in json_list and isinstance(v, list):
                proc[k] = json.dumps(v)
            else:
                proc[k] = v
        proc["updated_at"] = time.time()
        clause = ", ".join(f"{k} = ?" for k in proc)
        vals = tuple(proc.values()) + (run_id,)
        self._exec(f"UPDATE runs SET {clause} WHERE run_id = ?", vals)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = (
            self._conn()
            .execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            .fetchone()
        )
        return _to_dict(row) if row else None

    def query_runs(
        self, *, status=None, tag=None, sort_by=None, descending=True, limit=None
    ) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if tag:
            clauses.append("tags LIKE ?")
            params.append(f'%"{tag}"%')
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        order = ""
        if sort_by:
            if sort_by.startswith("summary."):
                col = f"json_extract(summary, '$.{sort_by[8:]}')"
            elif sort_by.startswith("hparams."):
                col = f"json_extract(hparams, '$.{sort_by[8:]}')"
            else:
                col = sort_by
            order = f" ORDER BY {col} {'DESC' if descending else 'ASC'}"
        lim = f" LIMIT {int(limit)}" if limit else ""
        rows = (
            self._conn()
            .execute(f"SELECT * FROM runs{where}{order}{lim}", tuple(params))
            .fetchall()
        )
        return [_to_dict(r) for r in rows]

    def count(self) -> int:
        return self._conn().execute("SELECT COUNT(*) FROM runs").fetchone()[0]


def _to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    for col in ("config", "hparams", "summary"):
        if isinstance(d.get(col), str):
            try:
                d[col] = json.loads(d[col])
            except (json.JSONDecodeError, TypeError):
                d[col] = {}
    if isinstance(d.get("tags"), str):
        try:
            d["tags"] = json.loads(d["tags"])
        except (json.JSONDecodeError, TypeError):
            d["tags"] = []
    return d


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def create_app(db_path: str) -> "FastAPI":  # noqa: F821
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel as _BaseModel

    # Models must be module-level for FastAPI introspection.  We define
    # them here to keep the deferred import, then hoist to module scope.
    global RunCreate, RunUpdate

    class RunCreate(_BaseModel):
        run_id: str
        status: str = "running"
        run_dir: Optional[str] = None
        config: Dict[str, Any] = {}
        hparams: Dict[str, Any] = {}
        tags: List[str] = []
        notes: str = ""

    class RunUpdate(_BaseModel):
        status: Optional[str] = None
        run_dir: Optional[str] = None
        checkpoint_path: Optional[str] = None
        config: Optional[Dict[str, Any]] = None
        hparams: Optional[Dict[str, Any]] = None
        summary: Optional[Dict[str, Any]] = None
        tags: Optional[List[str]] = None
        notes: Optional[str] = None

    app = FastAPI(title="spt-registry", version="1.0.0")
    db = _ServerDB(db_path)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "db_path": db.db_path}

    @app.post("/api/runs", status_code=201)
    def create_run(run: RunCreate):
        try:
            db.insert_run(
                run.run_id,
                status=run.status,
                run_dir=run.run_dir,
                config=run.config,
                hparams=run.hparams,
                tags=run.tags,
                notes=run.notes,
            )
        except Exception as exc:
            import traceback

            detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            raise HTTPException(500, detail=detail)
        return {"run_id": run.run_id}

    @app.patch("/api/runs/{run_id:path}")
    def update_run(run_id: str, update: RunUpdate):
        fields = {k: v for k, v in update.model_dump().items() if v is not None}
        if not fields:
            raise HTTPException(400, "No fields to update")
        try:
            db.update_run(run_id, **fields)
        except Exception as exc:
            import traceback

            detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            raise HTTPException(500, detail=detail)
        return {"run_id": run_id}

    @app.get("/api/runs/count")
    def count_runs():
        return {"count": db.count()}

    @app.get("/api/runs/{run_id:path}")
    def get_run(run_id: str):
        run = db.get_run(run_id)
        if run is None:
            raise HTTPException(404, f"Run '{run_id}' not found")
        return run

    @app.get("/api/runs")
    def list_runs(
        status: Optional[str] = None,
        tag: Optional[str] = None,
        sort_by: Optional[str] = None,
        descending: bool = True,
        limit: Optional[int] = None,
    ):
        return db.query_runs(
            status=status, tag=tag, sort_by=sort_by, descending=descending, limit=limit
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        create_app(args.db),
        host=args.host,
        port=args.port,
        workers=1,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
