# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Thread-safe SQLite backend for the registry.

Used directly as a fallback when the server is unavailable, and by the
server process itself.  The corruption issues that motivated the
client-server refactoring only occur when multiple *nodes* hit the same
``.db`` file over NFS — a single-process server with WAL is safe.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class LocalRegistryDB:
    """Thread-safe SQLite database for run metadata.

    Each thread gets its own connection via ``threading.local()``.
    WAL mode + ``busy_timeout`` handle concurrent readers/writers from
    the server's thread pool.

    Args:
        db_path: Path to the SQLite database file.  Created if it does
            not exist (parent directories must exist).
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path).resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()

    # -- connection management -------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.row_factory = sqlite3.Row
            conn.executescript(_SCHEMA_SQL)
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # -- retry wrapper ---------------------------------------------------------

    def _execute_with_retry(
        self,
        sql: str,
        params: tuple = (),
        *,
        max_retries: int = 10,
    ) -> sqlite3.Cursor:
        conn = self._get_connection()
        delay = 0.1
        for attempt in range(max_retries):
            try:
                cursor = conn.execute(sql, params)
                conn.commit()
                return cursor
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 5.0)
                else:
                    raise
        raise sqlite3.OperationalError("max retries exceeded")  # pragma: no cover

    # -- public API ------------------------------------------------------------

    def insert_run(
        self,
        run_id: str,
        *,
        status: str = "running",
        run_dir: Optional[str] = None,
        config: Optional[dict] = None,
        hparams: Optional[dict] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> None:
        now = time.time()
        self._execute_with_retry(
            "INSERT INTO runs "
            "(run_id, status, created_at, updated_at, "
            " run_dir, config, hparams, summary, tags, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET "
            "  status=excluded.status, updated_at=excluded.updated_at, "
            "  run_dir=excluded.run_dir, config=excluded.config, "
            "  hparams=excluded.hparams, summary=excluded.summary, "
            "  tags=excluded.tags, notes=excluded.notes",
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

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        json_dict_cols = {"config", "hparams", "summary"}
        json_list_cols = {"tags"}
        processed = {}
        for k, v in fields.items():
            if k in json_dict_cols and isinstance(v, dict):
                processed[k] = json.dumps(v)
            elif k in json_list_cols and isinstance(v, list):
                processed[k] = json.dumps(v)
            else:
                processed[k] = v
        processed["updated_at"] = time.time()
        set_clause = ", ".join(f"{k} = ?" for k in processed)
        values = tuple(processed.values()) + (run_id,)
        self._execute_with_retry(
            f"UPDATE runs SET {set_clause} WHERE run_id = ?",
            values,
        )

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return _row_to_dict(row)

    def query_runs(
        self,
        *,
        status: Optional[str] = None,
        tag: Optional[str] = None,
        sort_by: Optional[str] = None,
        descending: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if tag is not None:
            clauses.append("tags LIKE ?")
            params.append(f'%"{tag}"%')
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        order = ""
        if sort_by is not None:
            if sort_by.startswith("summary."):
                key = sort_by[len("summary.") :]
                col_expr = f"json_extract(summary, '$.{key}')"
            elif sort_by.startswith("hparams."):
                key = sort_by[len("hparams.") :]
                col_expr = f"json_extract(hparams, '$.{key}')"
            else:
                col_expr = sort_by
            direction = "DESC" if descending else "ASC"
            order = f" ORDER BY {col_expr} {direction}"

        limit_clause = f" LIMIT {int(limit)}" if limit is not None else ""
        sql = f"SELECT * FROM runs{where}{order}{limit_clause}"
        conn = self._get_connection()
        rows = conn.execute(sql, tuple(params)).fetchall()
        return [_row_to_dict(r) for r in rows]

    def count(self) -> int:
        conn = self._get_connection()
        row = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        return row[0]


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    for col in ("config", "hparams", "summary"):
        if col in d and isinstance(d[col], str):
            try:
                d[col] = json.loads(d[col])
            except (json.JSONDecodeError, TypeError):
                d[col] = {}
    if "tags" in d and isinstance(d["tags"], str):
        try:
            d["tags"] = json.loads(d["tags"])
        except (json.JSONDecodeError, TypeError):
            d["tags"] = []
    return d
