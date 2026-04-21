# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SQLite cache over sidecar files.

This is a **derived** database: the scanner is the single writer, query
clients open it read-only.  If the DB is lost or corrupted, delete it
and run ``spt registry scan --full`` — nothing in it is authoritative.

Design notes:
    * Single writer ⇒ no retry loop, no threadlocal gymnastics.
    * WAL journal mode so concurrent readers never block the scanner.
    * ``sidecar_mtime`` column enables incremental scans (``scanner``
      skips any sidecar whose mtime hasn't advanced).
    * JSON-typed columns stored as TEXT; ``json_extract`` used for
      summary/hparams sort keys.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    status          TEXT DEFAULT 'running',
    created_at      REAL DEFAULT 0,
    updated_at      REAL DEFAULT 0,
    sidecar_mtime   REAL DEFAULT 0,
    alive           INTEGER DEFAULT 0,
    run_dir         TEXT,
    checkpoint_path TEXT,
    config          TEXT DEFAULT '{}',
    hparams         TEXT DEFAULT '{}',
    summary         TEXT DEFAULT '{}',
    tags            TEXT DEFAULT '[]',
    notes           TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_alive  ON runs(alive);
CREATE INDEX IF NOT EXISTS idx_runs_updated_at ON runs(updated_at);
CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT
);
"""


class Store:
    """SQLite-backed cache for run records.

    Opened either in read-write mode (the scanner) or read-only mode
    (query clients).  Readers and writers do not conflict thanks to WAL.

    Args:
        db_path: Filesystem path to ``registry.db``.
        readonly: If True, open with ``mode=ro`` — safe to use from
            query code while the scanner is writing.  The file must
            already exist; the caller should trigger a scan first.
    """

    def __init__(self, db_path: Union[str, Path], *, readonly: bool = False) -> None:
        self.db_path = str(Path(db_path).resolve())
        self.readonly = readonly

        if readonly:
            # Fail fast if the file doesn't exist — callers should
            # always run a scan before opening read-only.
            if not Path(self.db_path).exists():
                raise FileNotFoundError(
                    f"Registry cache not found at {self.db_path}. Run a scan first."
                )
            uri = f"file:{self.db_path}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, timeout=10)
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, timeout=30)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.executescript(_SCHEMA_SQL)

        self._conn.row_factory = sqlite3.Row

    # ------------------------------------------------------------------ writes

    def upsert(
        self,
        run_id: str,
        data: Dict[str, Any],
        *,
        sidecar_mtime: float,
        alive: bool,
    ) -> None:
        """Insert or update a run record from a parsed sidecar dict.

        Preserves ``created_at`` across updates: if the row already
        exists, its original ``created_at`` is kept.
        """
        self._require_writable()

        now = time.time()
        created_at = float(data.get("created_at") or now)
        updated_at = float(data.get("updated_at") or now)

        self._conn.execute(
            """
            INSERT INTO runs
                (run_id, status, created_at, updated_at, sidecar_mtime,
                 alive, run_dir, checkpoint_path,
                 config, hparams, summary, tags, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                status          = excluded.status,
                updated_at      = excluded.updated_at,
                sidecar_mtime   = excluded.sidecar_mtime,
                alive           = excluded.alive,
                run_dir         = excluded.run_dir,
                checkpoint_path = excluded.checkpoint_path,
                config          = excluded.config,
                hparams         = excluded.hparams,
                summary         = excluded.summary,
                tags            = excluded.tags,
                notes           = excluded.notes
            """,
            (
                run_id,
                str(data.get("status") or "running"),
                created_at,
                updated_at,
                float(sidecar_mtime),
                1 if alive else 0,
                data.get("run_dir"),
                data.get("checkpoint_path"),
                json.dumps(data.get("config") or {}, default=str),
                json.dumps(data.get("hparams") or {}, default=str),
                json.dumps(data.get("summary") or {}, default=str),
                json.dumps(list(data.get("tags") or []), default=str),
                str(data.get("notes") or ""),
            ),
        )

    def mark_alive_bulk(self, alive_map: Dict[str, bool]) -> None:
        """Set ``alive`` for many runs at once.

        Used by the scanner to refresh liveness without re-parsing
        every sidecar — just restat the heartbeat.
        """
        self._require_writable()
        self._conn.executemany(
            "UPDATE runs SET alive = ? WHERE run_id = ?",
            [(1 if v else 0, k) for k, v in alive_map.items()],
        )

    def mark_orphaned(self, run_ids: List[str]) -> None:
        """Mark runs whose ``run_dir`` has disappeared from disk."""
        self._require_writable()
        if not run_ids:
            return
        self._conn.executemany(
            "UPDATE runs SET status = 'orphaned', alive = 0 WHERE run_id = ?",
            [(r,) for r in run_ids],
        )

    def begin(self) -> None:
        self._require_writable()
        self._conn.execute("BEGIN")

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def set_meta(self, key: str, value: str) -> None:
        self._require_writable()
        self._conn.execute(
            "INSERT INTO meta(k, v) VALUES(?, ?) "
            "ON CONFLICT(k) DO UPDATE SET v = excluded.v",
            (key, value),
        )

    # ------------------------------------------------------------------ reads

    def get_meta(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT v FROM meta WHERE k = ?", (key,)).fetchone()
        return row[0] if row else None

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None

    def all_run_dirs(self) -> Dict[str, str]:
        """Return ``{run_id: run_dir}`` for every row. Used for orphan sweep."""
        return {
            r["run_id"]: r["run_dir"]
            for r in self._conn.execute("SELECT run_id, run_dir FROM runs")
            if r["run_dir"]
        }

    def sidecar_mtimes(self) -> Dict[str, float]:
        """Return ``{run_id: sidecar_mtime}`` for incremental scan."""
        return {
            r["run_id"]: r["sidecar_mtime"]
            for r in self._conn.execute("SELECT run_id, sidecar_mtime FROM runs")
        }

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        return row[0] if row else 0

    def query_runs(
        self,
        *,
        status: Optional[str] = None,
        tag: Optional[str] = None,
        alive: Optional[bool] = None,
        sort_by: Optional[str] = None,
        descending: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if alive is not None:
            clauses.append("alive = ?")
            params.append(1 if alive else 0)
        if tag is not None:
            # tags is stored as JSON array text; LIKE is both fast and
            # correct because we include the surrounding quotes.
            clauses.append("tags LIKE ?")
            params.append(f'%"{tag}"%')
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        order = ""
        if sort_by is not None:
            if sort_by.startswith("summary."):
                col_expr = f"json_extract(summary, '$.{sort_by[len('summary.') :]}')"
            elif sort_by.startswith("hparams."):
                col_expr = f"json_extract(hparams, '$.{sort_by[len('hparams.') :]}')"
            elif sort_by.startswith("config."):
                col_expr = f"json_extract(config, '$.{sort_by[len('config.') :]}')"
            else:
                col_expr = sort_by
            direction = "DESC" if descending else "ASC"
            order = f" ORDER BY {col_expr} {direction}"

        limit_clause = f" LIMIT {int(limit)}" if limit is not None else ""
        sql = f"SELECT * FROM runs{where}{order}{limit_clause}"
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [_row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ internals

    def _require_writable(self) -> None:
        if self.readonly:
            raise RuntimeError("Store opened read-only; cannot modify.")


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    for col in ("config", "hparams", "summary"):
        v = d.get(col)
        if isinstance(v, str):
            try:
                d[col] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                d[col] = {}
    v = d.get("tags")
    if isinstance(v, str):
        try:
            d["tags"] = json.loads(v)
        except (json.JSONDecodeError, TypeError):
            d["tags"] = []
    d["alive"] = bool(d.get("alive"))
    return d
