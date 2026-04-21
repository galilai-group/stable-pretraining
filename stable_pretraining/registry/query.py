# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Read-only query API for the filesystem-backed run registry.

Usage::

    import stable_pretraining as spt

    reg = spt.open_registry()  # lazily scans before querying

    runs = reg.query(tag="sweep:12345")
    best = reg.query(tag="sweep:12345", sort_by="summary.val_acc", limit=5)

    df = reg.to_dataframe(tag="resnet50")

``open_registry()`` triggers an incremental scan of
``{cache_dir}/runs/**`` before returning.  A short in-process TTL
short-circuits back-to-back calls so scripts stay snappy.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from . import _scanner
from ._store import Store


@dataclasses.dataclass(frozen=True)
class RunRecord:
    """Immutable view of a single training run, hydrated from the cache."""

    run_id: str
    status: str
    created_at: float
    updated_at: float
    alive: bool
    run_dir: Optional[str]
    checkpoint_path: Optional[str]
    config: Dict[str, Any]
    hparams: Dict[str, Any]
    summary: Dict[str, Any]
    tags: List[str]
    notes: str


class Registry:
    """Read-only query interface over the registry cache.

    Instantiate via :func:`open_registry` rather than directly — the
    factory runs a lazy filesystem scan first so you don't query stale
    data.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    # ------------------------------------------------------------------ queries

    def query(
        self,
        *,
        tag: Optional[str] = None,
        status: Optional[str] = None,
        alive: Optional[bool] = None,
        hparams: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        descending: bool = True,
        limit: Optional[int] = None,
    ) -> List[RunRecord]:
        """Query runs matching filters.

        Args:
            tag: Include runs that carry this tag (uses substring match
                on the stored JSON tag array).
            status: Filter by ``status`` column (``running``,
                ``completed``, ``failed``, ``orphaned``, ``interrupted``).
            alive: Filter by heartbeat-based liveness.
            hparams: ``{key: value}`` pairs the flattened hparams must
                match (AND, client-side).
            sort_by: Column name or ``summary.<k>`` / ``hparams.<k>``
                / ``config.<k>``.
            descending: Sort order.
            limit: Max rows.
        """
        rows = self._store.query_runs(
            tag=tag,
            status=status,
            alive=alive,
            sort_by=sort_by,
            descending=descending,
            limit=limit,
        )
        records = [_row_to_record(r) for r in rows]
        if hparams:
            records = [
                r
                for r in records
                if all(r.hparams.get(k) == v for k, v in hparams.items())
            ]
        return records

    def get(self, run_id: str) -> Optional[RunRecord]:
        row = self._store.get_run(run_id)
        return _row_to_record(row) if row else None

    def to_dataframe(self, **query_kwargs: Any):
        """Return a DataFrame with flattened ``hparams.*`` / ``summary.*`` cols."""
        import pandas as pd

        records = self.query(**query_kwargs)
        if not records:
            return pd.DataFrame()

        rows: list[Dict[str, Any]] = []
        for r in records:
            row: Dict[str, Any] = {
                "run_id": r.run_id,
                "status": r.status,
                "alive": r.alive,
                "created_at": r.created_at,
                "updated_at": r.updated_at,
                "run_dir": r.run_dir,
                "checkpoint_path": r.checkpoint_path,
                "tags": r.tags,
                "notes": r.notes,
            }
            for k, v in (r.hparams or {}).items():
                row[f"hparams.{k}"] = v
            for k, v in (r.summary or {}).items():
                row[f"summary.{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ dunder

    def __len__(self) -> int:
        return self._store.count()

    def __getitem__(self, run_id: str) -> RunRecord:
        rec = self.get(run_id)
        if rec is None:
            raise KeyError(run_id)
        return rec

    def __repr__(self) -> str:
        return f"Registry(db_path={self._store.db_path!r}, runs={len(self)})"

    def close(self) -> None:
        self._store.close()


# --------------------------------------------------------------------- factory


def open_registry(
    db_path: Optional[Union[str, Path]] = None,
    *,
    cache_dir: Optional[Union[str, Path]] = None,
    scan: bool = True,
    scan_ttl_s: float = 2.0,
) -> Registry:
    """Open the registry for querying.

    Args:
        db_path: Path to the cache DB.  Defaults to
            ``{cache_dir}/registry.db``.
        cache_dir: Root where runs are stored (``{cache_dir}/runs/...``).
            Defaults to ``spt.set(cache_dir=...)``.
        scan: Run an incremental scan before returning, so the cache
            reflects the current filesystem state.  Disable when you
            know the scan was just done.
        scan_ttl_s: If another scan happened within this many seconds
            in the current process, skip.

    Returns:
        A read-only :class:`Registry`.
    """
    resolved_cache, resolved_db = _resolve_paths(cache_dir, db_path)

    if scan:
        _scanner.scan_for_query(
            resolved_cache,
            resolved_db,
            ttl_s=scan_ttl_s,
        )

    return Registry(Store(resolved_db, readonly=True))


def _resolve_paths(
    cache_dir: Optional[Union[str, Path]],
    db_path: Optional[Union[str, Path]],
) -> tuple[Path, Path]:
    """Resolve ``(cache_dir, db_path)`` from args + global config."""
    if cache_dir is None or db_path is None:
        try:
            from .._config import get_config

            cfg_cache = get_config().cache_dir
        except Exception:
            cfg_cache = None

        if cache_dir is None:
            if cfg_cache is None:
                raise ValueError(
                    "No cache_dir provided and spt.set(cache_dir=...) is not "
                    "configured. Pass an explicit cache_dir or set it globally."
                )
            cache_dir = cfg_cache

    cache_dir = Path(cache_dir).expanduser().resolve()
    if db_path is None:
        db_path = cache_dir / "registry.db"
    else:
        db_path = Path(db_path).expanduser().resolve()

    return cache_dir, db_path


def _row_to_record(d: Dict[str, Any]) -> RunRecord:
    return RunRecord(
        run_id=d["run_id"],
        status=d.get("status", "unknown"),
        created_at=float(d.get("created_at") or 0.0),
        updated_at=float(d.get("updated_at") or 0.0),
        alive=bool(d.get("alive")),
        run_dir=d.get("run_dir"),
        checkpoint_path=d.get("checkpoint_path"),
        config=d.get("config", {}) or {},
        hparams=d.get("hparams", {}) or {},
        summary=d.get("summary", {}) or {},
        tags=d.get("tags", []) or [],
        notes=d.get("notes", "") or "",
    )
