# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Read-only query API for the run registry.

Example::

    import stable_pretraining as spt

    reg = spt.open_registry()

    # All runs from a SLURM array sweep
    runs = reg.query(tag="sweep:12345")

    # Best runs by validation accuracy
    best = reg.query(tag="sweep:12345", sort_by="summary.val_acc", limit=5)
    print(best[0].summary)  # {"val_acc": 0.847, ...}
    print(best[0].run_dir)  # /scratch/runs/runs/20260408/.../12345_0/
    print(best[0].checkpoint_path)

    # Export to pandas DataFrame
    df = reg.to_dataframe(tag="resnet50")
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._db import RegistryDB


@dataclasses.dataclass(frozen=True)
class RunRecord:
    """Immutable record for a single training run."""

    run_id: str
    status: str
    created_at: float
    updated_at: float
    run_dir: Optional[str]
    checkpoint_path: Optional[str]
    config: Dict[str, Any]
    hparams: Dict[str, Any]
    summary: Dict[str, Any]
    tags: List[str]
    notes: str


class Registry:
    """Read-only query interface over the run registry database.

    Do not instantiate directly — use :func:`open_registry`.
    """

    def __init__(self, db: RegistryDB) -> None:
        self._db = db

    def query(
        self,
        *,
        tag: Optional[str] = None,
        status: Optional[str] = None,
        hparams: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        descending: bool = True,
        limit: Optional[int] = None,
    ) -> List[RunRecord]:
        """Query runs matching the given filters.

        Args:
            tag: Filter runs that contain this tag.  Use
                ``"sweep:<job_id>"`` to find all runs from a SLURM array.
            status: Filter by run status (``"running"``, ``"completed"``,
                ``"failed"``).
            hparams: Dict of ``{key: value}`` pairs to match against the
                flattened hparams JSON.  All pairs must match (AND logic).
            sort_by: Column or JSON path to sort by.  Use
                ``"summary.<key>"`` or ``"hparams.<key>"`` for JSON fields,
                or plain column names like ``"created_at"``.
            descending: Sort direction (default ``True``).
            limit: Maximum number of results.

        Returns:
            List of :class:`RunRecord` objects.
        """
        rows = self._db.query_runs(
            tag=tag,
            status=status,
            sort_by=sort_by,
            descending=descending,
            limit=limit,
        )

        records = [_dict_to_record(r) for r in rows]

        # Client-side hparams filtering (works even without json_extract)
        if hparams:
            records = [
                r
                for r in records
                if all(r.hparams.get(k) == v for k, v in hparams.items())
            ]

        return records

    def get(self, run_id: str) -> Optional[RunRecord]:
        """Fetch a single run by ID."""
        row = self._db.get_run(run_id)
        if row is None:
            return None
        return _dict_to_record(row)

    def to_dataframe(self, **query_kwargs: Any) -> "pd.DataFrame":
        """Query runs and return a pandas DataFrame.

        Flattens ``hparams`` and ``summary`` dicts into columns with
        ``hparams.`` and ``summary.`` prefixes.

        Accepts the same keyword arguments as :meth:`query`.
        """
        import pandas as pd

        records = self.query(**query_kwargs)
        if not records:
            return pd.DataFrame()

        rows = []
        for r in records:
            row: Dict[str, Any] = {
                "run_id": r.run_id,
                "status": r.status,
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

    def __len__(self) -> int:
        return self._db.count()

    def __getitem__(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        if record is None:
            raise KeyError(run_id)
        return record

    def __repr__(self) -> str:
        return f"Registry(db_path={self._db.db_path!r}, runs={len(self)})"

    def close(self) -> None:
        self._db.close()


def open_registry(db_path: Optional[str] = None) -> Registry:
    """Open a run registry database for querying.

    Args:
        db_path: Path to the SQLite database file.  If ``None``, uses
            ``{cache_dir}/registry.db`` from the global config.

    Returns:
        :class:`Registry` instance.

    Raises:
        ValueError: If ``db_path`` is None and no ``cache_dir`` is configured.
    """
    if db_path is None:
        from .._config import get_config

        cfg = get_config()
        if cfg.cache_dir is None:
            raise ValueError(
                "No db_path provided and spt.set(cache_dir=...) is not configured. "
                "Pass an explicit db_path or set cache_dir first."
            )
        db_path = str(Path(cfg.cache_dir).resolve() / "registry.db")

    return Registry(RegistryDB(db_path))


def _dict_to_record(d: Dict[str, Any]) -> RunRecord:
    return RunRecord(
        run_id=d["run_id"],
        status=d.get("status", "unknown"),
        created_at=d.get("created_at", 0.0),
        updated_at=d.get("updated_at", 0.0),
        run_dir=d.get("run_dir"),
        checkpoint_path=d.get("checkpoint_path"),
        config=d.get("config", {}),
        hparams=d.get("hparams", {}),
        summary=d.get("summary", {}),
        tags=d.get("tags", []),
        notes=d.get("notes", ""),
    )


try:
    import pandas as pd  # noqa: F401
except ImportError:
    pass
