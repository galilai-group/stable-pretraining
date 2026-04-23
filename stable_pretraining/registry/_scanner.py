# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Filesystem → SQLite scanner.

The scanner walks ``{cache_dir}/runs/**/sidecar.json`` and upserts each
run into the cache store.  It is the **only writer** to the SQLite DB
— training jobs never touch it directly.

Two entry points:

* :func:`scan` — full pass (``--full``) or incremental (default, based
  on per-sidecar ``st_mtime``).
* :func:`scan_for_query` — convenience wrapper used by
  :func:`open_registry`; honours a short in-memory TTL so repeated
  queries within a single script don't re-scan.

The whole point of this module is to be fast and boring: stat-heavy,
parse-light, single transaction per scan.
"""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path
from typing import Iterable, Optional, Union

from loguru import logger as _log

from . import _sidecar
from ._store import Store


@dataclasses.dataclass
class ScanReport:
    """Summary of a scan pass — returned to callers and logged."""

    total_sidecars: int = 0
    upserted: int = 0
    skipped_unchanged: int = 0
    skipped_invalid: int = 0
    orphaned: int = 0
    elapsed_s: float = 0.0

    def __str__(self) -> str:
        return (
            f"scan: {self.total_sidecars} sidecars, "
            f"{self.upserted} upserted, "
            f"{self.skipped_unchanged} unchanged, "
            f"{self.skipped_invalid} invalid, "
            f"{self.orphaned} orphaned "
            f"in {self.elapsed_s * 1000:.0f} ms"
        )


def scan(
    cache_dir: Union[str, Path],
    store: Store,
    *,
    full: bool = False,
    heartbeat_timeout_s: float = _sidecar.DEFAULT_HEARTBEAT_TIMEOUT_S,
) -> ScanReport:
    """Walk ``cache_dir/runs`` and upsert any sidecars into ``store``.

    Args:
        cache_dir: Root cache directory (layout ``{cache_dir}/runs/...``).
        store: An opened, writable :class:`Store`.
        full: If True, re-ingest every sidecar regardless of mtime.
            Use to recover from DB corruption / schema migration.
        heartbeat_timeout_s: Max age of heartbeat file before a non-terminal
            run is flagged as ``alive=False``.

    Returns:
        :class:`ScanReport` with counts and elapsed time.
    """
    t0 = time.monotonic()
    cache_dir = Path(cache_dir)
    runs_root = cache_dir / "runs"

    report = ScanReport()
    if not runs_root.is_dir():
        report.elapsed_s = time.monotonic() - t0
        return report

    known_mtimes = {} if full else store.sidecar_mtimes()
    now = time.time()

    store.begin()
    try:
        for sidecar_file in _iter_sidecars(runs_root):
            report.total_sidecars += 1

            try:
                st = os.stat(sidecar_file)
            except FileNotFoundError:
                continue  # raced with a delete
            mtime = st.st_mtime

            # Run id is the leaf directory name — cheap and stable.
            run_dir = sidecar_file.parent
            run_id = run_dir.name

            prev_mtime = known_mtimes.get(run_id, 0.0)

            if not full and mtime <= prev_mtime:
                # Still need to refresh liveness, but we already have
                # this sidecar in the DB.
                hb = _sidecar.heartbeat_mtime(run_dir)
                # Status is whatever we last wrote; fetch quickly.
                row = store.get_run(run_id)
                status = row["status"] if row else "running"
                alive = _sidecar.is_alive(
                    status, hb, now=now, timeout_s=heartbeat_timeout_s
                )
                if row is None or bool(row.get("alive")) != alive:
                    store.mark_alive_bulk({run_id: alive})
                report.skipped_unchanged += 1
                continue

            data = _sidecar.read_sidecar(sidecar_file)
            if data is None:
                report.skipped_invalid += 1
                continue

            # Ensure run_dir field points to the actual directory — the
            # sidecar may have been moved (e.g. cache relocation).
            data.setdefault("run_dir", str(run_dir))
            if data.get("run_id") != run_id:
                # Sidecar's run_id disagrees with its directory name.
                # Trust the sidecar (authoritative) but log for diag.
                _log.debug(
                    f"sidecar run_id={data.get('run_id')!r} != "
                    f"dir={run_id!r} at {sidecar_file}"
                )
                run_id = str(data["run_id"])

            hb = _sidecar.heartbeat_mtime(run_dir)
            alive = _sidecar.is_alive(
                data.get("status", "running"),
                hb,
                now=now,
                timeout_s=heartbeat_timeout_s,
            )

            store.upsert(run_id, data, sidecar_mtime=mtime, alive=alive)
            report.upserted += 1

        # Orphan sweep: any row whose run_dir no longer exists.
        orphans: list[str] = []
        for rid, rdir in store.all_run_dirs().items():
            if rdir and not Path(rdir).exists():
                orphans.append(rid)
        store.mark_orphaned(orphans)
        report.orphaned = len(orphans)

        store.set_meta("last_scan_at", str(now))
        store.commit()
    except BaseException:
        store.rollback()
        raise

    report.elapsed_s = time.monotonic() - t0
    return report


def _iter_sidecars(runs_root: Path) -> Iterable[Path]:
    """Yield every ``sidecar.json`` under ``runs_root``.

    Uses :func:`Path.rglob` with the exact filename so the walker can
    skip entire directory subtrees without matching.  Faster than
    parsing glob patterns for our case.
    """
    return runs_root.rglob(_sidecar.SIDECAR_NAME)


# --------------------------------------------------------------------- TTL

# Short in-process memo so repeated open_registry() calls in one script
# don't each re-scan.  Keyed by (cache_dir, db_path); value is the epoch
# seconds of the last scan.
_LAST_SCAN_AT: dict[tuple[str, str], float] = {}
_DEFAULT_TTL_S = 2.0


def scan_for_query(
    cache_dir: Union[str, Path],
    db_path: Union[str, Path],
    *,
    ttl_s: float = _DEFAULT_TTL_S,
    heartbeat_timeout_s: float = _sidecar.DEFAULT_HEARTBEAT_TIMEOUT_S,
) -> Optional[ScanReport]:
    """Trigger a lazy scan before a query, with a small in-memory TTL.

    Returns ``None`` when the TTL short-circuits the scan.
    """
    key = (str(Path(cache_dir).resolve()), str(Path(db_path).resolve()))
    last = _LAST_SCAN_AT.get(key, 0.0)
    if time.monotonic() - last < ttl_s:
        return None

    with Store(db_path, readonly=False) as store:
        report = scan(
            cache_dir,
            store,
            heartbeat_timeout_s=heartbeat_timeout_s,
        )
    _LAST_SCAN_AT[key] = time.monotonic()
    return report


def invalidate_ttl() -> None:
    """Forget the last-scan timestamps.  Test-only hook."""
    _LAST_SCAN_AT.clear()
