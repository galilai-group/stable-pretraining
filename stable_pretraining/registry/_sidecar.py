# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Sidecar file format: one ``sidecar.json`` per run.

The sidecar is the **source of truth** for run metadata. The SQLite
registry is a derived cache built by the scanner from these files; it
can be deleted and rebuilt at any time.

Layout (one run dir)::

    {run_dir}/
      sidecar.json        ← this file (atomically rewritten)
      heartbeat           ← empty file, mtime touched every flush
      metrics.csv         ← CSVLogger (per-step time series)
      hparams.yaml        ← CSVLogger (hparams)
      checkpoints/        ← Lightning

Write semantics: the writer always performs ``tmp + fsync + rename`` so
a reader can never observe a half-written sidecar. A crash mid-write
either leaves the previous version intact or (worst case) leaves a
``.sidecar.json.*.tmp`` file behind — readers ignore these.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

SIDECAR_NAME = "sidecar.json"
HEARTBEAT_NAME = "heartbeat"
SCHEMA_VERSION = 1

# A run is considered "alive" if its heartbeat is newer than this many
# seconds, unless its status is terminal.  Scanner-side only.
DEFAULT_HEARTBEAT_TIMEOUT_S = 180.0

TERMINAL_STATUSES = frozenset({"completed", "failed", "orphaned"})


def sidecar_path(run_dir: Union[str, Path]) -> Path:
    return Path(run_dir) / SIDECAR_NAME


def heartbeat_path(run_dir: Union[str, Path]) -> Path:
    return Path(run_dir) / HEARTBEAT_NAME


def make_sidecar(
    *,
    run_id: str,
    run_dir: str,
    status: str = "running",
    created_at: Optional[float] = None,
    hparams: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    notes: str = "",
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a sidecar dict with canonical field order and defaults."""
    now = time.time()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_dir": run_dir,
        "status": status,
        "created_at": created_at if created_at is not None else now,
        "updated_at": now,
        "tags": list(tags or []),
        "notes": notes or "",
        "hparams": dict(hparams or {}),
        "summary": dict(summary or {}),
        "checkpoint_path": checkpoint_path,
    }


def write_sidecar(run_dir: Union[str, Path], data: Dict[str, Any]) -> Path:
    """Atomically write ``data`` to ``{run_dir}/sidecar.json``.

    Uses ``tempfile.mkstemp`` in the destination directory + ``fsync`` +
    ``os.replace`` so readers never observe a partial write and the
    rename is atomic on POSIX.

    Returns the sidecar path.  Raises on I/O failure — the caller is
    responsible for swallowing exceptions during teardown if needed.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = run_dir / SIDECAR_NAME

    # Stamp updated_at at write time so it reflects the actual flush.
    data = dict(data)
    data["updated_at"] = time.time()

    fd, tmp = tempfile.mkstemp(dir=str(run_dir), prefix=".sidecar.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=False, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return dest


def read_sidecar(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Read a sidecar.  Returns ``None`` on missing / partial / invalid JSON.

    The scanner calls this on every changed sidecar; robustness to races
    (half-written, deleted mid-read) is essential.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return None
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    if not isinstance(data, dict) or "run_id" not in data:
        return None
    return data


def touch_heartbeat(run_dir: Union[str, Path]) -> None:
    """Update the heartbeat mtime.  Creates the file if missing.

    Called from the hot path (``log_metrics``); must be cheap and must
    never raise — we swallow all OS errors.
    """
    try:
        hb = Path(run_dir) / HEARTBEAT_NAME
        # os.utime with None sets both atime and mtime to "now".
        # Fast-path: if the file already exists, a single utime syscall.
        try:
            os.utime(hb, None)
        except FileNotFoundError:
            hb.parent.mkdir(parents=True, exist_ok=True)
            # Create empty file with current mtime.
            hb.touch()
    except OSError:
        pass


def heartbeat_mtime(run_dir: Union[str, Path]) -> Optional[float]:
    """Return heartbeat mtime in seconds since epoch, or ``None`` if absent."""
    try:
        return os.stat(Path(run_dir) / HEARTBEAT_NAME).st_mtime
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return None


def is_alive(
    status: str,
    hb_mtime: Optional[float],
    *,
    now: Optional[float] = None,
    timeout_s: float = DEFAULT_HEARTBEAT_TIMEOUT_S,
) -> bool:
    """Decide whether a run is currently alive.

    A terminal status always wins over heartbeat: a run reported as
    ``completed`` is never considered alive, even if its heartbeat is
    still warm from a post-training flush.
    """
    if status in TERMINAL_STATUSES:
        return False
    if hb_mtime is None:
        return False
    now = now if now is not None else time.time()
    return (now - hb_mtime) < timeout_s


def _json_default(obj: Any) -> Any:
    """Fallback encoder for values that aren't JSON-native.

    We only ever serialize simple metadata here (hparams flattened to
    scalars, tags, notes).  Anything exotic becomes a string so the
    sidecar stays round-trippable.
    """
    try:
        # numpy scalars, torch scalars, Path, etc.
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "__fspath__"):
            return os.fspath(obj)
    except Exception:
        pass
    return str(obj)
