# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Lightning logger for the filesystem-backed run registry.

:class:`RegistryLogger` is a thin subclass of Lightning's
:class:`~lightning.pytorch.loggers.CSVLogger`.  It writes the standard
CSV + hparams artifacts **and** an indexable ``sidecar.json`` + a
``heartbeat`` file in the run directory.

Nothing in the training path touches SQLite or a network server:

* ``log_hyperparams`` → CSV hparams + sidecar snapshot.
* ``log_metrics``     → CSV metrics row + summary accumulator + heartbeat touch.
* ``save``            → CSV flush + sidecar rewrite (atomic).
* ``finalize``        → terminal status in sidecar.
* ``after_save_checkpoint`` → ``checkpoint_path`` in sidecar.

A separate scanner (see :mod:`stable_pretraining.registry._scanner`)
turns sidecars into a fast-queryable SQLite cache.  Deleting that cache
is harmless — rerun ``spt registry scan --full`` to rebuild.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from . import _sidecar


class RegistryLogger(CSVLogger):
    """CSV logger with a filesystem-indexable sidecar.

    The sidecar is an atomically-rewritten JSON file that captures the
    run's hparams, latest metric values (``summary``), status, and
    checkpoint path.  It is the source of truth for the registry
    scanner.

    Args:
        run_dir: Directory this run writes to.  CSV logs,
            ``sidecar.json`` and ``heartbeat`` all live here.
        run_id: Unique identifier for this run (typically the SLURM job
            id or a deterministic hash).  Used as the primary key in
            the registry cache and as the CSV version component.
        tags: Free-form string tags for grouping runs (e.g. model
            architecture, experiment name, sweep id).  Any
            ``SLURM_ARRAY_JOB_ID`` env var is auto-appended as
            ``"sweep:<id>"`` for array-job convenience.
        notes: Optional free-text description.
        flush_logs_every_n_steps: How often the CSV is flushed; the
            sidecar is rewritten on the same cadence.  The heartbeat
            is touched on every ``log_metrics`` call (cheap).
    """

    def __init__(
        self,
        run_dir: Union[str, Path],
        run_id: str,
        *,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
        flush_logs_every_n_steps: int = 50,
    ) -> None:
        run_dir = Path(run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        # save_dir + name="" + version="" ⇒ CSVLogger.log_dir == run_dir.
        # Matches the existing Manager-auto-CSV layout.
        super().__init__(
            save_dir=str(run_dir),
            name="",
            version="",
            flush_logs_every_n_steps=flush_logs_every_n_steps,
        )

        self._run_dir = run_dir
        self._run_id = str(run_id)

        self._tags: list[str] = list(tags or [])
        array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
        if array_job and f"sweep:{array_job}" not in self._tags:
            self._tags.append(f"sweep:{array_job}")

        self._notes = notes or ""
        self._hparams: dict[str, Any] = {}
        self._summary: dict[str, Any] = {}
        self._checkpoint_path: Optional[str] = None
        self._status = "running"
        # Preserve the first-write timestamp across sidecar rewrites so
        # the registry can order runs chronologically regardless of how
        # often we flush.
        self._created_at: Optional[float] = None

    # -- identity ---------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    # -- lightning hooks --------------------------------------------------------

    @rank_zero_only
    def log_hyperparams(
        self, params: Union[dict[str, Any], Any], *args: Any, **kw: Any
    ) -> None:
        # Persist to CSVLogger's hparams.yaml.
        super().log_hyperparams(params, *args, **kw)
        self._hparams = _flatten_params(params)
        self._write_sidecar()

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        # CSV-side: write the raw per-step row.
        super().log_metrics(metrics, step)

        # Sidecar-side: accumulate last-value-per-key summary.
        for k, v in metrics.items():
            scalar = _to_scalar(v)
            if scalar is not None:
                self._summary[k] = scalar

        # Heartbeat: cheap, fire-and-forget; used by the scanner to
        # distinguish running / stalled / dead without contacting SLURM.
        _sidecar.touch_heartbeat(self._run_dir)

    @rank_zero_only
    def save(self) -> None:
        super().save()
        self._write_sidecar()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        # Map Lightning status strings to our canonical vocabulary.
        self._status = {"success": "completed", "failed": "failed"}.get(status, status)
        # Parent writes CSVs.  We don't call super().finalize first
        # because _experiment may be None on rank-zero callers that
        # never logged — super() handles that no-op correctly.
        super().finalize(status)
        self._write_sidecar()

    def after_save_checkpoint(self, checkpoint_callback: Any) -> None:
        # This callback fires on every rank; we gate on rank_zero via
        # the helper write (which is rank-zero-only upstream).
        path = (
            getattr(checkpoint_callback, "best_model_path", None)
            or getattr(checkpoint_callback, "last_model_path", None)
            or None
        )
        if path:
            self._checkpoint_path = str(path)
            self._write_sidecar_safe()

    # -- sidecar ----------------------------------------------------------------

    @rank_zero_only
    def _write_sidecar(self) -> None:
        """Atomically (re)write the sidecar.  Let exceptions propagate."""
        data = _sidecar.make_sidecar(
            run_id=self._run_id,
            run_dir=str(self._run_dir),
            status=self._status,
            created_at=self._created_at,
            hparams=self._hparams,
            summary=self._summary,
            tags=self._tags,
            notes=self._notes,
            checkpoint_path=self._checkpoint_path,
        )
        _sidecar.write_sidecar(self._run_dir, data)
        if self._created_at is None:
            self._created_at = data["created_at"]

    @rank_zero_only
    def _write_sidecar_safe(self) -> None:
        """Same as :meth:`_write_sidecar` but swallows I/O errors.

        Used from callback hooks where a failed write should never take
        down a training run.
        """
        try:
            self._write_sidecar()
        except OSError:
            pass


# --------------------------------------------------------------------- helpers


def _flatten_params(params: Any) -> dict[str, Any]:
    """Flatten a (possibly nested) hparams object to a flat JSON-safe dict.

    Accepts ``DictConfig``, ``Namespace``, dicts, lists, scalars, or
    anything.  Non-serializable values are stringified so the sidecar
    stays round-trippable.
    """
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(params, DictConfig):
            params = OmegaConf.to_container(params, resolve=True)
    except ImportError:
        pass

    if not isinstance(params, dict):
        params = (
            vars(params) if hasattr(params, "__dict__") else {"params": str(params)}
        )

    out: dict[str, Any] = {}
    _flatten(params, "", out)
    return out


def _flatten(obj: Any, prefix: str, out: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{prefix}{k}." if prefix else f"{k}.", out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten(v, f"{prefix}{i}.", out)
    else:
        key = prefix.rstrip(".")
        try:
            json.dumps(obj)
            out[key] = obj
        except (TypeError, ValueError):
            out[key] = str(obj)


def _to_scalar(v: Any) -> Optional[float]:
    """Coerce metric value to a float scalar, or ``None`` if not scalar.

    Handles torch Tensors, numpy scalars, int, float, bool.  Anything
    else (strings, multi-element tensors, etc.) is skipped — we
    deliberately keep the summary numeric so downstream tools can
    always ``float()`` it.
    """
    # Common path: plain float/int.
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    # Tensor-like with an ``item()`` method and 0-dim shape.
    item = getattr(v, "item", None)
    if callable(item):
        try:
            numel = getattr(v, "numel", None)
            if callable(numel) and numel() != 1:
                return None
            return float(item())
        except (RuntimeError, ValueError, TypeError):
            return None
    return None
