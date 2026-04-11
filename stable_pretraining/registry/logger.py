# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Lightning Logger that writes run metadata to a local SQLite registry.

Drop-in replacement (or complement) for WandbLogger / CSVLogger.  All
``self.log()`` and ``self.log_dict()`` calls from the LightningModule are
routed here automatically by Lightning.

Usage in YAML (uses ``spt.set(cache_dir=...)`` automatically)::

    logger:
      _target_: stable_pretraining.registry.RegistryLogger
      tags: [resnet50, simclr, ablation-v2]

Or auto-injected when ``spt.set(cache_dir=...)`` is configured.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from ._db import RegistryDB


class RegistryLogger(Logger):
    """Lightning Logger backed by a local SQLite database.

    Every ``self.log()``, ``self.log_dict()`` call in the LightningModule
    flows through :meth:`log_metrics`, accumulating a wandb-style summary
    dict (last value per key).  The summary is periodically flushed to
    the database and finalized at the end of training.

    Grouping is done entirely through **tags**.  SLURM array jobs
    automatically get a ``"sweep:<SLURM_ARRAY_JOB_ID>"`` tag so all
    tasks in the same array are queryable as a group.

    Args:
        db_path: Path to the SQLite database file.  If ``None``
            (default), uses ``{cache_dir}/registry.db`` from
            ``spt.set(cache_dir=...)``.
        tags: Optional list of string tags for this run.  Use tags for
            any kind of grouping: model architecture, experiment name,
            sweep identifier, etc.
        notes: Optional free-text notes / description for the run.
        flush_every: Flush summary to the database every *N* calls to
            :meth:`log_metrics`.  Lower values give more up-to-date
            results at the cost of more DB writes.  Default 50.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
        flush_every: int = 50,
    ) -> None:
        super().__init__()
        if db_path is None:
            db_path = self._resolve_db_path()
        self._db = RegistryDB(db_path)
        self._tags = list(tags or [])
        self._notes = notes or ""
        self._flush_every = flush_every

        # Auto-tag SLURM array jobs so sweeps are queryable
        array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
        if array_job and f"sweep:{array_job}" not in self._tags:
            self._tags.append(f"sweep:{array_job}")

        # Set externally by Manager before training starts
        self._run_id: Optional[str] = None
        self._run_dir: Optional[str] = None

        # Accumulated summary — latest value per metric key
        self._summary: dict[str, float] = {}
        self._log_count = 0
        self._registered = False

    # -- abstract properties ---------------------------------------------------

    @property
    def name(self) -> Optional[str]:
        return "registry"

    @property
    def version(self) -> Optional[str]:
        return self._run_id

    @property
    def save_dir(self) -> Optional[str]:
        return self._run_dir

    # -- resilient DB calls ----------------------------------------------------

    def _safe_db_call(self, fn, *args, **kwargs) -> bool:
        """Call a DB method.

        On failure (server died during preemption), save a sidecar JSON file
        in the run directory so data can be recovered later with
        ``spt registry sync``.

        Returns True on success, False on failure.
        """
        try:
            fn(*args, **kwargs)
            return True
        except Exception as exc:
            from loguru import logger as _log

            _log.warning(f"! Registry server unreachable ({exc}), writing sidecar.")
            self._write_sidecar()
            return False

    def _write_sidecar(self) -> None:
        """Write the current run state as a JSON sidecar file in the run directory.

        ``spt registry sync`` picks these up later.
        """
        if self._run_dir is None or self._run_id is None:
            return
        try:
            sidecar_dir = Path(self._run_dir)
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            sidecar = sidecar_dir / ".registry_sidecar.json"
            data = {
                "run_id": self._run_id,
                "status": "interrupted",
                "run_dir": self._run_dir,
                "config": {},
                "hparams": {},
                "summary": dict(self._summary),
                "tags": list(self._tags),
                "notes": self._notes,
            }
            import json
            import tempfile
            import os

            fd, tmp = tempfile.mkstemp(dir=str(sidecar_dir), suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp, str(sidecar))
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        except Exception:
            pass  # best-effort — don't crash during teardown

    # -- logging ---------------------------------------------------------------

    @rank_zero_only
    def log_hyperparams(
        self, params: Union[dict[str, Any], Any], *args: Any, **kwargs: Any
    ) -> None:
        if self._run_id is None:
            return

        # Flatten nested config to a simple dict
        flat = _flatten_params(params)

        # log_hyperparams is called once at startup — let errors
        # propagate so the user sees "server not running" immediately.
        self._db.insert_run(
            self._run_id,
            status="running",
            run_dir=self._run_dir,
            config=flat,
            hparams=flat,
            tags=self._tags,
            notes=self._notes,
        )
        self._registered = True

    @rank_zero_only
    def log_metrics(
        self, metrics: dict[str, float], step: Optional[int] = None
    ) -> None:
        self._summary.update(metrics)
        self._log_count += 1

        if self._log_count % self._flush_every == 0:
            self._flush_summary()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._run_id is None:
            return

        # Map Lightning status strings
        db_status = {"success": "completed", "failed": "failed"}.get(status, status)

        # Find best checkpoint path if available
        checkpoint_path = self._find_checkpoint()

        fields: dict[str, Any] = {
            "summary": self._summary,
            "status": db_status,
        }
        if checkpoint_path is not None:
            fields["checkpoint_path"] = checkpoint_path

        # Ensure the run exists even if log_hyperparams was never called
        if not self._registered:
            ok = self._safe_db_call(
                self._db.insert_run,
                self._run_id,
                status=db_status,
                run_dir=self._run_dir,
                tags=self._tags,
                notes=self._notes,
            )
            if ok:
                self._registered = True

        self._safe_db_call(self._db.update_run, self._run_id, **fields)
        self._db.close()

    def after_save_checkpoint(self, checkpoint_callback: Any) -> None:
        """Update checkpoint_path in the DB whenever Lightning saves a checkpoint."""
        if self._run_id is None:
            return
        path = getattr(checkpoint_callback, "best_model_path", None)
        if not path:
            path = getattr(checkpoint_callback, "last_model_path", None)
        if path:
            self._safe_db_call(
                self._db.update_run, self._run_id, checkpoint_path=str(path)
            )

    # -- path resolution -------------------------------------------------------

    @staticmethod
    def _resolve_db_path() -> str:
        from .._config import get_config

        cfg = get_config()
        if cfg.cache_dir is None:
            raise ValueError(
                "RegistryLogger requires a db_path or spt.set(cache_dir=...). "
                "Either pass db_path explicitly or configure cache_dir first."
            )
        return str(Path(cfg.cache_dir).resolve() / "registry.db")

    # -- internals -------------------------------------------------------------

    def _flush_summary(self) -> None:
        if self._run_id is None or not self._summary:
            return
        self._safe_db_call(self._db.update_run, self._run_id, summary=self._summary)

    def _find_checkpoint(self) -> Optional[str]:
        """Scan run_dir/checkpoints/ for the best or last checkpoint."""
        if self._run_dir is None:
            return None
        ckpt_dir = Path(self._run_dir) / "checkpoints"
        if not ckpt_dir.is_dir():
            return None
        # Prefer best model, fall back to last.ckpt
        ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if not ckpts:
            return None
        # If there's a non-"last" checkpoint, prefer it (likely the "best")
        for ckpt in reversed(ckpts):
            if ckpt.stem != "last":
                return str(ckpt)
        return str(ckpts[-1])


def _flatten_params(params: Any) -> dict[str, Any]:
    """Flatten nested config/params to a dot-separated dict of JSON-safe values."""
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(params, DictConfig):
            params = OmegaConf.to_container(params, resolve=True)
    except ImportError:
        pass

    if not isinstance(params, dict):
        # Namespace or other object
        params = (
            vars(params) if hasattr(params, "__dict__") else {"params": str(params)}
        )

    flat: dict[str, Any] = {}
    _flatten(params, "", flat)
    return flat


def _flatten(obj: Any, prefix: str, out: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{prefix}{k}." if prefix else f"{k}.", out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten(v, f"{prefix}{i}.", out)
    else:
        key = prefix.rstrip(".")
        # Ensure JSON-serializable
        try:
            json.dumps(obj)
            out[key] = obj
        except (TypeError, ValueError):
            out[key] = str(obj)
