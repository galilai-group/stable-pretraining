"""Programmatic entry point for the JAX backend with SLURM-preemption support.

Mirrors the logic of :class:`stable_pretraining.Manager` (the torch/Lightning
one) for the parts that matter on a cluster:

* **Cache-dir / run-dir resolution** — identical layout
  ``{cache_dir}/runs/{YYYYMMDD}/{HHMMSS}/{run_id}/`` with checkpoints under
  ``checkpoints/last.msgpack``; ``cache_dir`` comes from the shared
  :func:`stable_pretraining.get_config` (``SPT_CACHE_DIR`` / ``spt.set``).
* **SLURM requeue resume** — a ``{cache_dir}/.slurm_index/<job[_task]>`` entry
  maps the SLURM job to its run-dir; on requeue (``SLURM_RESTART_COUNT >= 1``)
  the original run-dir is reused and training auto-resumes from
  ``last.msgpack`` (full restore: params + optimizer + step).
* **Preemption** — a ``SIGTERM`` handler flags the run; the training loop then
  writes an atomic checkpoint and issues ``scontrol requeue`` before exiting,
  so the requeued job picks up exactly where it left off.

This is self-contained (no submitit/Lightning); the checkpoint loop does the
heavy work between batches (not inside the signal handler), which is the safe
pattern.
"""

import os
import signal
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from loguru import logger as logging

from .._config import get_config
from .checkpoint import Checkpoint, save
from .trainer import Callback


def _slurm_session_key() -> Optional[str]:
    """Stable per-task SLURM key (``job`` or ``job_task``), or ``None`` off-SLURM."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    return f"{job}_{task}" if task is not None else job


def _is_slurm_requeue() -> bool:
    """True only on a SLURM requeue (``SLURM_RESTART_COUNT >= 1``)."""
    try:
        return int(os.environ.get("SLURM_RESTART_COUNT", "0")) >= 1
    except (TypeError, ValueError):
        return False


def _now_parts():
    now = datetime.now()
    return now.strftime("%Y%m%d"), now.strftime("%H%M%S")


class _PreemptCallback(Callback):
    """Checkpoints + requeues when the manager's SIGTERM flag is set.

    Runs the heavy work in the training loop (``on_train_batch_end`` /
    ``on_train_epoch_end``), not the signal handler — async-signal-safe.
    """

    def __init__(self, manager: "Manager"):
        self.manager = manager

    def _maybe_preempt(self, trainer, module):
        if self.manager._preempt_requested:
            logging.warning("⏸ preemption flagged — checkpoint + requeue")
            self.manager.checkpoint()
            self.manager.requeue()
            sys.exit(1)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self._maybe_preempt(trainer, module)

    def on_train_epoch_end(self, trainer, module):
        self._maybe_preempt(trainer, module)


class Manager:
    """Orchestrates a JAX training run: run-dir, checkpointing, SLURM preempt/resume.

    Args:
        trainer: A :class:`stable_pretraining.jax.Trainer`.
        module: The :class:`stable_pretraining.jax.Module` to train.
        train_loader: Iterable of training batches.
        val_loader: Optional iterable of validation batches.
        run_dir: Override the resolved run directory (skips cache-dir logic).
        save_every_n_epochs: Periodic checkpoint cadence (``last.msgpack`` is also
            written at fit end and on preemption). Default ``1``.

    Attributes:
        run_dir: Resolved run directory (or ``None`` if no cache_dir configured).
        ckpt_path: ``run_dir/checkpoints/last.msgpack`` (or ``None``).
    """

    def __init__(
        self,
        trainer,
        module,
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        run_dir: Optional[str] = None,
        save_every_n_epochs: int = 1,
    ):
        self.trainer = trainer
        self.module = module
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_every_n_epochs = save_every_n_epochs
        self._preempt_requested = False

        self.run_dir = Path(run_dir) if run_dir else self._resolve_run_dir()
        self.ckpt_path = (
            self.run_dir / "checkpoints" / "last.msgpack" if self.run_dir else None
        )
        self._install_preempt_handler()

    # ------------------------------------------------------------------ #
    # Run-dir resolution (mirrors the torch Manager layout + slurm index)
    # ------------------------------------------------------------------ #
    def _resolve_run_dir(self) -> Optional[Path]:
        cfg = get_config()
        if cfg.cache_dir is None:
            logging.info("  cache_dir not configured — checkpointing disabled.")
            return None
        cache_dir = Path(os.path.expanduser(cfg.cache_dir)).resolve()
        slurm_key = _slurm_session_key()

        # Requeue: reuse the original run-dir recorded in the slurm index.
        if _is_slurm_requeue() and slurm_key is not None:
            idx = cache_dir / ".slurm_index" / slurm_key
            if idx.is_file():
                prior = Path(idx.read_text().strip())
                logging.info(f"  SLURM requeue → reusing run_dir {prior}")
                return prior
            logging.warning(
                f"! requeue but no index at {idx} — starting a fresh run_dir."
            )

        day, sec = _now_parts()
        run_dir = cache_dir / "runs" / day / sec / uuid.uuid4().hex[:12]
        run_dir.mkdir(parents=True, exist_ok=True)
        if slurm_key is not None:
            idx_dir = cache_dir / ".slurm_index"
            idx_dir.mkdir(parents=True, exist_ok=True)
            (idx_dir / slurm_key).write_text(str(run_dir))
            logging.info(f"  wrote slurm index {slurm_key} → {run_dir}")
        logging.info(f"  run_dir = {run_dir}")
        return run_dir

    # ------------------------------------------------------------------ #
    # Preemption
    # ------------------------------------------------------------------ #
    def _install_preempt_handler(self) -> None:
        def _handler(signum, frame):
            # Keep minimal: just flag; the loop does the checkpoint+requeue.
            self._preempt_requested = True

        signal.signal(signal.SIGTERM, _handler)
        if os.environ.get("SLURM_JOB_ID"):
            logging.info("  SIGTERM preempt handler installed (checkpoint + requeue).")

    def checkpoint(self) -> None:
        """Atomically write ``last.msgpack`` (module + optimizer + step)."""
        if self.ckpt_path is None:
            logging.warning("  no run_dir — cannot checkpoint on preemption.")
            return
        objects = {"module": self.module}
        if self.trainer.optimizer is not None:
            objects["optimizer"] = self.trainer.optimizer
        save(self.ckpt_path, step=self.trainer.global_step, **objects)
        logging.info(f"  ✓ checkpoint written: {self.ckpt_path}")

    def requeue(self) -> None:
        """Requeue the SLURM job (no-op off-SLURM)."""
        job = os.environ.get("SLURM_JOB_ID")
        if not job:
            return
        logging.warning(f"  scontrol requeue {job}")
        try:
            subprocess.run(["scontrol", "requeue", job], check=False)
        except FileNotFoundError:
            logging.error("  scontrol not found — cannot requeue.")

    def _resume_path(self) -> Optional[str]:
        if self.ckpt_path and self.ckpt_path.exists():
            logging.info(f"  resuming from {self.ckpt_path}")
            return str(self.ckpt_path)
        return None

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    def _inject_registry_logger(self) -> None:
        """Attach a RegistryLogger at the run-dir so ``spt registry`` sees this run.

        Reuses the exact same (framework-agnostic) RegistryLogger as the torch
        Manager — identical ``sidecar.json`` / ``summary.json`` / heartbeat — so
        ``spt registry ls/best/...`` works the same regardless of backend.
        """
        if self.run_dir is None:
            return
        from ..registry import RegistryLogger

        if any(type(lg).__name__ == "RegistryLogger" for lg in self.trainer.loggers):
            return
        self.trainer.loggers.insert(
            0, RegistryLogger(run_dir=self.run_dir, run_id=self.run_dir.name)
        )

    def __call__(self):
        """Run training: auto-resume if a checkpoint exists, checkpoint as we go."""
        if self.ckpt_path is not None:
            self._inject_registry_logger()
            self.trainer.callbacks.append(
                Checkpoint(str(self.ckpt_path), every_n_epochs=self.save_every_n_epochs)
            )
            self.trainer.callbacks.append(_PreemptCallback(self))
        self.trainer.fit(
            self.module,
            self.train_loader,
            self.val_loader,
            resume_from=self._resume_path(),
        )
        return self.trainer


__all__ = ["Manager"]
