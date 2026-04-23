# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Callback for persisting SwanLab run identity across checkpoint save/load.

Mirrors :class:`WandbCheckpoint` / :class:`TrackioCheckpoint` but for
SwanLab:

* **On save** — stores ``{"id": ..., "project": ..., ...}`` in the
  checkpoint dict and writes a ``swanlab_resume.json`` sidecar so the
  Manager can inject the experiment ID *before* ``swanlab.init()`` fires
  on the next SLURM job.
* **On load** — verifies the active SwanLab logger matches what the
  checkpoint expects.

Only performs I/O on ``trainer.is_global_zero`` to stay safe under DDP.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from .utils import log_header

_SWANLAB_RESUME_FILENAME = "swanlab_resume.json"


class SwanLabCheckpoint(Callback):
    """Persist the SwanLab experiment ID across checkpoint save/load for seamless requeue resume.

    On save:
        - Stores ``{"id": ..., "project": ..., "experiment_name": ...,
          "group": ...}`` in the checkpoint dict under the ``"swanlab"`` key.
        - Writes a ``swanlab_resume.json`` sidecar in the run directory (and
          optionally CWD) so the Manager can configure the logger's
          ``resume="must"`` *before* ``swanlab.init()`` fires on the next job.

    On load:
        - Verifies the active SwanLab logger matches the checkpoint.
    """

    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        log_header("SwanLabCheckpoint")

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        from ..loggers.swanlab import find_swanlab_logger

        swanlab_logger = find_swanlab_logger(trainer)
        if swanlab_logger is None:
            return

        resume_info = swanlab_logger.resume_info
        if resume_info.get("id") is None:
            # swanlab hasn't assigned an ID yet (experiment hasn't been
            # accessed).  Skip — nothing to resume.
            return

        checkpoint["swanlab"] = resume_info
        logging.info(f"  Saved swanlab resume info to checkpoint: {resume_info}")

        if trainer.is_global_zero:
            root = Path(trainer.default_root_dir)
            sidecar = root / _SWANLAB_RESUME_FILENAME
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(json.dumps(resume_info))
            logging.info(f"  Wrote {sidecar.resolve()}")

            from stable_pretraining._config import get_config

            if get_config().cache_dir is None:
                cwd_sidecar = Path(_SWANLAB_RESUME_FILENAME)
                if cwd_sidecar.resolve() != sidecar.resolve():
                    cwd_sidecar.write_text(json.dumps(resume_info))
                    logging.info(f"  Wrote {cwd_sidecar.resolve()} (compat)")

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        if "swanlab" not in checkpoint:
            return

        expected_id = checkpoint["swanlab"].get("id")
        if expected_id is None:
            return

        from ..loggers.swanlab import find_swanlab_logger

        swanlab_logger = find_swanlab_logger(trainer)
        if swanlab_logger is None:
            logging.warning(
                f"! Checkpoint contains swanlab experiment id '{expected_id}' but "
                "no SwanLabLogger is configured — swanlab resume will not happen."
            )
            return

        init_cfg = getattr(swanlab_logger, "_swanlab_init", {}) or {}
        configured_id = init_cfg.get("id")
        if configured_id == expected_id:
            logging.info(
                f"  SwanLab experiment id '{expected_id}' matches — "
                "resume is set up correctly."
            )
        else:
            logging.error(
                f"! SwanLab experiment id mismatch: checkpoint expects "
                f"'{expected_id}' but the logger is configured with "
                f"'{configured_id}'. The run may not resume correctly."
            )
