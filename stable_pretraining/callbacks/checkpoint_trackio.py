# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Callback for persisting Trackio run identity across checkpoint save/load.

Mirrors :class:`WandbCheckpoint` but for Trackio:

* **On save** — stores ``{"name": ..., "project": ..., "group": ...}`` in
  the checkpoint dict and writes a ``trackio_resume.json`` sidecar so the
  Manager can inject the run name *before* ``trackio.init()`` fires on the
  next SLURM job.
* **On load** — verifies the active Trackio run matches what the
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

_TRACKIO_RESUME_FILENAME = "trackio_resume.json"


class TrackioCheckpoint(Callback):
    """Persist the Trackio run name across checkpoint save/load for seamless requeue resume.

    On save:
        - Stores ``{"name": ..., "project": ..., "group": ...}`` in the
          checkpoint dict under the ``"trackio"`` key.
        - Writes a ``trackio_resume.json`` sidecar in the run directory (and
          optionally CWD) so the Manager can configure the logger's
          ``resume="must"`` *before* ``trackio.init()`` fires on the next job.

    On load:
        - Verifies the active Trackio logger matches the checkpoint.
    """

    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        log_header("TrackioCheckpoint")

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        from ..loggers.trackio import find_trackio_logger

        trackio_logger = find_trackio_logger(trainer)
        if trackio_logger is None:
            return

        resume_info = trackio_logger.resume_info
        # The name may still be None if trackio auto-generated one and we
        # haven't accessed the experiment yet.  Try to resolve it.
        if resume_info["name"] is None and trackio_logger._run is not None:
            resume_info["name"] = getattr(trackio_logger._run, "name", None)

        checkpoint["trackio"] = resume_info
        logging.info(f"  Saved trackio resume info to checkpoint: {resume_info}")

        if trainer.is_global_zero:
            root = Path(trainer.default_root_dir)
            sidecar = root / _TRACKIO_RESUME_FILENAME
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(json.dumps(resume_info))
            logging.info(f"  Wrote {sidecar.resolve()}")

            from stable_pretraining._config import get_config

            if get_config().cache_dir is None:
                cwd_sidecar = Path(_TRACKIO_RESUME_FILENAME)
                if cwd_sidecar.resolve() != sidecar.resolve():
                    cwd_sidecar.write_text(json.dumps(resume_info))
                    logging.info(f"  Wrote {cwd_sidecar.resolve()} (compat)")

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        if "trackio" not in checkpoint:
            return

        expected_name = checkpoint["trackio"].get("name")
        if expected_name is None:
            return

        from ..loggers.trackio import find_trackio_logger

        trackio_logger = find_trackio_logger(trainer)
        if trackio_logger is None:
            logging.warning(
                f"! Checkpoint contains trackio run name '{expected_name}' but "
                "no TrackioLogger is configured — trackio resume will not happen."
            )
            return

        configured_name = trackio_logger._name
        if configured_name == expected_name:
            logging.info(
                f"  Trackio run name '{expected_name}' matches — "
                "resume is set up correctly."
            )
        else:
            logging.error(
                f"! Trackio run name mismatch: checkpoint expects "
                f"'{expected_name}' but the logger is configured with "
                f"'{configured_name}'. The run may not resume correctly."
            )
