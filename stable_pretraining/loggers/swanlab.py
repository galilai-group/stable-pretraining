# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Lightning logger backed by `SwanLab <https://github.com/SwanHubX/SwanLab>`_.

Thin subclass of SwanLab's built-in Lightning logger
(:class:`swanlab.integration.pytorch_lightning.SwanLabLogger`) that adds
two helpers used by :class:`~stable_pretraining.callbacks.SwanLabCheckpoint`
for seamless SLURM requeue:

* :meth:`SwanLabLogger.resume_info` — snapshot of run identity for the
  sidecar.
* :meth:`SwanLabLogger.set_resume` — configure the logger to resume a
  previous experiment on next ``swanlab.init()``.

Everything else — lazy init, hyperparam / metric / media logging,
``log_image`` / ``log_audio`` / ``log_text`` helpers, cloud / self-hosted
(via ``SWANLAB_API_HOST``) / offline modes — is handled by SwanLab's
upstream logger.

Example::

    from stable_pretraining.loggers import SwanLabLogger

    logger = SwanLabLogger(
        project="my-project",
        experiment_name="run-1",
        mode="cloud",           # or "offline"
    )

    trainer = pl.Trainer(logger=logger, ...)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from swanlab.integration.pytorch_lightning import (
        SwanLabLogger as _SwanLabLogger,
    )

    SWANLAB_AVAILABLE = True
except ImportError:
    _SwanLabLogger = None
    SWANLAB_AVAILABLE = False


if SWANLAB_AVAILABLE:

    class SwanLabLogger(_SwanLabLogger):
        """SwanLab Lightning logger with stable-pretraining resume helpers.

        All ``__init__`` kwargs are forwarded to the upstream
        :class:`swanlab.integration.pytorch_lightning.SwanLabLogger`:

        * ``project``, ``workspace``, ``experiment_name``, ``description``,
          ``logdir``, ``mode``, ``save_dir``, ``tags``, ``id``, ``group``,
          ``config``, ``resume``.
        * Any extra kwargs are passed to ``swanlab.init()``.

        Self-hosted SwanLab servers are configured via env vars
        (``SWANLAB_API_HOST``, ``SWANLAB_WEB_HOST``) — SwanLab's SDK picks
        them up automatically.
        """

        @property
        def resume_info(self) -> Dict[str, Any]:
            """Snapshot of state needed to resume this run after requeue.

            Prefers the live experiment's ``run_id`` when the experiment
            has been initialised; falls back to whatever ``id=`` was passed
            at construction.
            """
            live_id = None
            exp = getattr(self, "_experiment", None)
            if exp is not None:
                public = getattr(exp, "public", None)
                live_id = getattr(public, "run_id", None) if public else None
            init_cfg = getattr(self, "_swanlab_init", {}) or {}
            return {
                "project": getattr(self, "_project", init_cfg.get("project")),
                "experiment_name": init_cfg.get("experiment_name"),
                "group": init_cfg.get("group"),
                "id": live_id or init_cfg.get("id"),
            }

        def set_resume(self, id: str) -> None:
            """Configure this logger to resume a previous experiment.

            Mutates the underlying ``_swanlab_init`` dict so that the next
            call to ``swanlab.init()`` (triggered on first access to
            ``self.experiment``) picks up ``id=<id>`` with
            ``resume="must"``.  Call this *before* any logging happens.
            """
            init_cfg = getattr(self, "_swanlab_init", None)
            if init_cfg is None:
                # Upstream shape changed — store on the instance so that
                # users relying on set_resume() still see a well-defined
                # attribute for debugging.
                self._swanlab_init = {"id": id, "resume": "must"}
                return
            init_cfg["id"] = id
            init_cfg["resume"] = "must"

else:
    # Placeholder that raises on instantiation so imports don't break when
    # swanlab is missing — matches the pattern used by optional deps.
    class SwanLabLogger:  # type: ignore[no-redef]
        """Placeholder — install ``swanlab`` to use this logger."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "swanlab is required for SwanLabLogger but is not installed. "
                "Install it with: pip install swanlab"
            )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def find_swanlab_logger(trainer: Any) -> Optional[SwanLabLogger]:
    """Find the unique :class:`SwanLabLogger` among trainer loggers.

    Returns ``None`` if no SwanLabLogger is configured.

    Raises:
        RuntimeError: If more than one SwanLabLogger is attached.
    """
    if not SWANLAB_AVAILABLE:
        return None
    found = [lg for lg in trainer.loggers if isinstance(lg, SwanLabLogger)]
    if len(found) == 0:
        return None
    if len(found) > 1:
        raise RuntimeError(
            f"Found {len(found)} SwanLabLoggers attached to the Trainer. "
            "Only one is supported for run resume across requeues."
        )
    return found[0]
