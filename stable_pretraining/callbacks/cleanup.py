"""Callback to clean up local training artifacts after successful training.

Add this callback explicitly to your Trainer when you want automatic cleanup
of logs, checkpoints, and other artifacts after training completes successfully.
On failure, everything is kept for debugging.

Example::

    trainer = pl.Trainer(
        callbacks=[
            CleanUpCallback(
                keep_checkpoints=False,  # delete checkpoint files
                keep_logs=False,  # delete CSV/wandb local logs
                keep_hydra=False,  # delete .hydra/ and hydra.log
                keep_slurm=False,  # delete slurm-*.out/err
                keep_env_dump=False,  # delete environment.json etc.
                keep_callback_artifacts=False,  # delete LatentViz, Writer, HF export dirs
            )
        ]
    )
"""

import glob
import os
import shutil
from typing import List, Optional, Sequence, Tuple

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger

from .utils import log_header

try:
    from hydra.core.hydra_config import HydraConfig
except ImportError:
    HydraConfig = None


def _human_size(nbytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def _resolve_hydra_output_dir() -> Optional[str]:
    if HydraConfig is not None:
        try:
            return HydraConfig.get().runtime.output_dir
        except Exception:
            pass
    return None


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class CleanUpCallback(Callback):
    """Clean up local training artifacts after successful training.

    This callback should be added **explicitly** by the user — it is NOT
    included in the default callback set.  On successful training completion
    it removes the selected artifact categories.  If training fails (exception),
    nothing is deleted so you can debug.

    Args:
        keep_checkpoints: Keep checkpoint files. Default ``True``.
        keep_logs: Keep logger output dirs (CSV logs, wandb local, tensorboard).
            Default ``True``.
        keep_hydra: Keep Hydra artifacts (``.hydra/`` dir and ``hydra.log``).
            Default ``False``.
        keep_slurm: Keep SLURM log files (``slurm-*.out/err``).
            Default ``False``.
        keep_env_dump: Keep environment dump files (``environment.json``,
            ``requirements_frozen.txt``). Default ``False``.
        keep_callback_artifacts: Keep dirs produced by other callbacks
            (``LatentViz`` plots, ``OnlineWriter`` output, ``hf_exports/``).
            Default ``True``.
        slurm_patterns: Glob patterns for SLURM files.
        extra_patterns: Additional glob patterns to delete (relative to cwd).
        dry_run: If ``True``, only log what would be deleted.
    """

    # Sentinel to distinguish "not passed" from an explicit ``True``/``False``.
    _UNSET = object()

    def __init__(
        self,
        keep_checkpoints: bool = _UNSET,
        keep_logs: bool = _UNSET,
        keep_hydra: bool = _UNSET,
        keep_slurm: bool = _UNSET,
        keep_env_dump: bool = _UNSET,
        keep_callback_artifacts: bool = _UNSET,
        slurm_patterns: Optional[Sequence[str]] = None,
        extra_patterns: Optional[Sequence[str]] = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__()
        from .._config import get_config

        cfg_cleanup = get_config().cleanup
        self.keep_checkpoints = (
            keep_checkpoints
            if keep_checkpoints is not self._UNSET
            else cfg_cleanup["checkpoints"]
        )
        self.keep_logs = (
            keep_logs if keep_logs is not self._UNSET else cfg_cleanup["logs"]
        )
        self.keep_hydra = (
            keep_hydra if keep_hydra is not self._UNSET else cfg_cleanup["hydra"]
        )
        self.keep_slurm = (
            keep_slurm if keep_slurm is not self._UNSET else cfg_cleanup["slurm"]
        )
        self.keep_env_dump = (
            keep_env_dump
            if keep_env_dump is not self._UNSET
            else cfg_cleanup["env_dump"]
        )
        self.keep_callback_artifacts = (
            keep_callback_artifacts
            if keep_callback_artifacts is not self._UNSET
            else cfg_cleanup["callback_artifacts"]
        )
        self.slurm_patterns = list(slurm_patterns or ["slurm-*.out", "slurm-*.err"])
        self.extra_patterns = list(extra_patterns or [])
        self.dry_run = dry_run
        self._exception = False

    def _collect_targets(self, trainer: Trainer) -> List[Tuple[str, str]]:
        """Collect (category, path) pairs of artifacts to delete."""
        targets: List[Tuple[str, str]] = []

        # --- SLURM logs ---
        if not self.keep_slurm:
            search_dirs = [os.getcwd()]
            slurm_submit = os.environ.get("SLURM_SUBMIT_DIR")
            if slurm_submit and slurm_submit not in search_dirs:
                search_dirs.append(slurm_submit)
            for d in search_dirs:
                for pattern in self.slurm_patterns:
                    for f in glob.glob(os.path.join(d, pattern)):
                        if os.path.isfile(f):
                            targets.append(("slurm", f))

        # --- Hydra artifacts ---
        if not self.keep_hydra:
            hydra_dir = _resolve_hydra_output_dir()
            if hydra_dir:
                hydra_log = os.path.join(hydra_dir, "hydra.log")
                if os.path.isfile(hydra_log):
                    targets.append(("hydra", hydra_log))
                hydra_dot = os.path.join(hydra_dir, ".hydra")
                if os.path.isdir(hydra_dot):
                    targets.append(("hydra", hydra_dot))

        # --- Checkpoints ---
        if not self.keep_checkpoints:
            for cb in trainer.checkpoint_callbacks:
                ckpt_dir = getattr(cb, "dirpath", None)
                if ckpt_dir and os.path.isdir(ckpt_dir):
                    targets.append(("checkpoint", ckpt_dir))

        # --- Logger output dirs ---
        if not self.keep_logs:
            for lg in trainer.loggers:
                log_dir = getattr(lg, "log_dir", None) or getattr(lg, "save_dir", None)
                if log_dir and os.path.isdir(log_dir):
                    targets.append(("logs", log_dir))

        # --- Environment dump files ---
        if not self.keep_env_dump:
            root = trainer.default_root_dir
            for pattern in ["environment*.json", "requirements_frozen*.txt"]:
                for f in glob.glob(os.path.join(root, pattern)):
                    if os.path.isfile(f):
                        targets.append(("env_dump", f))

        # --- Callback-produced artifact dirs ---
        if not self.keep_callback_artifacts:
            for cb in trainer.callbacks:
                # LatentViz: saves to save_dir or latent_viz_{name}
                if hasattr(cb, "save_dir") and hasattr(cb, "name"):
                    d = cb.save_dir if cb.save_dir else f"latent_viz_{cb.name}"
                    if os.path.isdir(d):
                        targets.append(("callback", d))
                # OnlineWriter: saves to cb.path
                if hasattr(cb, "path") and hasattr(cb, "key"):
                    p = str(getattr(cb, "path", ""))
                    if p and os.path.isdir(p):
                        targets.append(("callback", p))
                # HuggingFaceCheckpointCallback: saves to cb.save_dir
                if type(cb).__name__ == "HuggingFaceCheckpointCallback":
                    d = str(getattr(cb, "save_dir", ""))
                    if d and os.path.isdir(d):
                        targets.append(("callback", d))

        # --- Extra patterns ---
        for pattern in self.extra_patterns:
            for f in glob.glob(pattern):
                targets.append(("extra", f))

        return targets

    @rank_zero_only
    def on_exception(self, trainer, pl_module, exception) -> None:
        self._exception = True
        logger.warning("! training failed, skipping cleanup")

    @rank_zero_only
    def on_fit_end(self, trainer: Trainer, pl_module) -> None:
        if self._exception:
            return

        targets = self._collect_targets(trainer)
        if not targets:
            logger.info("  no artifacts to clean up")
            return

        log_header("CleanUpCallback")

        total_bytes = 0
        deleted = 0
        for category, path in targets:
            size = (
                _dir_size(path)
                if os.path.isdir(path)
                else os.path.getsize(path)
                if os.path.exists(path)
                else 0
            )
            total_bytes += size

            if self.dry_run:
                logger.info(
                    f"  [dry-run] would delete {category}: {path} ({_human_size(size)})"
                )
                continue

            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                deleted += 1
                logger.info(f"  deleted {category}: {path} ({_human_size(size)})")
            except Exception as e:
                logger.warning(f"! failed to delete {path}: {e}")

        if self.dry_run:
            logger.info(
                f"  dry-run: would free {_human_size(total_bytes)} "
                f"across {len(targets)} item(s)"
            )
        else:
            logger.success(
                f"✓ deleted {deleted}/{len(targets)} item(s), "
                f"freed {_human_size(total_bytes)}"
            )
