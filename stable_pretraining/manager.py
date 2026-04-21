import copy
import inspect
import json
import os
import signal
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import hydra
import lightning
import lightning as pl
import pandas as pd
import submitit
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf

from . import WANDB_AVAILABLE
from ._config import get_config

if WANDB_AVAILABLE:
    import wandb
else:
    wandb = None

from .callbacks.checkpoint_sklearn import find_wandb_logger, _WANDB_RESUME_FILENAME
from .callbacks.checkpoint_trackio import _TRACKIO_RESUME_FILENAME
from .callbacks.checkpoint_swanlab import _SWANLAB_RESUME_FILENAME
from .loggers.trackio import find_trackio_logger
from .loggers.swanlab import find_swanlab_logger
from .utils import get_required_fn_parameters
from stable_pretraining.callbacks.utils import log_header
from stable_pretraining.utils.error_handling import catch_errors_class


def print_logger_info(logger):
    if isinstance(logger, lightning.pytorch.loggers.logger.DummyLogger):
        log_header("DummyLogger")

    elif isinstance(logger, lightning.pytorch.loggers.tensorboard.TensorBoardLogger):
        log_header("TensorBoardLogger")
        logging.info(f"  root_dir: {logger.root_dir}")
        logging.info(f"  save_dir: {logger.save_dir}")
        logging.info(f"  log_dir: {logger.log_dir}")

    elif isinstance(logger, lightning.pytorch.loggers.csv_logs.CSVLogger):
        log_header("CSVLogger")
        logging.info(f"  root_dir: {logger.root_dir}")
        logging.info(f"  save_dir: {logger.save_dir}")
        logging.info(f"  log_dir: {logger.log_dir}")

    elif isinstance(logger, lightning.pytorch.loggers.wandb.WandbLogger):
        log_header("WandbLogger")
        logging.info(f"  init: {logger._wandb_init}")

    elif logger is None:
        logging.warning("! No logger used!")
    else:
        # Check for known loggers without importing at module level
        cls_name = type(logger).__name__
        if cls_name == "RegistryLogger":
            log_header("RegistryLogger")
            logging.info(f"  db_path: {logger._db.db_path}")
            logging.info(f"  run_id:  {logger.version}")
            if logger._tags:
                logging.info(f"  tags:    {logger._tags}")
        elif cls_name == "TrackioLogger":
            log_header("TrackioLogger")
            logging.info(f"  project: {logger._project}")
            logging.info(f"  name:    {logger._name}")
            if logger._group:
                logging.info(f"  group:   {logger._group}")
            logging.info(f"  resume:  {logger._resume}")
        elif cls_name == "SwanLabLogger":
            log_header("SwanLabLogger")
            init_cfg = getattr(logger, "_swanlab_init", {}) or {}
            logging.info(f"  project:         {init_cfg.get('project')}")
            logging.info(f"  experiment_name: {init_cfg.get('experiment_name')}")
            if init_cfg.get("group"):
                logging.info(f"  group:           {init_cfg.get('group')}")
            if init_cfg.get("id"):
                logging.info(f"  id:              {init_cfg.get('id')}")
            if init_cfg.get("mode"):
                logging.info(f"  mode:            {init_cfg.get('mode')}")
        else:
            logging.warning("! Unrecognized logger!")


def print_signal_info():
    log_header("SignalHandlers")
    logging.info(f"  SIGUSR1: `{signal.getsignal(signal.SIGUSR1)}`")
    logging.info(f"  SIGUSR2: `{signal.getsignal(signal.SIGUSR2)}`")
    logging.info(f"  SIGCONT: `{signal.getsignal(signal.SIGCONT)}`")
    logging.info(f"  SIGTERM: `{signal.getsignal(signal.SIGTERM)}`")


_RUN_META_FILENAME = "run_meta.json"


def _generate_run_id() -> str:
    """Generate a deterministic run ID that all ranks in the same job agree on.

    Priority:
        1. ``SLURM_JOB_ID`` (+ ``SLURM_ARRAY_TASK_ID`` if array job) — same
           for every rank in the same SLURM job/task.
        2. ``TORCHELASTIC_RUN_ID`` — same for every rank under ``torchrun``.
        3. Random ``uuid4`` hex (12 chars) — single-process fallback.
    """
    slurm_job = os.environ.get("SLURM_JOB_ID")
    if slurm_job is not None:
        array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if array_task is not None:
            return f"{slurm_job}_{array_task}"
        return slurm_job
    elastic_run = os.environ.get("TORCHELASTIC_RUN_ID")
    if elastic_run is not None:
        return elastic_run
    return uuid.uuid4().hex[:12]


class _RunDirCallback(Callback):
    """Internal callback that persists the run directory path inside every checkpoint."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["spt_run_dir"] = self.run_dir


@catch_errors_class()
class Manager(submitit.helpers.Checkpointable):
    """Manages training with logging, scheduling, and checkpointing support.

    Args:
        trainer (Union[dict, DictConfig, pl.Trainer]): PyTorch Lightning trainer configuration or instance.
        module (Union[dict, DictConfig, pl.LightningModule]): Lightning module configuration or instance.
        data (Union[dict, DictConfig, pl.LightningDataModule]): Data module configuration or instance.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        ckpt_path (str, optional): Path to checkpoint for resuming training. Defaults to "last".
        resume_weights_only (bool | None, optional): Forwarded to ``Trainer.fit(weights_only=...)``
            when supported by the installed Lightning version. Defaults to ``False``.
        compile (bool, optional): Should we compile the given module. Defaults to False.
    """

    def __init__(
        self,
        trainer: Union[dict, DictConfig, pl.Trainer],
        module: Union[dict, DictConfig, pl.LightningModule],
        data: Union[dict, DictConfig, pl.LightningDataModule],
        seed: int = None,
        ckpt_path: str = None,
        resume_weights_only: Optional[bool] = False,
        compile: bool = False,
    ):
        if seed is None:
            logging.warning(
                "User didn't specify seed, runs won't be exactly reproducible!"
            )
        self.compile = compile
        self._register_trainer(trainer)
        self._register_module(module)
        self._register_data(data)

        self.seed = seed
        if ckpt_path is not None:
            ckpt_path = Path(ckpt_path).with_suffix(".ckpt").resolve()
        self.ckpt_path = ckpt_path
        self.resume_weights_only = resume_weights_only

    def _maybe_restore_wandb_run_id(self):
        """Inject a previous wandb run ID into the logger BEFORE wandb.init() fires.

        Reads the sidecar ``wandb_resume.json`` written by :class:`WandbCheckpoint`
        and, if the checkpoint file also exists, sets ``_wandb_init["id"]`` on the
        WandbLogger so that ``wandb.init()`` resumes the correct run instead of
        creating (and later deleting) a throwaway one.

        Must be called after the Trainer is created but before anything accesses
        ``trainer.logger.experiment``.
        """
        wandb_logger = find_wandb_logger(self._trainer)
        if wandb_logger is None:
            return

        # Only attempt resume if there's evidence of a previous run.
        # In cache_dir mode, the run_dir sidecar is sufficient (ckpt_path may be None).
        # In legacy mode, we need ckpt_path to exist on disk.
        has_run_dir = hasattr(self, "_run_dir")
        has_ckpt = self.ckpt_path is not None and self.ckpt_path.is_file()
        if not has_run_dir and not has_ckpt:
            return

        # Check run_dir first (cache_dir mode), then CWD (legacy)
        sidecar = None
        if hasattr(self, "_run_dir"):
            candidate = self._run_dir / _WANDB_RESUME_FILENAME
            if candidate.is_file():
                sidecar = candidate
        if sidecar is None:
            candidate = Path(_WANDB_RESUME_FILENAME)
            if candidate.is_file():
                sidecar = candidate
        if sidecar is None:
            logging.debug("  No wandb_resume.json found, skipping run ID injection")
            return

        try:
            resume_info = json.loads(sidecar.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(
                f"! Failed to read {sidecar}: {e} — skipping run ID injection"
            )
            return

        run_id = resume_info.get("id")
        if not run_id:
            logging.warning("! wandb_resume.json has no 'id' — skipping")
            return

        # Validate project/entity match the current logger config
        saved_project = resume_info.get("project")
        saved_entity = resume_info.get("entity")
        current_project = wandb_logger._wandb_init.get("project")
        current_entity = wandb_logger._wandb_init.get("entity")

        if saved_project and current_project and saved_project != current_project:
            logging.error(
                f"! wandb_resume.json project '{saved_project}' does not match "
                f"current logger project '{current_project}'. "
                "Skipping run ID injection to avoid resuming into the wrong project."
            )
            return

        if saved_entity and current_entity and saved_entity != current_entity:
            logging.error(
                f"! wandb_resume.json entity '{saved_entity}' does not match "
                f"current logger entity '{current_entity}'. "
                "Skipping run ID injection to avoid resuming into the wrong entity."
            )
            return

        # Inject the run ID — wandb.init() hasn't been called yet
        wandb_logger._wandb_init["id"] = run_id
        wandb_logger._id = run_id
        log_header("WandbResume")
        logging.info(f"  Injected wandb run ID '{run_id}' from {sidecar}")
        logging.info(f"  project={saved_project}  entity={saved_entity}")

    def _maybe_restore_trackio_run(self):
        """Inject a previous Trackio run name into the logger BEFORE trackio.init().

        Reads the sidecar ``trackio_resume.json`` written by
        :class:`TrackioCheckpoint` and, if present, calls
        ``set_resume(name)`` on the :class:`TrackioLogger` so that
        ``trackio.init()`` resumes the correct run.

        Must be called after the Trainer is created but before anything
        accesses ``trainer.logger.experiment``.
        """
        trackio_logger = find_trackio_logger(self._trainer)
        if trackio_logger is None:
            return

        has_run_dir = hasattr(self, "_run_dir")
        has_ckpt = self.ckpt_path is not None and self.ckpt_path.is_file()
        if not has_run_dir and not has_ckpt:
            return

        sidecar = None
        if hasattr(self, "_run_dir"):
            candidate = self._run_dir / _TRACKIO_RESUME_FILENAME
            if candidate.is_file():
                sidecar = candidate
        if sidecar is None:
            candidate = Path(_TRACKIO_RESUME_FILENAME)
            if candidate.is_file():
                sidecar = candidate
        if sidecar is None:
            logging.debug("  No trackio_resume.json found, skipping run name injection")
            return

        try:
            resume_info = json.loads(sidecar.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(
                f"! Failed to read {sidecar}: {e} — skipping trackio resume"
            )
            return

        run_name = resume_info.get("name")
        if not run_name:
            logging.warning("! trackio_resume.json has no 'name' — skipping")
            return

        saved_project = resume_info.get("project")
        current_project = trackio_logger._project
        if saved_project and current_project and saved_project != current_project:
            logging.error(
                f"! trackio_resume.json project '{saved_project}' does not match "
                f"current logger project '{current_project}'. "
                "Skipping run name injection to avoid resuming into the wrong project."
            )
            return

        trackio_logger.set_resume(run_name)
        log_header("TrackioResume")
        logging.info(f"  Injected trackio run name '{run_name}' from {sidecar}")
        logging.info(f"  project={saved_project}")

    def _maybe_restore_swanlab_run(self):
        """Inject a previous SwanLab experiment ID into the logger BEFORE swanlab.init().

        Reads the sidecar ``swanlab_resume.json`` written by
        :class:`SwanLabCheckpoint` and, if present, calls
        ``set_resume(id)`` on the :class:`SwanLabLogger` so that
        ``swanlab.init()`` resumes the correct experiment.

        Must be called after the Trainer is created but before anything
        accesses ``trainer.logger.experiment``.
        """
        swanlab_logger = find_swanlab_logger(self._trainer)
        if swanlab_logger is None:
            return

        has_run_dir = hasattr(self, "_run_dir")
        has_ckpt = self.ckpt_path is not None and self.ckpt_path.is_file()
        if not has_run_dir and not has_ckpt:
            return

        sidecar = None
        if hasattr(self, "_run_dir"):
            candidate = self._run_dir / _SWANLAB_RESUME_FILENAME
            if candidate.is_file():
                sidecar = candidate
        if sidecar is None:
            candidate = Path(_SWANLAB_RESUME_FILENAME)
            if candidate.is_file():
                sidecar = candidate
        if sidecar is None:
            logging.debug(
                "  No swanlab_resume.json found, skipping experiment id injection"
            )
            return

        try:
            resume_info = json.loads(sidecar.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(
                f"! Failed to read {sidecar}: {e} — skipping swanlab resume"
            )
            return

        run_id = resume_info.get("id")
        if not run_id:
            logging.warning("! swanlab_resume.json has no 'id' — skipping")
            return

        saved_project = resume_info.get("project")
        current_project = swanlab_logger._project
        if saved_project and current_project and saved_project != current_project:
            logging.error(
                f"! swanlab_resume.json project '{saved_project}' does not match "
                f"current logger project '{current_project}'. "
                "Skipping id injection to avoid resuming into the wrong project."
            )
            return

        swanlab_logger.set_resume(run_id)
        log_header("SwanLabResume")
        logging.info(f"  Injected swanlab experiment id '{run_id}' from {sidecar}")
        logging.info(f"  project={saved_project}")

    # -- cache_dir / run_dir ----------------------------------------------------

    def _resolve_run_dir(self) -> Optional[Path]:
        """Compute the run directory under ``cache_dir``.

        Layout::

            {cache_dir}/runs/{YYYYMMDD}/{HHMMSS}/{run_id}/

        On requeue (checkpoint with ``run_meta.json`` sidecar exists), the
        previous run directory is reused so that the same job continues
        writing to the same location.

        Returns ``None`` when cache_dir is not configured.
        """
        cfg = get_config()
        if cfg.cache_dir is None:
            return None

        cache_dir = Path(os.path.expanduser(cfg.cache_dir)).resolve()

        # Try to restore from a previous run (requeue)
        restored = self._try_restore_run_dir(cache_dir)
        if restored is not None:
            self._run_dir = restored
            self._run_id = restored.name
            log_header("RunDirectory (restored)")
            logging.info(f"  run_dir: {self._run_dir}")
            logging.info(f"  run_id:  {self._run_id}")
            return self._run_dir

        # Fresh run
        now = datetime.now()
        run_id = _generate_run_id()
        run_dir = (
            cache_dir
            / "runs"
            / now.strftime("%Y%m%d")
            / now.strftime("%H%M%S")
            / run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write sidecar so requeue can find this directory later
        meta = {"run_dir": str(run_dir), "run_id": run_id}
        (run_dir / _RUN_META_FILENAME).write_text(json.dumps(meta))

        self._run_dir = run_dir
        self._run_id = run_id
        log_header("RunDirectory")
        logging.info(f"  run_dir: {self._run_dir}")
        logging.info(f"  run_id:  {self._run_id}")
        return self._run_dir

    def _try_restore_run_dir(self, cache_dir: Path) -> Optional[Path]:
        """Attempt to find a previous run directory for this job.

        Checks two strategies:
            1. Sidecar next to ``ckpt_path`` (explicit checkpoint from user).
            2. Glob for the deterministic run_id in the cache_dir (requeue
               with SLURM_JOB_ID / TORCHELASTIC_RUN_ID — no ckpt_path needed).
        """
        # Strategy 1: sidecar next to ckpt_path
        if self.ckpt_path is not None and self.ckpt_path.is_file():
            meta_path = self.ckpt_path.parent / _RUN_META_FILENAME
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text())
                    run_dir = Path(meta["run_dir"])
                    if run_dir.is_dir():
                        return run_dir
                except Exception as exc:
                    logging.warning(f"! Could not read {meta_path}: {exc}")

        # Strategy 2: search by deterministic run_id (SLURM/torchrun requeue)
        run_id = _generate_run_id()
        matches = sorted(cache_dir.glob(f"runs/*/*/{run_id}"))
        if matches:
            # Take the most recent (last sorted by date/time path)
            return matches[-1]

        return None

    def _inject_run_dir_into_trainer_config(self, run_dir: Path) -> None:
        """Set ``default_root_dir`` in the trainer config before instantiation.

        This is the public Trainer API — Lightning will propagate this to all
        loggers and callbacks that rely on it (CSVLogger, TensorBoardLogger,
        ModelCheckpoint without explicit ``dirpath``, etc.).

        If the trainer is already a ``pl.Trainer`` instance (pre-built by the
        user), we warn instead of hacking private attributes.
        """
        if isinstance(self.trainer, (DictConfig, dict)):
            if OmegaConf.is_missing(self.trainer, "default_root_dir"):
                pass  # replace the MISSING sentinel
            elif "default_root_dir" in self.trainer:
                logging.warning(
                    f"! Overriding trainer.default_root_dir "
                    f"({self.trainer.default_root_dir}) with cache_dir run_dir: {run_dir}"
                )
            self.trainer.default_root_dir = str(run_dir)
        elif isinstance(self.trainer, pl.Trainer):
            logging.warning(
                "! cache_dir is set but the Trainer was passed as an already-"
                "instantiated object. Cannot override default_root_dir cleanly. "
                "Consider passing the trainer as a config dict instead."
            )

    def _resolve_load_path(self, run_dir: Path) -> Optional[str]:
        """Determine what checkpoint to pass to ``trainer.fit(ckpt_path=...)``.

        Priority:
            1. User's explicit ``ckpt_path`` (if it exists on disk).
            2. ``{run_dir}/checkpoints/last.ckpt`` (requeue from cache_dir).
            3. ``None`` (fresh run).

        This is purely the *load* path.  Where new checkpoints are *saved*
        is controlled separately by ``_configure_cache_dir_checkpointing``.
        """
        # 1. User explicitly provided a checkpoint
        if self.ckpt_path is not None:
            if self.ckpt_path.is_file():
                logging.info(f"  Load checkpoint (user): {self.ckpt_path}")
                return str(self.ckpt_path)
            logging.warning(
                f"  {self.ckpt_path} specified but does not exist, "
                "falling back to auto-detection"
            )

        # 2. Auto-detect requeue checkpoint in run_dir
        auto_ckpt = run_dir / "checkpoints" / "last.ckpt"
        if auto_ckpt.is_file():
            logging.info(f"  Load checkpoint (requeue): {auto_ckpt}")
            return str(auto_ckpt)

        # 3. Fresh run
        return None

    def _configure_cache_dir_checkpointing(self) -> None:
        """Ensure all checkpoints are saved into ``run_dir/checkpoints/``.

        When ``cache_dir`` is active:
        1. Every user ``ModelCheckpoint`` is redirected to
           ``run_dir/checkpoints/`` (preserving filename/monitor/etc.).
        2. A **requeue checkpoint** (``last.ckpt``, saved every epoch) is
           always added so that SLURM preemption recovery works even if the
           user's callbacks only save "best" models.
        """
        run_dir = self._run_dir
        save_dir = run_dir / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)

        log_header("CacheDirCheckpointing")
        logging.info(f"  Saving checkpoints to: {save_dir}")

        # Redirect every existing ModelCheckpoint to our save_dir
        for cb in self._trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                old_dir = cb.dirpath
                if (
                    old_dir is not None
                    and Path(old_dir).resolve() != save_dir.resolve()
                ):
                    logging.warning(
                        f"  Redirecting ModelCheckpoint from '{old_dir}' "
                        f"to '{save_dir}' (cache_dir is active)"
                    )
                cb.dirpath = str(save_dir)

        # Add a requeue checkpoint (last.ckpt) so preemption recovery works
        # regardless of what the user's callbacks save.  Can be disabled via
        # spt.set(requeue_checkpoint=False) to save time/disk.
        cfg = get_config()
        if cfg.requeue_checkpoint:
            requeue_saver = ModelCheckpoint(
                dirpath=str(save_dir),
                filename="last",
                save_last=False,
                save_on_train_epoch_end=True,
                verbose=True,
                enable_version_counter=False,
            )
            self._trainer.callbacks.append(requeue_saver)
            logging.info("  Added requeue checkpoint (filename='last')")
        else:
            logging.info(
                "  Requeue checkpoint disabled (spt.set(requeue_checkpoint=False))"
            )

    @staticmethod
    def _warn_hydra_conflicts() -> None:
        """Emit warnings when Hydra settings may conflict with the run directory."""
        try:
            from hydra.core.hydra_config import HydraConfig

            if not HydraConfig.initialized():
                return
            hcfg = HydraConfig.get()
            if getattr(hcfg.job, "chdir", False):
                logging.warning(
                    "! Hydra job.chdir=True detected. stable_pretraining's "
                    "cache_dir overrides output paths — Hydra's chdir is "
                    "redundant and may cause confusion."
                )
            # run.dir / sweep.dir
            try:
                run_dir_cfg = hcfg.run.dir
                if run_dir_cfg:
                    logging.warning(
                        f"! Hydra run.dir='{run_dir_cfg}' will be ignored for "
                        "trainer outputs (cache_dir takes precedence)."
                    )
            except Exception:
                pass
            try:
                sweep_dir_cfg = hcfg.sweep.dir
                if sweep_dir_cfg:
                    logging.warning(
                        f"! Hydra sweep.dir='{sweep_dir_cfg}' will be ignored for "
                        "trainer outputs (cache_dir takes precedence)."
                    )
            except Exception:
                pass
        except Exception:
            pass  # Hydra not active

    def _inject_registry_logger(self) -> None:
        """Auto-add :class:`RegistryLogger`.

        ``RegistryLogger`` is a :class:`~lightning.pytorch.loggers.CSVLogger`
        subclass — a single logger captures per-step CSV metrics and
        writes a ``sidecar.json`` + ``heartbeat`` file for fast querying
        via ``spt registry …``.  Works with or without ``cache_dir``:

        * **With ``cache_dir``**: run writes under
          ``{cache_dir}/runs/YYYYMMDD/HHMMSS/{run_id}/``.
        * **Without ``cache_dir``**: falls back to the Trainer's
          ``default_root_dir``.

        Can be disabled via ``spt.set(default_loggers={"registry": False})``.
        If a sibling :class:`CSVLogger` is already present, the
        ``RegistryLogger`` replaces it to avoid two writers on the same
        ``metrics.csv``.
        """
        cfg = get_config()
        if not cfg.default_loggers.get("registry", True):
            return

        from .registry.logger import RegistryLogger

        # Resolve run_dir + run_id.  Manager._resolve_run_dir already
        # populated these when cache_dir is set.
        if hasattr(self, "_run_dir") and self._run_dir is not None:
            run_dir = str(self._run_dir)
            run_id = self._run_id
        else:
            run_dir = str(Path(self._trainer.default_root_dir).resolve())
            run_id = _generate_run_id()

        # Drop any pre-existing CSVLogger — RegistryLogger *is* a
        # CSVLogger and would otherwise race on metrics.csv.
        self._trainer.loggers = [
            lgr
            for lgr in self._trainer.loggers
            if not (
                isinstance(lgr, lightning.pytorch.loggers.CSVLogger)
                and not isinstance(lgr, RegistryLogger)
            )
        ]

        registry_logger = RegistryLogger(run_dir=run_dir, run_id=run_id)
        self._trainer.loggers.append(registry_logger)

        log_header("RegistryLogger")
        logging.info(f"  run_dir: {registry_logger.run_dir}")
        logging.info(f"  run_id:  {registry_logger.run_id}")
        if registry_logger._tags:
            logging.info(f"  tags:    {registry_logger._tags}")

    def _flatten_hydra_config(self) -> dict:
        """Build a flat dot-separated dict from the raw Hydra configs.

        Collects ``trainer``, ``module``, and ``data`` DictConfigs, flattens
        them with ``pd.json_normalize``, and recursively expands lists.
        Returns an empty dict when everything is already instantiated.
        """
        config = {}
        if isinstance(self.trainer, (dict, DictConfig)):
            config["trainer"] = OmegaConf.to_container(self.trainer, resolve=True)
        if isinstance(self.module, (dict, DictConfig)):
            config["module"] = OmegaConf.to_container(self.module, resolve=True)
        if isinstance(self.data, (dict, DictConfig)):
            config["data"] = OmegaConf.to_container(self.data, resolve=True)
        if not config:
            return {}

        config = pd.json_normalize(config, sep=".").to_dict(orient="records")[0]
        while True:
            changed = False
            for k in list(config.keys()):
                if isinstance(config[k], list):
                    changed = True
                    for i, v in enumerate(config[k]):
                        config[f"{k}.{i}"] = v
                    del config[k]
            if changed:
                config = pd.json_normalize(config, sep=".").to_dict(orient="records")[0]
            else:
                break
        return config

    def _inject_hydra_hparams(self) -> None:
        """Inject the full flattened Hydra config into the module's hparams.

        Called right before ``trainer.fit()`` so that Lightning's built-in
        ``_log_hyperparams`` sends the config to **all** loggers (wandb,
        CSV, TensorBoard, registry, etc.) automatically — no per-logger
        special-casing required.
        """
        flat = self._flatten_hydra_config()
        if not flat:
            return
        module = self.instantiated_module
        module.save_hyperparameters(flat)
        log_header("HydraHparams")
        logging.info(f"  Injected {len(flat)} config keys into module.hparams")

    @rank_zero_only
    def init_and_sync_wandb(self):
        """Handles some utilities for WandB."""
        wandb_logger = find_wandb_logger(self._trainer)
        if wandb_logger is None:
            return
        log_header("Wandb")
        exp = wandb_logger.experiment

        if exp.offline:
            previous_run = self._wandb_previous_dir(wandb_logger)
            logging.info(f"  Found a previous run ({previous_run}), reusing config")
            with open(previous_run / "files/wandb-config.json", "r") as f:
                last_config = json.load(f)
            # at most last_config has an extra `ckpt_path`
            exp.config.update(last_config)
            logging.info("  reloaded!")
        elif WANDB_AVAILABLE and wandb.run and len(wandb.config.keys()):
            logging.info("  a Wandb config is provided, not uploading Hydra's:")
        else:
            logging.info("  Wandb's config is empty, trying to use Hydra's")
            config = self._flatten_hydra_config()
            if not config:
                logging.info(
                    "  Everything already instantiated, nothing is added to config!"
                )
                return
            logging.info(f"  Final Hydra's config has {len(config)} items")
            if WANDB_AVAILABLE and wandb.run:
                wandb.config.update(config)

    @property
    def instantiated_module(self):
        if not isinstance(self.module, pl.LightningModule):
            logging.info("  instantiating pl_module...")
            # with self._trainer.init_module():
            self._instantiated_module = hydra.utils.instantiate(
                self.module, _convert_="object"
            )
            logging.success("✓ module instantiated")
        else:
            self._instantiated_module = self.module
        return self._instantiated_module

    @property
    def instantiated_data(self):
        if not isinstance(self.data, pl.LightningDataModule):
            self._instantiated_data = hydra.utils.instantiate(
                self.data, _convert_="object", _recursive_=False
            )
            logging.success("✓ data instantiated")
        else:
            self._instantiated_data = self.data
        return self._instantiated_data

    def __call__(self):
        log_header("WorkingDirectory")
        logging.info(f"  cwd: {Path().resolve()}")
        log_header("Seed")
        logging.info(f"  seed: {self.seed}")
        pl.seed_everything(self.seed, workers=True)

        # --- cache_dir: resolve run directory and inject into trainer config ---
        run_dir = self._resolve_run_dir()
        if run_dir is not None:
            self._inject_run_dir_into_trainer_config(run_dir)
            self._warn_hydra_conflicts()

        if isinstance(self.trainer, pl.Trainer):
            self._trainer = self.trainer
        else:
            if "callbacks" in self.trainer:
                logging.info("  instantiating callbacks...")
                callbacks = hydra.utils.instantiate(
                    self.trainer.callbacks, _convert_="object"
                )
                for i, callback in enumerate(callbacks):
                    if not callable(callback):
                        continue
                    assert ["module"] == get_required_fn_parameters(callback)
                    callbacks[i] = callback(module=self.instantiated_module)
                logging.success("✓ callbacks instantiated")
                del self.trainer.callbacks

            else:
                callbacks = []

            # we use the following partial to give our init callbacks manually since otherwise
            # hydra instantiate throws an error
            self._trainer = hydra.utils.instantiate(
                self.trainer, _convert_="object", _partial_=True
            )
            self._trainer = self._trainer(callbacks=callbacks)
            if not isinstance(self._trainer, pl.Trainer):
                raise ValueError("`trainer` should be a Trainer")
            logging.success("✓ trainer instantiated")

        # Persist run_dir in every checkpoint so requeue can restore it
        if run_dir is not None:
            self._trainer.callbacks.append(_RunDirCallback(str(run_dir)))

        # Always inject RegistryLogger + CSVLogger (works with or without cache_dir)
        self._inject_registry_logger()

        # Auto-detect TeacherStudentWrapper and add callback if needed
        # This runs AFTER trainer is set up, regardless of how it was created
        from .callbacks.teacher_student import TeacherStudentCallback

        needs_teacher_student = False
        for module in self.instantiated_module.modules():
            if hasattr(module, "update_teacher") and hasattr(module, "teacher"):
                needs_teacher_student = True
                break

        if needs_teacher_student:
            # Check if TeacherStudentCallback is already in the list
            has_ts_callback = any(
                isinstance(cb, TeacherStudentCallback) for cb in self._trainer.callbacks
            )
            if not has_ts_callback:
                logging.success(
                    "✓ Auto-detected TeacherStudentWrapper, adding TeacherStudentCallback"
                )
                self._trainer.callbacks.append(TeacherStudentCallback())

        self._maybe_restore_wandb_run_id()
        self._maybe_restore_trackio_run()
        self._maybe_restore_swanlab_run()
        self.init_and_sync_wandb()
        print_logger_info(self._trainer.logger)
        print_signal_info()

        log_header("Callbacks")
        logging.info(f"  count: {len(self._trainer.callbacks)}")

        # --- Checkpointing setup (load vs save are separate concerns) ---
        if run_dir is not None:
            # cache_dir mode: save always goes to run_dir/checkpoints/,
            # load is resolved separately (user ckpt_path or requeue auto-detect)
            self._configure_cache_dir_checkpointing()
            ckpt_path = self._resolve_load_path(run_dir)
        else:
            # Legacy mode: ckpt_path controls both load and save location
            if "SLURM_JOB_ID" in os.environ and self.ckpt_path is None:
                logging.warning(
                    "Using SLURM but no ckpt_path, if requeued it will start "
                    "from scratch. Consider using spt.set(cache_dir=...) or "
                    "passing a value to the Manager's `ckpt_path`."
                )
            else:
                self._configure_checkpointing()

            if self.ckpt_path is not None and self.ckpt_path.is_file():
                ckpt_path = str(self.ckpt_path)
            elif self.ckpt_path is not None and not self.ckpt_path.is_file():
                logging.warning(
                    f"{self.ckpt_path} specified, but does not exist, using None for now!"
                )
                ckpt_path = None
            else:
                ckpt_path = None

        if self.compile:
            logging.warning("Compiling module!")
            self.instantiated_module.compile()

        fit_kwargs = {
            "datamodule": self.instantiated_data,
            "ckpt_path": ckpt_path,
        }
        if "weights_only" in inspect.signature(self._trainer.fit).parameters:
            fit_kwargs["weights_only"] = self.resume_weights_only
        elif self.resume_weights_only is not None:
            logging.warning(
                "Installed Lightning Trainer.fit does not accept `weights_only`; "
                f"ignoring manager.resume_weights_only={self.resume_weights_only}."
            )

        # Inject the full flattened Hydra config into the module's hparams
        # so Lightning's _log_hyperparams sends it to ALL loggers automatically
        # (wandb, CSV, TensorBoard, registry, etc.)
        self._inject_hydra_hparams()

        log_header("TrainerFit")
        logging.info(f"  ckpt_path: {ckpt_path}")
        logging.info(f"  resume_weights_only: {self.resume_weights_only}")
        self._trainer.fit(
            self.instantiated_module,
            **fit_kwargs,
        )
        self._dump_wandb_data()

    def validate(self):
        log_header("TrainerValidate")

        self._trainer.validate(
            self.instantiated_module, datamodule=self.instantiated_data
        )
        self._dump_wandb_data()

    def predict(self):
        log_header("TrainerPredict")

        self._trainer.predict(
            self.instantiated_module, datamodule=self.instantiated_data
        )
        self._dump_wandb_data()

    def test(self):
        log_header("TrainerTest")

        self._trainer.test(self.instantiated_module, datamodule=self.instantiated_data)
        self._dump_wandb_data()
        # wandb.finish()
        # logging.info(f"closing wandb 🗑️")
        # cfg = wandb.run.config.as_dict()
        # return cfg, module.info

    @rank_zero_only
    def _dump_wandb_data(self):
        if not WANDB_AVAILABLE or wandb.run is None or not wandb.run.offline:
            return

        # Print the summary
        logging.info("Summary:")
        summary_dict = wandb.run.summary._as_dict()
        logging.info(json.dumps(summary_dict, indent=2))
        fname = Path(wandb.run.dir) / "wandb-summary.json"
        if fname.is_file():
            raise RuntimeError(f"Summary file already exists {fname}")
        with open(fname, "w") as f:
            json.dump(summary_dict, f)
        logging.success(f"✓ Saved summary at {fname}")
        fname = Path(wandb.run.dir) / "wandb-config.json"
        if fname.is_file():
            raise RuntimeError(f"Config file already exists {fname}")
        with open(fname, "w") as f:
            json.dump(wandb.run.config.as_dict(), f)
        logging.success(f"✓ Saved config at {fname}")

    def _wandb_previous_dir(self, wandb_logger=None):
        if not WANDB_AVAILABLE or not wandb.run:
            return None
        # to remove the /files
        path = Path(wandb.run.dir).parent
        logging.info(f"  fetching previous Wandb runs from {path.parent}")
        # this will be of the form
        # offline-run-20250413_025716-p8117tgi
        runs = list(path.parent.glob(f"offline-run-*-{wandb.run.id}"))
        logging.info(f"  found {len(runs)} run(s):")
        runs = sorted(runs)
        for run in runs:
            logging.info(f"  {run.name}")
        assert runs[-1] == path
        if len(runs) == 1:
            return None
        return runs[-2]

    def save_checkpoint(
        self, path: str = None, upload_wandb: bool = False, verbose=True
    ):
        # TODO: figure out how to flush logging in subprocess
        if verbose:
            print("Entering checkpoint method", flush=True)
        if path is None:
            if hasattr(self, "_run_dir"):
                path = (self._run_dir / "checkpoints" / "checkpoint.ckpt").resolve()
            else:
                path = (Path() / "checkpoint.ckpt").resolve()
            if verbose:
                print(f"  saving checkpoint to local path {path} ...", flush=True)
        else:
            path = Path(path)
            if not path.parent.is_dir():
                path.parent.mkdir(parents=True)
            if verbose:
                print(f"  saving checkpoint to user's path {path} ...", flush=True)
        self._trainer.save_checkpoint(str(path))
        if verbose:
            print("  checkpoint saved", flush=True)
        if upload_wandb:
            self._upload_checkpoint_for_requeue(path)

    @rank_zero_only
    def _upload_checkpoint_for_requeue(self, ckpt_path):
        # if "ckpt_path" in wandb.run.config:
        #     ckpt_path = Path(wandb.run.config["ckpt_path"])
        #     print(f"\t● `ckpt_path` already in config, updating it!", flush=True)

        # else:
        #     ckpt_path = Path(wandb.run.dir) / "checkpoint.ckpt"
        #     print(f"\t● `ckpt_path` set to {ckpt_path}!", flush=True)

        if WANDB_AVAILABLE and wandb.run and not wandb.run.offline:
            print("  Wandb used and online:", flush=True)
            artifact = wandb.Artifact("requeue_checkpoint", "model")
            artifact.add_file(str(ckpt_path))
            artifact.ttl = timedelta(days=30)
            print("  artifact created", flush=True)
            wandb.run.log_artifact(artifact)
            print("  artifact logged", flush=True)
            ckpt_path.unlink()
            print("  local checkpoint deleted", flush=True)
        else:
            print("  Wandb used and offline:", flush=True)
            if WANDB_AVAILABLE and wandb.run:
                wandb.run.config.update({"ckpt_path": str(ckpt_path.resolve())})
            print("  `ckpt_path` added to Wandb config", flush=True)
        # for offline case
        self._dump_wandb_data()

    @staticmethod
    def _matches_template(ckpt_name: str, callback: ModelCheckpoint) -> bool:
        """Checks if a concrete checkpoint filename could have been generated by a callback's template.

        This is a heuristic that handles two cases:
        1.  Guaranteed Match: Checks if the name is 'last.ckpt' and the callback has `save_last=True`.
        2.  Template Match: Checks if all metric keys from the filename template (e.g., "epoch", "step")
            are present in the concrete checkpoint name (e.g., "epoch=10-step=5000.ckpt").

        Args:
            ckpt_name: The concrete filename (e.g., "last.ckpt", "epoch=1-step=100.ckpt").
            callback: The ModelCheckpoint callback instance.

        Returns:
            True if the name is a plausible match, False otherwise.
        """
        import re

        # Case 1: guaranteed `last.pt` case
        ckpt_stem = Path(ckpt_name).stem

        # the user can customize the name for the last checkpoint, so use the callback's property
        if ckpt_stem == callback.CHECKPOINT_NAME_LAST:
            # If the user's path is 'last.ckpt', the callback MUST have `save_last` enabled.
            return bool(callback.save_last)

        # Case 2: versioned `last.pt` case
        if (
            ckpt_stem.startswith(f"{callback.CHECKPOINT_NAME_LAST}-v")
            and callback.save_last
        ):
            return True

        # Case 3: templated filename case
        # Get the template from the callback, using the default if not set.
        template = (
            callback.filename or "{epoch}" + callback.CHECKPOINT_JOIN_CHAR + "{step}"
        )

        # Find all unique metric keys within the template string (e.g., from "{epoch}-{val_loss:.2f}")
        # This regex finds the name inside the curly braces, ignoring any formatting specs.
        template_keys = set(re.findall(r"\{([a-zA-Z0-9_/-]+)", template))

        # If the template has no keys, we can't perform a match, so we assume it's valid if the dir matches.
        if not template_keys:
            return True

        # Check if all keys from the template appear in the concrete filename in the format "key=...".
        # This is how PyTorch Lightning formats them by default.
        filename_keys = set()
        for part in ckpt_stem.split(callback.CHECKPOINT_JOIN_CHAR):
            if callback.CHECKPOINT_EQUALS_CHAR in part:
                filename_keys.add(part.split(callback.CHECKPOINT_EQUALS_CHAR)[0])

        return template_keys == filename_keys

    def _configure_checkpointing(self) -> None:
        """Analyzes user configuration for checkpointing and ensures it's set up correctly.

        This function is designed to handle four primary user scenarios by inspecting
        the state of the Trainer's callbacks and the `ckpt_path` provided to the Manager.
        It provides informative logs for each case and can add a `ModelCheckpoint`
        callback as a safety net if needed.

        Args:
            trainer: The PyTorch Lightning Trainer instance whose callbacks will be checked and
                    potentially modified.
            ckpt_path: The checkpoint path provided to the Manager, which indicates the user's
                    intent to resume from or save to a specific file.
        """
        log_header("CheckpointingSetup")
        trainer = self._trainer
        ckpt_path = self.ckpt_path

        # This flag checks if the user *explicitly* added any ModelCheckpoint
        # instance in their configuration. It runs before Lightning's potential
        # default callback is added.
        is_mc_explicitly_configured = any(
            isinstance(cb, pl.pytorch.callbacks.ModelCheckpoint)
            for cb in trainer.callbacks
        )

        # This flag checks if any of the *explicitly added* callbacks are configured
        # to save to the directory containing the specific path the Manager cares about.
        is_manager_path_handled_by_callback = False
        is_slurm_job = "SLURM_JOB_ID" in os.environ

        if is_mc_explicitly_configured and ckpt_path:
            for callback in trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    # manually resolve the directory path the callback will use.
                    resolved_dirpath = Path(
                        callback._ModelCheckpoint__resolve_ckpt_dir(trainer)
                    ).resolve()

                    if ckpt_path.parent == resolved_dirpath and self._matches_template(
                        ckpt_path.name, callback
                    ):
                        is_manager_path_handled_by_callback = True
                        break

        # Case 1: Intentional ckpt_path, correct callback passed in - do nothing
        if ckpt_path is not None and is_manager_path_handled_by_callback:
            logging.info(
                f"  Checkpoint: `manager.ckpt_path` ({ckpt_path}) is set and a matching `ModelCheckpoint` callback was found to be saving to the same directory."
            )
            if is_slurm_job:
                logging.info(
                    "  This setup is ready for SLURM preemption and requeueing."
                )

        # Case 2: Intentional ckpt_path, but no callback found - assume the user forgot and add a callback
        elif ckpt_path is not None and not is_manager_path_handled_by_callback:
            logging.warning(
                f"! Checkpoint mismatch: `manager.ckpt_path` ({ckpt_path}) was provided, but no matching `ModelCheckpoint` callback was found."
            )
            logging.warning(
                "! Automatically creating a `ModelCheckpoint` to save to the specified path to prevent data loss."
            )

            saver = ModelCheckpoint(
                dirpath=str(ckpt_path.parent),
                filename=ckpt_path.with_suffix("").name,
                save_last=False,  # be explicit, last.ckpt is a special case
                save_on_train_epoch_end=True,
                verbose=True,
                enable_version_counter=False,
            )
            trainer.callbacks.append(saver)
            logging.warning(
                "! Automatic `ModelCheckpoint` callback has been added to the trainer."
            )

        # Case 3: No checkpoint, but with ModelCheckpoint callback - assume we are training from scratch.
        elif ckpt_path is None and is_mc_explicitly_configured:
            logging.info(
                "  Checkpointing: A user-defined `ModelCheckpoint` callback was found. It will be used for saving checkpoints."
            )
            logging.info(
                "  The `Manager` will not manage resuming from a specific path as `manager.ckpt_path` was not provided."
            )
            if is_slurm_job:
                logging.warning(
                    "! SLURM WARNING: Since `manager.ckpt_path` is not set, this job will restart from scratch if requeued, even though checkpoints are being saved elsewhere."
                )

        # Case 4: No checkpoint and no ModelCheckpoint callback - assume we are training without saving checkpoints
        elif ckpt_path is None and not is_mc_explicitly_configured:
            logging.info(
                "  No Checkpointing: No `manager.ckpt_path` was provided and no `ModelCheckpoint` callback was found."
            )
            logging.info("  The model will not be saved during this run.")
            if is_slurm_job:
                logging.error(
                    "  CRITICAL SLURM WARNING: This job will lose all progress if it is preempted or requeued. It is highly recommended to configure checkpointing."
                )

    def _register_trainer(self, trainer):
        if type(trainer) is dict:
            trainer = OmegaConf.create(trainer)
        if type(trainer) is DictConfig:
            self.trainer: DictConfig = copy.deepcopy(trainer)
            logging.debug("  trainer config saved")
        elif isinstance(trainer, pl.Trainer):
            self.trainer = trainer
            logging.debug("  trainer already instantiated")
        else:
            raise ValueError(
                f"`trainer` must be a dict, DictConfig or pl.Trainer, not {type(trainer)}"
            )

    def _register_module(self, module):
        if type(module) is dict:
            module = OmegaConf.create(module)
        if type(module) is DictConfig:
            self.module: DictConfig = copy.deepcopy(module)
            logging.debug("  module config saved")
        elif isinstance(module, pl.LightningModule):
            self.module = module
            logging.debug("  module already instantiated")
        else:
            raise ValueError(
                f"`module` must be a dict, DictConfig or pl.LightningModule, not {type(module)}"
            )

    def _register_data(self, data):
        if type(data) is dict:
            data = OmegaConf.create(data)
        if type(data) is DictConfig:
            self.data: DictConfig = copy.deepcopy(data)
            logging.debug("  data config saved")
        elif isinstance(data, pl.LightningDataModule):
            self.data = data
            logging.debug("  data already instantiated")
        else:
            raise ValueError(
                f"`data` must be a dict, DictConfig or pl.LightningDataModule, not {type(data)}"
            )
