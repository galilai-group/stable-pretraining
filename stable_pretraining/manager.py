import copy
import inspect
import json
import os
import signal
import time
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
from stable_pretraining.utils.fsdp import (
    describe_fsdp_strategy,
    is_fsdp_strategy,
)


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
    """Always return a fresh uuid4 hex (12 chars).

    Run dirs are now uniquely identified by uuid regardless of execution
    context (interactive shell, batch, array task, torchrun, ...). SLURM
    preempt/requeue resume is handled separately by ``_resolve_run_dir``,
    which records ``SLURM_JOB_ID[_SLURM_ARRAY_TASK_ID] → run_dir`` in
    ``cache_dir/.slurm_index/`` and looks the value up when
    ``SLURM_RESTART_COUNT > 0`` on a re-run.

    This sidesteps the historical trap where every consecutive ``python``
    invocation inside ``srun --pty`` would land in the same run dir
    because ``SLURM_JOB_ID`` is shared across them.
    """
    return uuid.uuid4().hex[:12]


def _slurm_session_key() -> Optional[str]:
    """Stable per-SLURM-task key for requeue lookup, or ``None`` outside SLURM.

    Form: ``"<SLURM_JOB_ID>"`` or ``"<SLURM_JOB_ID>_<SLURM_ARRAY_TASK_ID>"``.
    Same value across preempt/requeue cycles (SLURM keeps job/task ids
    stable on requeue), so a requeued process can find the run_dir of
    the original invocation.
    """
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    return f"{job}_{task}" if task is not None else job


def _is_slurm_requeue() -> bool:
    """SLURM exports ``SLURM_RESTART_COUNT >= 1`` only on requeue.

    Interactive ``srun --pty`` reruns share ``SLURM_JOB_ID`` but never bump
    ``SLURM_RESTART_COUNT`` — so checking it lets us distinguish a real
    preempt-resume from an interactive re-invocation.
    """
    try:
        return int(os.environ.get("SLURM_RESTART_COUNT", "0")) >= 1
    except (TypeError, ValueError):
        return False


def _ddp_launch_key() -> Optional[str]:
    """Identifier shared by every rank in the same DDP launch, or ``None``.

    Used as the filename under ``{cache_dir}/.rank_handoff/`` so rank-0 can
    publish its chosen ``run_dir`` and other ranks can read the same value
    instead of each generating their own.

    Returns ``None`` for single-process invocations (no DDP env vars set) —
    in that case no handoff is needed.

    The key is intentionally identical for every rank in the same launch
    AND distinct between concurrent launches:

    * SLURM (batch / array) — keyed on ``SLURM_JOB_ID[_TASK_ID]``.
    * ``torchrun`` / torchelastic — keyed on ``TORCHELASTIC_RUN_ID``.
    * Local DDP via Lightning's ``SubprocessScriptLauncher`` — keyed on
      ``MASTER_ADDR:MASTER_PORT`` plus the launcher's process group id, so
      two parallel local-DDP launches on the same machine don't collide
      even if they happen to pick the same MASTER_PORT.
    """
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        return f"slurm-{job}_{task}" if task is not None else f"slurm-{job}"
    er = os.environ.get("TORCHELASTIC_RUN_ID")
    if er:
        return f"elastic-{er}"
    addr = os.environ.get("MASTER_ADDR")
    port = os.environ.get("MASTER_PORT")
    if addr and port:
        try:
            pgid = str(os.getpgid(0))
        except OSError:
            pgid = "nopgid"
        return f"local-{addr}-{port}-{pgid}"
    return None


# Rank-N waits up to this many seconds for rank-0 to publish run_dir before
# falling back to local resolution. Generous to absorb slow NFS mkdir + the
# actual time rank-0 spends resolving (requeue index lookup, mkdir, etc.).
# Override via env var ``SPT_RANK_HANDOFF_TIMEOUT_S`` if the cluster's NFS
# is unusually slow.
try:
    _RANK_HANDOFF_TIMEOUT_S = float(
        os.environ.get("SPT_RANK_HANDOFF_TIMEOUT_S", "60.0")
    )
except ValueError:
    _RANK_HANDOFF_TIMEOUT_S = 60.0
_RANK_HANDOFF_POLL_S = 0.05


class _RunDirCallback(Callback):
    """Internal callback that persists the run directory path inside every checkpoint.

    Also acts as a defensive *guard* on the SLURM index: at
    ``on_train_start`` (after the user's setup is done, before any
    training step) it verifies that
    ``<cache_dir>/.slurm_index/<SLURM_JOB_ID[_TASK_ID]>`` exists and
    points at this run's ``run_dir``. If it's missing or stale, the
    callback re-writes it (atomically) and logs loudly so it's obvious
    in the run output. This is a belt-and-braces check — every code
    path in :meth:`Manager._resolve_run_dir` already calls
    :meth:`Manager._write_slurm_index`, but a future regression in
    that wiring would otherwise only surface as a cryptic
    ``RuntimeError`` on the *next* requeue, hours into a sweep.
    """

    def __init__(self, run_dir: str, cache_dir: Optional[str] = None):
        self.run_dir = run_dir
        self.cache_dir = cache_dir

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["spt_run_dir"] = self.run_dir

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        slurm_key = _slurm_session_key()
        if slurm_key is None or self.cache_dir is None:
            return
        idx_path = Path(self.cache_dir) / ".slurm_index" / slurm_key
        if idx_path.is_file():
            return  # index already there for this SLURM session — done

        # Missing index by the time training starts means
        # ``Manager._resolve_run_dir`` settled on a ``run_dir`` without
        # going through ``_write_slurm_index`` — a real regression in
        # that wiring. Don't self-heal: papering over the bug here lets
        # it persist silently. Kill the run loudly with every piece of
        # state we have so it can be diagnosed quickly.
        idx_dir = Path(self.cache_dir) / ".slurm_index"
        existing_entries = (
            sorted(p.name for p in idx_dir.iterdir() if p.is_file())
            if idx_dir.is_dir()
            else []
        )
        slurm_env = {
            k: os.environ.get(k)
            for k in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_RESTART_COUNT",
                "SLURM_PROCID",
                "SLURMD_NODENAME",
            )
        }
        diagnostic = (
            "SLURM-index guard FAILED: index file is missing at training "
            f"start — this should never happen.\n"
            f"  expected path : {idx_path}\n"
            f"  expected → run_dir : {self.run_dir}\n"
            f"  cache_dir : {self.cache_dir}\n"
            f"  slurm_key : {slurm_key}\n"
            f"  index dir exists : {idx_dir.is_dir()}\n"
            f"  index dir contents ({len(existing_entries)} entries): "
            f"{existing_entries[:20]}"
            f"{' ...' if len(existing_entries) > 20 else ''}\n"
            f"  SLURM env : {slurm_env}\n"
            "Possible causes: a code path in Manager._resolve_run_dir "
            "settled on a run_dir without calling _write_slurm_index "
            "(regression), the index dir was wiped between resolution "
            "and now, or the cache_dir on this node differs from the "
            "one used to write the entry. Refusing to silently rewrite — "
            "fix the root cause."
        )
        logging.error(diagnostic)
        raise RuntimeError(diagnostic)


@catch_errors_class()
class Manager(submitit.helpers.Checkpointable):
    """Manages training with logging, scheduling, and checkpointing support.

    Args:
        trainer (Union[dict, DictConfig, pl.Trainer]): PyTorch Lightning trainer configuration or instance.
        module (Union[dict, DictConfig, pl.LightningModule]): Lightning module configuration or instance.
        data (Union[dict, DictConfig, pl.LightningDataModule]): Data module configuration or instance.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        ckpt_path (str, optional): **Absolute** path to a checkpoint to load
            from at the very start of a *fresh* run. Loaded once at step 0;
            after that the run lives in its own freshly-created ``run_dir``
            and produces its own ``last.ckpt``. **Ignored** on SLURM requeue
            — see below. Must be absolute and must exist on disk; otherwise
            ``Manager`` raises before training. Defaults to ``None`` (train
            from scratch / pretrained backbone).
        weights_only (bool, optional): Controls how ``ckpt_path`` is loaded
            on a fresh run. Forwarded to ``Trainer.fit(weights_only=...)``
            when supported by the installed Lightning version. ``True``
            (the PyTorch ≥ 2.6 default for ``torch.load``) loads only model
            weights — optimizer / scheduler / RNG state are discarded, which
            is the usual "transfer-learning init" semantics. Set ``False``
            to fully restore everything from the checkpoint.

            **Has no effect on SLURM requeue**: when ``SLURM_RESTART_COUNT
            >= 1`` the Manager always loads ``<run_dir>/checkpoints/last.ckpt``
            with full state (``weights_only=False``) regardless of this flag,
            because the goal is to resume in-flight training exactly where
            preempt struck.
        compile (bool, optional): Should we compile the given module. Defaults to False.
    """

    def __init__(
        self,
        trainer: Union[dict, DictConfig, pl.Trainer],
        module: Union[dict, DictConfig, pl.LightningModule],
        data: Union[dict, DictConfig, pl.LightningDataModule],
        seed: int = None,
        ckpt_path: str = None,
        weights_only: bool = True,
        compile: bool = False,
    ):
        if seed is None:
            logging.warning(
                "User didn't specify seed, runs won't be exactly reproducible!"
            )
        # Fail-fast on bad user input BEFORE any heavy setup
        # (trainer/module/data instantiation), so a typo doesn't waste
        # multi-second config loads / hydra instantiation.
        # Strict validation of ckpt_path: must be absolute + must exist.
        # We refuse a relative path because a fresh-run resolver creates a
        # run_dir under cache_dir, so a relative path is ambiguous (resolved
        # against what — CWD? run_dir? cache_dir?). We refuse a missing file
        # because falling back to "no ckpt" silently turns a fine-tune
        # request into a from-scratch run, which is a common foot-gun.
        if ckpt_path is not None:
            p = Path(ckpt_path).expanduser()
            if not p.is_absolute():
                raise ValueError(
                    f"`ckpt_path` must be an absolute path; got {ckpt_path!r}. "
                    "Pass a fully-qualified path so loading is unambiguous "
                    "regardless of where the process is launched from."
                )
            p = p.with_suffix(".ckpt")
            if not p.is_file():
                raise FileNotFoundError(
                    f"`ckpt_path` was set to {p} but no such file exists. "
                    "Refusing to silently start training from scratch."
                )
            ckpt_path = p
        self.ckpt_path = ckpt_path
        self.weights_only = bool(weights_only)
        logging.info(
            f"  Manager init: ckpt_path={self.ckpt_path}, "
            f"weights_only={self.weights_only}"
        )
        # Flips to True if `_resolve_run_dir` detects the "SLURM says
        # requeue but no index exists" early-preempt scenario and falls
        # back to fresh-run resolution. `_resolve_load_path` reads it so
        # it doesn't insist on a non-existent ``last.ckpt``.
        self._early_preempt_fallback: bool = False

        self.compile = compile
        self._register_trainer(trainer)
        self._register_module(module)
        self._register_data(data)
        self.seed = seed

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
            logging.info(
                "  cache_dir is not configured — falling back to "
                "Trainer.default_root_dir for run_dir."
            )
            return None

        cache_dir = Path(os.path.expanduser(cfg.cache_dir)).resolve()
        log_header("RunDirectory")
        logging.info(f"  cache_dir = {cache_dir}")

        # ----- DDP coordination ------------------------------------------------
        # `_resolve_run_dir` runs once per rank (each rank is its own process at
        # this point — Trainer/Strategy aren't built yet). To avoid every rank
        # generating its own uuid (and writing its own .slurm_index entry,
        # last-writer-wins), rank-0 picks the dir and publishes it to a shared
        # handoff file; non-zero ranks block on that file and adopt it.
        #
        # Lightning's `rank_zero_only.rank` is the source of truth: it's
        # initialised at import time from the same env vars (`RANK`,
        # `SLURM_PROCID`, ...) that DDP launchers set, and reused by every
        # `@rank_zero_only`-gated logger we ship — keeping detection consistent.
        launch_key = _ddp_launch_key()
        rank = int(getattr(rank_zero_only, "rank", 0) or 0)
        is_rank_zero = rank == 0
        logging.info(
            f"  ddp: launch_key={launch_key or '(single-process)'} "
            f"rank={rank} is_rank_zero={is_rank_zero}"
        )

        if launch_key is not None and not is_rank_zero:
            adopted = self._wait_for_rank_zero_handoff(cache_dir, launch_key)
            if adopted is not None:
                self._run_dir = adopted
                self._run_id = adopted.name
                log_header(f"RunDirectory (rank {rank}, adopted from rank-0)")
                logging.info(f"  run_dir: {self._run_dir}")
                logging.info(f"  run_id:  {self._run_id}")
                # Non-rank-0 doesn't write the index (rank-0 owns that); we
                # just record what we adopted.
                return self._run_dir
            # Timeout — fall through. We log loudly inside the helper. Falling
            # back to local resolution is safer than crashing because only
            # rank-0 actually writes via @rank_zero_only loggers; the worst
            # case is an orphaned empty rank-N dir.
            logging.warning(
                f"! Falling back to local run_dir resolution on rank {rank} — "
                "metrics/sidecar/media won't write here (rank-0 handles those) "
                "but ModelCheckpoint paths may diverge."
            )

        # ============================================================
        # SLURM requeue branch — RESTART_COUNT >= 1 means the *same*
        # SLURM_JOB_ID has already had a previous invocation that
        # presumably wrote the index entry. Look it up and reuse the
        # original run_dir; do nothing else (no index rewrite, no
        # ckpt_path handling — load-path resolution will pick
        # ``<run_dir>/checkpoints/last.ckpt`` later).
        # ============================================================
        slurm_key = _slurm_session_key()
        in_requeue = _is_slurm_requeue()
        log_header("RunDirectory: requeue probe")
        logging.info(f"  SLURM session key   = {slurm_key or '(no SLURM)'}")
        logging.info(
            f"  SLURM_RESTART_COUNT = {os.environ.get('SLURM_RESTART_COUNT', '0')}"
        )
        logging.info(f"  in_requeue          = {in_requeue}")

        if in_requeue:
            if slurm_key is None:
                # SLURM said requeue but no JOB_ID — pathological env, fail loud.
                raise RuntimeError(
                    "SLURM_RESTART_COUNT >= 1 but SLURM_JOB_ID is unset. "
                    "Cannot resolve which run to resume — refusing to start "
                    "a fresh run that would lose prior history."
                )
            idx_path = cache_dir / ".slurm_index" / slurm_key
            logging.info(f"  → looking up index file: {idx_path}")
            if not idx_path.is_file():
                # SLURM bumped RESTART_COUNT but no index entry exists. Two
                # possible causes — and they have opposite correct responses:
                #
                #   (a) Early preempt: prior task was killed before reaching
                #       Manager.__init__ (e.g. cluster-wide eviction during
                #       the submitit pickle load). Nothing was written, no
                #       run_dir was created, no state was lost. Correct
                #       response: fall through to fresh-run.
                #
                #   (b) Partial write: a prior attempt under this JOB_ID got
                #       far enough to mkdir+stamp a run_dir but died before
                #       (or during) writing the index. There IS an orphan
                #       run_dir on disk we'd be ignoring. Correct response:
                #       raise — the user needs to know.
                #
                # Distinguish by scanning ``cache_dir/runs/`` for any
                # ``run_meta.json`` that was stamped with our SLURM_JOB_ID.
                # If we find one, it's case (b): orphan. Otherwise case (a):
                # nothing exists, safe to start fresh.
                orphans = self._find_orphans_for_slurm_key(
                    cache_dir, slurm_key=slurm_key
                )
                if orphans:
                    raise RuntimeError(
                        f"SLURM reports requeue (RESTART_COUNT="
                        f"{os.environ.get('SLURM_RESTART_COUNT')}) for key "
                        f"'{slurm_key}', the index file at {idx_path} is "
                        "missing, AND yet there is/are run_dir(s) on disk "
                        f"already stamped with this SLURM_JOB_ID: "
                        f"{[str(p) for p in orphans]}. This is not the "
                        "early-preempt case (that would leave no artefact). "
                        "Most likely the prior attempt died after creating "
                        "its run_dir but before writing the index. Inspect "
                        f"those dirs and either (i) point the index at the "
                        "right one manually, or (ii) delete them if you'd "
                        "rather start fresh."
                    )
                logging.warning(
                    f"! SLURM_RESTART_COUNT="
                    f"{os.environ.get('SLURM_RESTART_COUNT')} but no index "
                    f"entry at {idx_path}, and no run_dir under {cache_dir} "
                    f"is stamped with SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')}. "
                    "Diagnosis: prior task was preempted before reaching "
                    "Manager.__init__ (typical: cluster-wide eviction "
                    "during submitit pickle load). Treating this as a "
                    "fresh run — nothing was lost because nothing was "
                    "written. A new index entry will be created below."
                )
                in_requeue = False  # fall through to the fresh-run branch
                self._early_preempt_fallback = True
            else:
                try:
                    recorded = Path(idx_path.read_text().strip())
                except OSError as exc:
                    raise RuntimeError(
                        f"Failed to read SLURM index {idx_path}: {exc}"
                    ) from exc
                if not recorded.is_dir():
                    raise RuntimeError(
                        f"SLURM index for '{slurm_key}' points at {recorded}, "
                        "but that directory no longer exists. Stale index or "
                        f"manual deletion. Either restore the dir or remove {idx_path} "
                        "to start fresh."
                    )

                self._run_dir = recorded
                self._run_id = recorded.name
                log_header("RunDirectory (REQUEUE — restored from index)")
                logging.info(f"  run_dir = {self._run_dir}")
                logging.info(f"  run_id  = {self._run_id}")
                if self.ckpt_path is not None:
                    logging.warning(
                        f"! REQUEUE — ignoring user ckpt_path={self.ckpt_path}. "
                        "On requeue we always resume from "
                        f"<run_dir>/checkpoints/last.ckpt (full state); user "
                        "ckpt_path is only consumed on the FIRST (fresh) "
                        "invocation under a SLURM_JOB_ID."
                    )
                if launch_key is not None and is_rank_zero:
                    self._publish_rank_zero_handoff(cache_dir, launch_key, recorded)
                return self._run_dir

        # ============================================================
        # Fresh-run branch — covers both no-SLURM and
        # SLURM RESTART_COUNT=0 (first invocation under this JOB_ID).
        # Always creates a brand-new run_dir.
        # ============================================================
        log_header("RunDirectory: fresh")
        if self.ckpt_path is not None:
            # Already validated absolute+exists in __init__, but log it.
            logging.info(f"  user ckpt_path = {self.ckpt_path}")
            logging.info(f"  weights_only   = {self.weights_only}")
            logging.info(
                "  → checkpoint will be loaded at step 0 of this fresh run; "
                "training otherwise lives in a brand-new run_dir."
            )
        else:
            logging.info(
                "  no user ckpt_path — training from scratch / pretrained backbone."
            )

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
        logging.info(f"  created run_dir = {run_dir}")

        # Hard sanity check: a freshly-created uuid'd dir cannot already
        # contain a last.ckpt. If it does, something is very wrong (uuid
        # collision, stale FS state, caller reusing a path they shouldn't,
        # ...). Fail loud — this is impossible by construction in normal
        # operation.
        stale_last = run_dir / "checkpoints" / "last.ckpt"
        if stale_last.exists():
            raise RuntimeError(
                f"Sanity check failed: just-created fresh run_dir {run_dir} "
                f"already contains {stale_last}. This should be impossible "
                f"(run_id was a fresh uuid). Possible causes: filesystem "
                "weirdness (NFS stale mount, uuid collision in a test), "
                "or a caller reusing a path that should have been unique. "
                "Refusing to start training in a non-empty fresh run_dir."
            )

        # Write sidecar so external tooling (registry scanner, etc.) can
        # discover the run_id from any path under run_dir.
        # We stamp the SLURM_JOB_ID (and array task id, if any) so a future
        # requeue can detect "orphan partial-writes from a prior attempt
        # under this same JOB_ID" — that's how the early-preempt fallback
        # in the requeue branch tells "nothing happened" apart from "a
        # prior attempt did partial work then died" (see that branch).
        meta = {"run_dir": str(run_dir), "run_id": run_id}
        slurm_job_id_env = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id_env:
            meta["slurm_job_id"] = slurm_job_id_env
            slurm_array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
            if slurm_array_task:
                meta["slurm_array_task_id"] = slurm_array_task
        (run_dir / _RUN_META_FILENAME).write_text(json.dumps(meta))
        logging.info(f"  wrote run_meta.json with run_id={run_id}")

        # Record SLURM-key → run_dir so a future preempt-and-requeue
        # cycle finds us. No-op outside SLURM.
        index_msg = self._write_slurm_index(cache_dir, run_dir)
        logging.info(f"  SLURM index = {index_msg}")

        self._run_dir = run_dir
        self._run_id = run_id

        # Publish for non-zero ranks waiting on us. Done LAST so the published
        # path is fully usable (mkdir done, run_meta.json written, .slurm_index
        # updated) by the time another rank picks it up.
        if launch_key is not None and is_rank_zero:
            self._publish_rank_zero_handoff(cache_dir, launch_key, run_dir)

        if slurm_key is not None:
            logging.info(
                "  → future SLURM requeue (SLURM_RESTART_COUNT ≥ 1) for "
                f"key '{slurm_key}' will resume into this directory via the "
                "index entry written above."
            )
        return self._run_dir

    @staticmethod
    def _find_orphans_for_slurm_key(cache_dir: Path, slurm_key: str) -> list[Path]:
        """Return run_dirs whose ``run_meta.json`` matches the given SLURM key.

        Used by the requeue-with-missing-index branch to distinguish:

        * **early-preempt** (no result) — nothing exists for this key, the
          prior attempt died before stamping a run_dir.
        * **partial write** (≥1 hit) — a prior attempt under this same
          ``SLURM_JOB_ID`` got far enough to mkdir + stamp a run_dir but
          died before writing the index. Caller should raise.

        The scan walks ``cache_dir/runs/**/run_meta.json`` and matches the
        full ``slurm_key`` (job id, optionally ``_<array_task_id>``). A
        malformed or unreadable ``run_meta.json`` is skipped (best-effort
        — we'd rather miss a degenerate case than crash on it).
        """
        runs_root = cache_dir / "runs"
        if not runs_root.is_dir():
            return []
        # slurm_key is "<JOB_ID>" or "<JOB_ID>_<TASK_ID>"; pull them apart so
        # we match exactly what the fresh-run branch stamped.
        if "_" in slurm_key:
            job_id, _, task_id = slurm_key.partition("_")
        else:
            job_id, task_id = slurm_key, None
        hits: list[Path] = []
        for meta_file in runs_root.rglob(_RUN_META_FILENAME):
            try:
                meta = json.loads(meta_file.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(meta, dict):
                continue
            if str(meta.get("slurm_job_id", "")) != job_id:
                continue
            if str(meta.get("slurm_array_task_id") or "") != (task_id or ""):
                continue
            hits.append(meta_file.parent)
        return hits

    @staticmethod
    def _write_slurm_index(cache_dir: Path, run_dir: Path) -> str:
        """Record ``SLURM_JOB_ID[_TASK_ID] → run_dir`` iff missing.

        Semantics — match SLURM's lifecycle:

        * **First invocation under a SLURM_JOB_ID** (fresh run, or first
          time hitting Strategy 1 under this job ID): index doesn't
          exist yet → atomic-write it.
        * **Subsequent requeues of the same SLURM_JOB_ID**: index already
          exists from the first invocation → leave it alone.
        * **Different SLURM_JOB_ID**: writes to a different filename
          (different ``slurm_key``) so we never touch another session's
          entry.

        Atomic write via sibling temp + ``fsync`` + ``os.replace`` so a
        process killed mid-write never leaves a partial file. Returns a
        human-readable status string for logging.
        """
        slurm_key = _slurm_session_key()
        if slurm_key is None:
            return "(no SLURM, skipped)"
        idx_dir = cache_dir / ".slurm_index"
        idx_path = idx_dir / slurm_key
        if idx_path.is_file():
            # Same SLURM_JOB_ID, requeue cycle — keep the original
            # mapping written by the first invocation.
            return f"{idx_path} (already present, kept)"
        tmp_path = idx_dir / f".{slurm_key}.tmp.{os.getpid()}"
        try:
            idx_dir.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w") as f:
                f.write(str(run_dir))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, idx_path)
            return f"{idx_path} → {run_dir}"
        except OSError as exc:
            logging.warning(f"! Could not record SLURM index: {exc}")
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            return f"FAILED to write index: {exc}"

    # -- DDP rank-0 handoff (used by `_resolve_run_dir`) -----------------------

    @staticmethod
    def _handoff_path(cache_dir: Path, launch_key: str) -> Path:
        return cache_dir / ".rank_handoff" / launch_key

    def _publish_rank_zero_handoff(
        self, cache_dir: Path, launch_key: str, run_dir: Path
    ) -> None:
        """Atomically write rank-0's chosen ``run_dir`` for non-zero ranks.

        Atomic via temp+``replace`` so a rank-N reader never observes a
        partially-written file. The target path is naturally idempotent: if a
        previous launch with the same key crashed, this rewrite stomps it.
        """
        handoff = self._handoff_path(cache_dir, launch_key)
        try:
            handoff.parent.mkdir(parents=True, exist_ok=True)
            tmp = handoff.with_name(handoff.name + ".tmp")
            tmp.write_text(str(run_dir))
            tmp.replace(handoff)
            logging.info(f"  rank-0 published handoff → {handoff}")
        except OSError as exc:
            logging.warning(f"! Could not publish rank-handoff to {handoff}: {exc}")

    def _wait_for_rank_zero_handoff(
        self, cache_dir: Path, launch_key: str
    ) -> Optional[Path]:
        """Block (poll) until rank-0 has published a usable ``run_dir``.

        Returns the published path, or ``None`` on timeout. The validity check
        (``is_dir()``) means rank-N never adopts a dangling pointer left over
        from a stale prior launch with the same key.
        """
        handoff = self._handoff_path(cache_dir, launch_key)
        deadline = time.monotonic() + _RANK_HANDOFF_TIMEOUT_S
        logged_waiting = False
        while time.monotonic() < deadline:
            try:
                if handoff.is_file():
                    candidate = handoff.read_text().strip()
                    if candidate and Path(candidate).is_dir():
                        return Path(candidate)
                    # Pointer present but invalid (rank-0 still mid-write or
                    # stale crashed launch). Keep polling — rank-0 may rewrite.
            except OSError:
                pass
            if not logged_waiting:
                logging.info(
                    f"  waiting for rank-0 handoff at {handoff} "
                    f"(timeout {_RANK_HANDOFF_TIMEOUT_S:.0f}s)"
                )
                logged_waiting = True
            time.sleep(_RANK_HANDOFF_POLL_S)
        logging.warning(
            f"! rank-0 handoff timed out after {_RANK_HANDOFF_TIMEOUT_S:.0f}s "
            f"({handoff} never appeared with a valid dir)"
        )
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

    def _resolve_load_path(self, run_dir: Path) -> tuple[Optional[str], Optional[bool]]:
        """Decide what to load and how, given the resolved ``run_dir``.

        Returns ``(ckpt_path, weights_only)`` to pass to
        ``Trainer.fit(...)``. Both can be ``None`` (train from scratch).

        Behaviour matrix:

        =========================================  ===============================  =================
        State                                       ckpt_path                        weights_only
        =========================================  ===============================  =================
        SLURM requeue (RESTART_COUNT >= 1)         ``<run_dir>/checkpoints/last.ckpt``  ``False`` (forced)
        Fresh run + user ``ckpt_path`` set         user's absolute path             user's flag (default ``True``)
        Fresh run + no user ``ckpt_path``          ``None``                         ``None``
        =========================================  ===============================  =================

        On requeue we *always* full-restore (optimizer, scheduler, RNG)
        from ``last.ckpt`` and ignore any user-given ``ckpt_path`` —
        loading a stale pretrain checkpoint mid-run would discard
        whatever progress was made. We log loudly when this overrides
        the user's ``ckpt_path`` so it's obvious in run logs.

        Raises ``RuntimeError`` on requeue if ``last.ckpt`` is missing
        (preempt before the first save — there's nothing to resume from).
        """
        log_header("LoadPath resolution")
        in_requeue = _is_slurm_requeue()
        # If `_resolve_run_dir` already decided this is an early-preempt
        # fallback (SLURM RESTART_COUNT≥1 but no index entry → fresh
        # run_dir), behave exactly like a fresh run for load-path purposes:
        # there's no `last.ckpt` to resume from, but the user's ckpt_path
        # (if any) is still a legitimate fresh-run starting point.
        if self._early_preempt_fallback:
            logging.info(
                "  early_preempt_fallback=True — SLURM env says requeue but "
                "the index was missing; treating as fresh for load-path."
            )
            in_requeue = False
        logging.info(f"  in_requeue = {in_requeue}")
        logging.info(f"  user ckpt_path = {self.ckpt_path}")
        logging.info(f"  user weights_only = {self.weights_only}")

        if in_requeue:
            last_ckpt = run_dir / "checkpoints" / "last.ckpt"
            if self.ckpt_path is not None:
                logging.warning(
                    f"! REQUEUE — ignoring user ckpt_path={self.ckpt_path}. "
                    "On requeue we always resume from "
                    f"{last_ckpt} (full state restore) so in-flight training "
                    "picks up exactly where it left off. The user ckpt_path "
                    "is only consumed on the FIRST (fresh) invocation."
                )
            if not last_ckpt.is_file():
                raise RuntimeError(
                    f"REQUEUE but no last.ckpt to resume from at {last_ckpt}. "
                    "The original run was preempted before saving its first "
                    "checkpoint, OR the requeue-checkpoint callback was "
                    "disabled (spt.set(requeue_checkpoint=False)). Either "
                    "way there is no in-flight state to recover. Refusing "
                    "to silently restart from scratch."
                )
            logging.info(
                f"  → loading {last_ckpt} with weights_only=False (full state)"
            )
            return str(last_ckpt), False

        # Fresh run.
        if self.ckpt_path is None:
            logging.info("  → no ckpt_path; training from scratch.")
            return None, None

        # ckpt_path was already validated (absolute + exists) in __init__.
        logging.info(
            f"  → loading user ckpt_path={self.ckpt_path} with "
            f"weights_only={self.weights_only}"
        )
        return str(self.ckpt_path), self.weights_only

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

        # Same redirect for HuggingFaceCheckpointCallback. Without this, an
        # absolute ``save_dir`` (e.g. ``/mnt/data/spt_cache/hf_exports``)
        # gets reused by every array task — 1024 jobs ``rmtree``-ing the
        # same ``last/`` subdir produce ``FileNotFoundError`` storms.
        # Per-run redirection puts each export under
        # ``<run_dir>/hf_exports/`` so concurrency on the export path is
        # impossible by construction.
        try:
            from .callbacks.hf_models import HuggingFaceCheckpointCallback
        except Exception:  # pragma: no cover - HF is optional
            HuggingFaceCheckpointCallback = None
        if HuggingFaceCheckpointCallback is not None:
            hf_save_dir = run_dir / "hf_exports"
            for cb in self._trainer.callbacks:
                if isinstance(cb, HuggingFaceCheckpointCallback):
                    old_dir = cb.save_dir
                    if (
                        old_dir is not None
                        and Path(old_dir).resolve() != hf_save_dir.resolve()
                    ):
                        logging.warning(
                            f"  Redirecting HuggingFaceCheckpointCallback "
                            f"from '{old_dir}' to '{hf_save_dir}' "
                            "(cache_dir is active — prevents races on a shared cache)"
                        )
                    cb.save_dir = hf_save_dir

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
        elif "SLURM_JOB_ID" in os.environ:
            logging.warning(
                "! Requeue checkpoint disabled "
                "(spt.set(requeue_checkpoint=False)) but running under SLURM. "
                "If this run is preempted the next requeue will fail with "
                "'REQUEUE but no last.ckpt to resume from' — Manager looks "
                "for that file to recover in-flight state."
            )
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

        # Drop only ``CSVLogger`` instances that aren't us — RegistryLogger
        # *is* a CSVLogger and the two would otherwise race on the same
        # ``metrics.csv``. Anything else is kept as-is: a user who passes
        # ``TensorBoardLogger`` (or ``logger=True`` which auto-creates one)
        # gets to keep TB writing to its own ``lightning_logs/`` dir;
        # ``WandbLogger``/``TrackioLogger``/etc. are obviously kept.
        self._trainer.loggers = [
            lgr
            for lgr in self._trainer.loggers
            if not (
                isinstance(lgr, lightning.pytorch.loggers.CSVLogger)
                and not isinstance(lgr, RegistryLogger)
            )
        ]

        registry_logger = RegistryLogger(run_dir=run_dir, run_id=run_id)
        # Insert at index 0 so ``trainer.logger`` (the singular alias for
        # ``loggers[0]``) resolves to RegistryLogger. Callbacks gating on
        # ``hasattr(trainer.logger, "log_image")`` then route media through
        # us, not whatever Lightning happens to put first (e.g. an
        # auto-created ``TensorBoardLogger`` when ``logger`` was left unset).
        self._trainer.loggers.insert(0, registry_logger)

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

        # Persist run_dir in every checkpoint so requeue can restore it.
        # Also passes cache_dir so the callback can verify (and self-heal)
        # the SLURM index entry on ``on_train_start`` — see
        # :class:`_RunDirCallback` for the rationale.
        if run_dir is not None:
            cfg = get_config()
            cache_dir = cfg.cache_dir
            self._trainer.callbacks.append(
                _RunDirCallback(str(run_dir), cache_dir=cache_dir)
            )

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

        self._log_fsdp_info(needs_teacher_student=needs_teacher_student)

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
            ckpt_path, weights_only_for_load = self._resolve_load_path(run_dir)
        else:
            # Legacy mode (no cache_dir configured): ckpt_path controls both
            # load and save location. No SLURM-requeue auto-discovery — the
            # user is responsible for resume.
            if "SLURM_JOB_ID" in os.environ and self.ckpt_path is None:
                logging.warning(
                    "Using SLURM but no cache_dir + no ckpt_path: a requeue "
                    "will restart from scratch. Configure cache_dir via "
                    "spt.set(cache_dir=...) for proper preempt/requeue."
                )
            else:
                self._configure_checkpointing()
            ckpt_path = str(self.ckpt_path) if self.ckpt_path else None
            weights_only_for_load = self.weights_only

        if self.compile:
            logging.warning("Compiling module!")
            self.instantiated_module.compile()

        fit_kwargs = {
            "datamodule": self.instantiated_data,
            "ckpt_path": ckpt_path,
        }
        if "weights_only" in inspect.signature(self._trainer.fit).parameters:
            if ckpt_path is not None:
                fit_kwargs["weights_only"] = weights_only_for_load
        elif ckpt_path is not None and weights_only_for_load is not None:
            logging.warning(
                "! Installed Lightning Trainer.fit does not accept "
                "`weights_only`; ignoring requested "
                f"weights_only={weights_only_for_load}. The checkpoint at "
                f"{ckpt_path} will be loaded with Lightning's default policy."
            )

        # Inject the full flattened Hydra config into the module's hparams
        # so Lightning's _log_hyperparams sends it to ALL loggers automatically
        # (wandb, CSV, TensorBoard, registry, etc.)
        self._inject_hydra_hparams()

        log_header("TrainerFit")
        logging.info(f"  ckpt_path:     {ckpt_path}")
        logging.info(f"  weights_only:  {weights_only_for_load}")
        # Wrap fit() so any callback/model error gets a full, flushed,
        # multi-stream traceback in stdout BEFORE it climbs the
        # Hydra/submitit chain (those layers can swallow tracebacks into
        # result.pkl files that never reach SLURM .out logs). We re-raise
        # so the process still exits with a nonzero status — the goal is
        # visibility, not silent recovery.
        try:
            self._trainer.fit(
                self.instantiated_module,
                **fit_kwargs,
            )
        except BaseException as e:
            import sys as _sys
            import traceback as _tb

            _msg = (
                f"\n!!! TRAINER FIT FAILED — {type(e).__name__}: {e}\n"
                f"    epoch={getattr(self._trainer, 'current_epoch', '?')}/"
                f"{getattr(self._trainer, 'max_epochs', '?')}, "
                f"global_step={getattr(self._trainer, 'global_step', '?')}\n"
            )
            # Print to BOTH streams + flush so log captures see it.
            print(_msg, flush=True)
            print(_tb.format_exc(), flush=True)
            _sys.stderr.write(_msg)
            _sys.stderr.write(_tb.format_exc())
            _sys.stderr.flush()
            try:
                logging.exception("Trainer.fit raised — re-raising after loud log")
            except Exception:
                pass
            raise
        self._dump_wandb_data()

    def _log_fsdp_info(self, *, needs_teacher_student: bool) -> None:
        """Log FSDP2 strategy details and warn about TeacherStudentWrapper alignment.

        No-op when the trainer is not using FSDP. When the FSDP2-backed
        :class:`ModelParallelStrategy` is detected, logs the strategy
        subclass, the data/tensor parallel sizes, the checkpoint mode, and
        the parallelize_fn name.
        """
        if not is_fsdp_strategy(self._trainer):
            return

        info = describe_fsdp_strategy(self._trainer)
        logging.info("\tFSDP2 STRATEGY DETECTED")
        logging.info(f"\t\t- subclass: {info['subclass']}")
        logging.info(f"\t\t- data_parallel_size: {info['data_parallel_size']}")
        logging.info(f"\t\t- tensor_parallel_size: {info['tensor_parallel_size']}")
        logging.info(
            f"\t\t- save_distributed_checkpoint: {info['save_distributed_checkpoint']}"
        )
        logging.info(f"\t\t- mp_policy: {info['mp_policy']}")

        if needs_teacher_student:
            logging.warning(
                "\t\t- TeacherStudentWrapper present under FSDP2: student and "
                "teacher must be sharded on the same device mesh with the same "
                "parallelize_fn for the in-place EMA update to be correct. "
                "assert_aligned_wrapping() is the safety net but plan ahead — "
                "see docs/fsdp.md."
            )

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
