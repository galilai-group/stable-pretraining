import json
from pathlib import Path
from typing import Optional

from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf

from .. import WANDB_AVAILABLE

if WANDB_AVAILABLE:
    import wandb
else:
    wandb = None


class WandbCallback(Callback):
    """Manage the wandb run lifecycle: init, config sync, and teardown.

    This callback consolidates wandb-specific logic.
    It handles the following concerns:

    1. **Run initialization** -- On ``setup()``, triggers ``wandb.init()``
       via the trainer's ``WandbLogger`` and captures the run ID for
       later use.
    2. **Config sync** -- Uploads the Hydra ``DictConfig`` to wandb by
       flattening nested dicts into dot-separated keys.  For offline
       runs that are being resumed (SLURM requeue), reloads the config
       from the previous offline run directory instead.
    3. **Offline data persistence** -- On ``teardown()``, writes
       ``wandb-summary.json`` and ``wandb-config.json`` into the wandb
       run directory so that offline runs can be synced or inspected.
    4. **Clean shutdown** -- Calls ``wandb.finish()`` on ``teardown()``
       and resets the logger so a subsequent ``setup()`` can re-init
       the same run (for multi-stage workflows like fit then test) or a new
       run (for multirun sweeps).

    **Compatible Loggers:**
    - ``WandbLogger`` (online and offline modes)

    ** Ignored Loggers:**
    - ``TensorBoardLogger``, ``CSVLogger``, ``DummyLogger``, ``None``

    Args:
        hydra_config: Optional raw Hydra configuration (a dict with keys
            like ``"trainer"``, ``"module"``, ``"data"``).  Each value
            should be a ``DictConfig`` or plain ``dict``.  When provided,
            the config is flattened and uploaded to wandb on the first
            ``setup()`` call.  When ``None``, the callback skips config
            sync (useful when the user has already configured wandb
            externally).  The ``Manager`` injects this automatically.

    Example:
        .. code-block:: python

            from stable_pretraining.callbacks import WandbCallback

            # With Hydra config (typical Manager usage -- injected automatically):
            callback = WandbCallback(
                hydra_config={
                    "trainer": trainer_cfg,
                    "module": module_cfg,
                    "data": data_cfg,
                }
            )
            trainer = Trainer(callbacks=[callback], logger=WandbLogger(...))
            trainer.fit(model, datamodule=dm)
            # wandb.finish() is called automatically in teardown

            # Without config (manual usage):
            callback = WandbCallback()
            trainer = Trainer(callbacks=[callback], logger=WandbLogger(...))
            trainer.fit(model, datamodule=dm)

    Notes:
        - For SLURM preemption, pair this with ``WandbCheckpoint`` which
          persists the wandb run ID inside Lightning checkpoints.
        - After ``teardown()``, the logger is reset so the next
          ``setup()`` re-inits the same run with ``resume="allow"``.
    """

    def __init__(self, hydra_config: Optional[dict] = None):
        super().__init__()
        self._hydra_config = hydra_config
        self._config_synced = False
        self._run_id: Optional[str] = None

    @rank_zero_only
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Initialize wandb and sync Hydra config.

        Called at the start of fit, validate, test, or predict.  Triggers
        ``trainer.logger.experiment`` which lazily calls ``wandb.init()``.
        Then uploads the flattened Hydra config (once per callback lifetime).
        """
        if not self._is_wandb(trainer):
            return

        logging.info(f"WandbCallback: setup for stage='{stage}'")

        # If we have a run ID from a previous teardown, set it so the
        # logger re-inits the same run (resume="allow").
        if self._run_id is not None and hasattr(trainer.logger, "_wandb_init"):
            trainer.logger._wandb_init["id"] = self._run_id
            trainer.logger._wandb_init["resume"] = "allow"
            trainer.logger._id = self._run_id

        # Trigger wandb.init() via the logger's lazy property
        exp = trainer.logger.experiment
        self._run_id = exp.id
        logging.info(f"WandbCallback: wandb run active (id={exp.id})")

        # Sync config only once
        if not self._config_synced:
            self._sync_config(exp)
            self._config_synced = True

    @rank_zero_only
    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Dump offline data and close the wandb run.

        Called at the end of fit, validate, test, or predict.  Writes
        offline summary/config files, then calls ``wandb.finish()`` and
        resets the logger so a subsequent ``setup()`` can re-init.
        """
        if not self._is_wandb(trainer):
            return

        logging.info(f"WandbCallback: teardown for stage='{stage}'")
        self._dump_wandb_data()

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
            logging.info("WandbCallback: wandb.finish() called")

        # Reset the logger so the next setup() triggers a fresh init
        trainer.logger._experiment = None

    @rank_zero_only
    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ) -> None:
        """Dump offline data on exception for crash recovery."""
        if not self._is_wandb(trainer):
            return

        logging.error(
            f"WandbCallback: exception during training: "
            f"{type(exception).__name__}: {exception}"
        )
        self._dump_wandb_data()

    def _sync_config(self, exp) -> None:
        """Upload Hydra config to wandb, handling offline resume."""
        if not WANDB_AVAILABLE or wandb.run is None:
            return

        if exp.offline:
            self._sync_config_offline(exp)
        elif len(wandb.config.keys()):
            logging.info(
                "WandbCallback: wandb config already populated, "
                "skipping Hydra config upload"
            )
        else:
            self._sync_config_online()

    def _sync_config_offline(self, exp) -> None:
        """For offline resume: reload config from previous run directory."""
        previous_run = self._find_previous_run_dir()
        if previous_run is None:
            logging.info("WandbCallback: first offline run, syncing Hydra config")
            self._sync_config_online()
            return

        logging.info(
            f"WandbCallback: found previous offline run ({previous_run}), "
            f"reusing config"
        )
        config_path = previous_run / "files" / "wandb-config.json"
        with open(config_path, "r") as f:
            last_config = json.load(f)
        exp.config.update(last_config)
        logging.info("WandbCallback: offline config reloaded")

    def _sync_config_online(self) -> None:
        """Flatten and upload Hydra config to wandb."""
        if self._hydra_config is None:
            logging.info("WandbCallback: no Hydra config provided, skipping upload")
            return

        import pandas as pd

        config = {}
        for key in ("trainer", "module", "data"):
            section = self._hydra_config.get(key)
            if section is None:
                continue
            if isinstance(section, (dict, DictConfig)):
                config[key] = (
                    OmegaConf.to_container(section, resolve=True)
                    if isinstance(section, DictConfig)
                    else section
                )

        if not config:
            logging.info(
                "WandbCallback: everything already instantiated, "
                "nothing to add to wandb config"
            )
            return

        logging.info("WandbCallback: flattening Hydra config for wandb upload")
        config = pd.json_normalize(config, sep=".").to_dict(orient="records")[0]

        # Iteratively flatten lists into indexed keys
        while True:
            all_flat = True
            for k in list(config.keys()):
                if isinstance(config[k], list):
                    all_flat = False
                    for i, v in enumerate(config[k]):
                        config[f"{k}.{i}"] = v
                    del config[k]
            config = pd.json_normalize(config, sep=".").to_dict(orient="records")[0]
            if all_flat:
                break

        logging.info(f"WandbCallback: uploading {len(config)} config items to wandb")
        if WANDB_AVAILABLE and wandb.run:
            wandb.config.update(config)

    @staticmethod
    def _dump_wandb_data() -> None:
        """Write summary and config JSON files for offline runs."""
        if not WANDB_AVAILABLE or wandb.run is None or not wandb.run.offline:
            return

        run_dir = Path(wandb.run.dir)

        # Dump summary
        summary_path = run_dir / "wandb-summary.json"
        if not summary_path.is_file():
            summary_dict = wandb.run.summary._as_dict()
            logging.info(
                f"WandbCallback: summary: {json.dumps(summary_dict, indent=2)}"
            )
            with open(summary_path, "w") as f:
                json.dump(summary_dict, f)
            logging.info(f"WandbCallback: saved summary at {summary_path}")
        else:
            logging.debug("WandbCallback: summary file already exists, skipping")

        # Dump config
        config_path = run_dir / "wandb-config.json"
        if not config_path.is_file():
            with open(config_path, "w") as f:
                json.dump(wandb.run.config.as_dict(), f)
            logging.info(f"WandbCallback: saved config at {config_path}")
        else:
            logging.debug("WandbCallback: config file already exists, skipping")

    @staticmethod
    def _find_previous_run_dir() -> Optional[Path]:
        """Find the previous offline run directory for config reuse.

        When a SLURM job is requeued and the wandb run ID is preserved
        (via ``WandbCheckpoint``), the same run ID produces multiple
        ``offline-run-*-<id>`` directories.  This method returns the
        second-to-last one (the previous run).

        Returns:
            Path to the previous run directory, or ``None`` if this is
            the first run or wandb is not running offline.
        """
        if not WANDB_AVAILABLE or not wandb.run:
            return None

        current_dir = Path(wandb.run.dir).parent
        parent = current_dir.parent
        logging.info(f"WandbCallback: searching for previous offline runs in {parent}")

        runs = sorted(parent.glob(f"offline-run-*-{wandb.run.id}"))
        logging.info(f"WandbCallback: found {len(runs)} run(s)")
        for run in runs:
            logging.debug(f"WandbCallback:   - {run.name}")

        if len(runs) <= 1:
            return None

        assert runs[-1] == current_dir, (
            f"Expected current dir {current_dir} to be the last run, but got {runs[-1]}"
        )
        return runs[-2]

    @staticmethod
    def _is_wandb(trainer: Trainer) -> bool:
        """Check if the trainer uses a WandbLogger."""
        return isinstance(trainer.logger, WandbLogger)


class WandbCheckpoint(Callback):
    """Persist the wandb run ID inside Lightning checkpoints.

    This callback ensures wandb run continuity across SLURM preemption
    and requeue cycles.  On checkpoint save, it stores the current wandb
    run ID.  On checkpoint load, it detects a mismatch between the active
    run and the checkpoint's run, and re-initializes wandb with the
    correct run ID so that metrics continue to be logged to the same run.

    This callback is automatically included in the default callback list
    via ``callbacks.factories.default()``.  It works in tandem with
    ``WandbCallback`` which handles the broader wandb lifecycle.

    Notes:
        - No-op when the trainer does not use ``WandbLogger``.
        - On load, if the run IDs already match, no re-initialization
          is performed.
        - On load with a mismatched ID, the current (incorrect) wandb
          run is deleted via the wandb API and a new run is created with
          the checkpoint's ID.
    """

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None
    ) -> None:
        logging.info("WandbCheckpoint: setup")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        logging.info("WandbCheckpoint: saving wandb run ID to checkpoint")
        if isinstance(trainer.logger, WandbLogger):
            checkpoint["wandb"] = {"id": trainer.logger.version}
            logging.info(f"WandbCheckpoint: saved run ID={trainer.logger.version}")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        logging.info("WandbCheckpoint: checking for wandb run ID in checkpoint")
        if "wandb" not in checkpoint:
            return

        logging.info("WandbCheckpoint: found wandb info, restoring run continuity")
        if not hasattr(trainer, "logger"):
            logging.warning("WandbCheckpoint: trainer has no logger, skipping")
            return
        if not isinstance(trainer.logger, WandbLogger):
            logging.warning(
                f"WandbCheckpoint: expected WandbLogger, got {trainer.logger}, skipping"
            )
            return

        logging.info("WandbCheckpoint: trainer has a WandbLogger")
        import wandb as _wandb

        if _wandb.run is None and trainer.global_rank > 0:
            logging.info(
                "WandbCheckpoint: run not initialized on non-zero rank, skipping"
            )
            return

        if _wandb.run is not None and _wandb.run.id == checkpoint["wandb"]["id"]:
            logging.info("WandbCheckpoint: run ID already matches checkpoint, skipping")
            return

        # Run ID mismatch: delete the incorrect run and re-init with correct ID
        logging.info(
            f"WandbCheckpoint: deleting mismatched run "
            f"{_wandb.run.entity}/{_wandb.run.project}/{_wandb.run.id}"
        )
        api = _wandb.Api()
        run = api.run(f"{_wandb.run.entity}/{_wandb.run.project}/{_wandb.run.id}")
        _wandb.finish()
        run.delete()
        trainer.logger._experiment = None
        wandb_id = checkpoint["wandb"]["id"]
        trainer.logger._wandb_init["id"] = wandb_id
        trainer.logger._id = wandb_id
        # Trigger re-init
        trainer.logger.experiment
        logging.info(
            f"WandbCheckpoint: re-initialized run "
            f"{_wandb.run.entity}/{_wandb.run.project}/{_wandb.run.id}"
        )
