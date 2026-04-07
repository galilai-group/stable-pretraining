import copy
import inspect
import json
import signal
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import hydra
import lightning
import lightning as pl
import pandas as pd
import submitit
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf
import os
from . import WANDB_AVAILABLE

if WANDB_AVAILABLE:
    import wandb
else:
    wandb = None

from .callbacks.checkpoint_sklearn import find_wandb_logger, _WANDB_RESUME_FILENAME
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
        logging.warning("! Unrecognized logger!")


def print_signal_info():
    log_header("SignalHandlers")
    logging.info(f"  SIGUSR1: `{signal.getsignal(signal.SIGUSR1)}`")
    logging.info(f"  SIGUSR2: `{signal.getsignal(signal.SIGUSR2)}`")
    logging.info(f"  SIGCONT: `{signal.getsignal(signal.SIGCONT)}`")
    logging.info(f"  SIGTERM: `{signal.getsignal(signal.SIGTERM)}`")


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

        # Only attempt resume if we actually have a checkpoint to load
        if self.ckpt_path is None or not self.ckpt_path.is_file():
            return

        sidecar = Path(_WANDB_RESUME_FILENAME)
        if not sidecar.is_file():
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
            config = {}
            if isinstance(self.trainer, dict):
                config["trainer"] = OmegaConf.to_container(self.trainer, resolve=True)
            if isinstance(self.module, dict):
                config["module"] = OmegaConf.to_container(self.module, resolve=True)
            if isinstance(self.data, dict):
                config["data"] = OmegaConf.to_container(self.data, resolve=True)
            if not config:
                logging.info(
                    "  Everything already instantiated, nothing is added to config!"
                )
                return
            config = pd.json_normalize(config, sep=".")
            config = config.to_dict(orient="records")[0]
            while True:
                logging.info("  flattening one level of Hydra's config")
                valid = True
                for k in list(config.keys()):
                    if type(config[k]) is list:
                        valid = False
                        for i, j in enumerate(config[k]):
                            config[f"{k}.{i}"] = j
                        del config[k]
                config = pd.json_normalize(config, sep=".")
                config = config.to_dict(orient="records")[0]
                if valid:
                    break
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
        self.init_and_sync_wandb()
        print_logger_info(self._trainer.logger)
        print_signal_info()

        log_header("Callbacks")
        logging.info(f"  count: {len(self._trainer.callbacks)}")
        if "SLURM_JOB_ID" in os.environ and self.ckpt_path is None:
            logging.warning(
                "Using SLURM but no ckpt_path, if requeued it will start from scratch"
            )
            logging.warning("Consider passing a value to the Manager's `ckpt_path` ")
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
