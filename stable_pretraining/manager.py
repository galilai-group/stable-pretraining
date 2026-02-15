import copy
import signal
from datetime import timedelta
from pathlib import Path
from typing import Union

import hydra
import lightning
import lightning as pl
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

from .utils import get_required_fn_parameters
from stable_pretraining.utils.error_handling import catch_errors_class


def print_logger_info(logger):
    if isinstance(logger, lightning.pytorch.loggers.logger.DummyLogger):
        logging.info("📈📈📈 DummyLogger setup 📈📈📈")

    elif isinstance(logger, lightning.pytorch.loggers.tensorboard.TensorBoardLogger):
        logging.info("📈📈📈 TensorBoardLogger setup 📈📈📈")
        logging.info(f"📈📈📈 root_dir={logger.root_dir} 📈📈📈")
        logging.info(f"📈📈📈 save_dir={logger.save_dir} 📈📈📈")
        logging.info(f"📈📈📈 log_dir={logger.log_dir} 📈📈📈")

    elif isinstance(logger, lightning.pytorch.loggers.csv_logs.CSVLogger):
        logging.info("📈📈📈 CSVLogger setup 📈📈📈")
        logging.info(f"📈📈📈 root_dir={logger.root_dir} 📈📈📈")
        logging.info(f"📈📈📈 save_dir={logger.save_dir} 📈📈📈")
        logging.info(f"📈📈📈 log_dir={logger.log_dir} 📈📈📈")

    elif isinstance(logger, lightning.pytorch.loggers.wandb.WandbLogger):
        logging.info("📈📈📈 WandbLogger setup 📈📈📈")
        logging.info(f"📈📈📈 init={logger._wandb_init} 📈📈📈")

    elif logger is None:
        logging.warning("📈📈📈 No logger used! 📈📈📈")
    else:
        logging.warning("📈📈📈 Unrecogniezed logger! 📈📈📈")


def print_signal_info():
    logging.info("\t● 👂👂👂 SIGNALS HANDLERS 👂👂👂")
    logging.info(f"\t\t- SIGUSR1: `{signal.getsignal(signal.SIGUSR1)}`")
    logging.info(f"\t\t- SIGUSR2: `{signal.getsignal(signal.SIGUSR2)}`")
    logging.info(f"\t\t- SIGCONT: `{signal.getsignal(signal.SIGCONT)}`")
    logging.info(f"\t\t- SIGTERM: `{signal.getsignal(signal.SIGTERM)}`")


@catch_errors_class()
class Manager(submitit.helpers.Checkpointable):
    """Manages training with logging, scheduling, and checkpointing support.

    Args:
        trainer (Union[dict, DictConfig, pl.Trainer]): PyTorch Lightning trainer configuration or instance.
        module (Union[dict, DictConfig, pl.LightningModule]): Lightning module configuration or instance.
        data (Union[dict, DictConfig, pl.LightningDataModule]): Data module configuration or instance.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        ckpt_path (str, optional): Path to checkpoint for resuming training. Defaults to "last".
        compile (bool, optional): Should we compile the given module. Defaults to False.
    """

    def __init__(
        self,
        trainer: Union[dict, DictConfig, pl.Trainer],
        module: Union[dict, DictConfig, pl.LightningModule],
        data: Union[dict, DictConfig, pl.LightningDataModule],
        seed: int = None,
        ckpt_path: str = None,
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

    @property
    def instantiated_module(self):
        if not isinstance(self.module, pl.LightningModule):
            logging.info("\t● instantiating pl_module...")
            # with self._trainer.init_module():
            self._instantiated_module = hydra.utils.instantiate(
                self.module, _convert_="object"
            )
            logging.info("\t● module instantiated ✅")
        else:
            self._instantiated_module = self.module
        return self._instantiated_module

    @property
    def instantiated_data(self):
        if not isinstance(self.data, pl.LightningDataModule):
            self._instantiated_data = hydra.utils.instantiate(
                self.data, _convert_="object", _recursive_=False
            )
            logging.info("\t● data instantiated ✅")
        else:
            self._instantiated_data = self.data
        return self._instantiated_data

    def __call__(self):
        logging.info(f"📁📁📁 CURRENT WORKING DIR: {Path().resolve()} 📁📁📁")
        logging.info(f"🌱🌱🌱 SEEDING EVERYTHING with {self.seed=} 🌱🌱🌱")
        pl.seed_everything(self.seed, workers=True)
        if isinstance(self.trainer, pl.Trainer):
            self._trainer = self.trainer
        else:
            if "callbacks" in self.trainer:
                logging.info("\t● instantiating callbacks...")
                callbacks = hydra.utils.instantiate(
                    self.trainer.callbacks, _convert_="object"
                )
                for i, callback in enumerate(callbacks):
                    if not callable(callback):
                        continue
                    assert ["module"] == get_required_fn_parameters(callback)
                    callbacks[i] = callback(module=self.instantiated_module)
                logging.info("\t● callbacks instantiated ✅")
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
            logging.info("\t● trainer instantiated ✅")

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
                logging.info(
                    "\t● Auto-detected TeacherStudentWrapper, adding TeacherStudentCallback ✅"
                )
                self._trainer.callbacks.append(TeacherStudentCallback())

        # Inject Hydra config into WandbCallback so it can sync to wandb
        from .callbacks.wandb_lifecycle import WandbCallback

        for cb in self._trainer.callbacks:
            if isinstance(cb, WandbCallback):
                hydra_config = {}
                if isinstance(self.trainer, DictConfig):
                    hydra_config["trainer"] = self.trainer
                if isinstance(self.module, DictConfig):
                    hydra_config["module"] = self.module
                if isinstance(self.data, DictConfig):
                    hydra_config["data"] = self.data
                cb._hydra_config = hydra_config if hydra_config else None
                break

        print_logger_info(self._trainer.logger)
        print_signal_info()

        logging.info("\t● 📞📞📞 CALLBACKS 📞📞📞")
        logging.info(f"\t\t - we found {len(self._trainer.callbacks)} callbacks")
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
        logging.info(f"📣📣📣 CALLING trainer.fit with {ckpt_path=} 📣📣📣")
        self._trainer.fit(
            self.instantiated_module,
            datamodule=self.instantiated_data,
            ckpt_path=ckpt_path,
        )

    def validate(self):
        logging.info("📣📣📣 CALLING trainer.validate 📣📣📣")

        self._trainer.validate(
            self.instantiated_module, datamodule=self.instantiated_data
        )

    def predict(self):
        logging.info("📣📣📣 CALLING trainer.predict 📣📣📣")

        self._trainer.predict(
            self.instantiated_module, datamodule=self.instantiated_data
        )

    def test(self):
        logging.info("📣📣📣 CALLING trainer.test 📣📣📣")

        self._trainer.test(self.instantiated_module, datamodule=self.instantiated_data)

    def save_checkpoint(
        self, path: str = None, upload_wandb: bool = False, verbose=True
    ):
        # TODO: figure out how to flush logging in subprocess
        if verbose:
            print("Entering checkpoint method", flush=True)
        if path is None:
            path = (Path() / "checkpoint.ckpt").resolve()
            if verbose:
                print(f"\t● saving checkpoint to local path {path} ⏳", flush=True)
        else:
            path = Path(path)
            if not path.parent.is_dir():
                path.parent.mkdir(parents=True)
            if verbose:
                print(f"\t● saving checkpoint to user's path {path} ⏳", flush=True)
        self._trainer.save_checkpoint(str(path))
        if verbose:
            print("\t● checkpoint saved ✅", flush=True)
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
            print("\t● Wandb used and online:", flush=True)
            artifact = wandb.Artifact("requeue_checkpoint", "model")
            artifact.add_file(str(ckpt_path))
            artifact.ttl = timedelta(days=30)
            print("\t\t● artifact created ✅", flush=True)
            wandb.run.log_artifact(artifact)
            print("\t\t● artifact logged ✅", flush=True)
            ckpt_path.unlink()
            print("\t\t● local checkpoint deleted ✅", flush=True)
        else:
            print("\t● Wandb used and offline:", flush=True)
            if WANDB_AVAILABLE and wandb.run:
                wandb.run.config.update({"ckpt_path": str(ckpt_path.resolve())})
            print("\t● `ckpt_path` added to Wandb config ✅", flush=True)
        # for offline case
        from .callbacks.wandb_lifecycle import WandbCallback

        WandbCallback._dump_wandb_data()

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
        logging.info("\t● 📞📞📞 CHECKPOINTING SETUP 📞📞📞")
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
                f"\t\t Checkpoint: `manager.ckpt_path` ({ckpt_path}) is set and a matching `ModelCheckpoint` callback was found to be saving to the same directory."
            )
            if is_slurm_job:
                logging.info(
                    "\t\t This setup is ready for SLURM preemption and requeueing."
                )

        # Case 2: Intentional ckpt_path, but no callback found - assume the user forgot and add a callback
        elif ckpt_path is not None and not is_manager_path_handled_by_callback:
            logging.warning(
                f"\t\t Checkpoint mismatch: `manager.ckpt_path` ({ckpt_path}) was provided, but no matching `ModelCheckpoint` callback was found."
            )
            logging.warning(
                "\t\t Automatically creating a `ModelCheckpoint` to save to the specified path to prevent data loss."
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
                "\t\t - Automatic `ModelCheckpoint` callback has been added to the trainer."
            )

        # Case 3: No checkpoint, but with ModelCheckpoint callback - assume we are training from scratch.
        elif ckpt_path is None and is_mc_explicitly_configured:
            logging.info(
                "\t\t Checkpointing: A user-defined `ModelCheckpoint` callback was found. It will be used for saving checkpoints."
            )
            logging.info(
                "\t\t The `Manager` will not manage resuming from a specific path as `manager.ckpt_path` was not provided."
            )
            if is_slurm_job:
                logging.warning(
                    "\t\t SLURM WARNING: Since `manager.ckpt_path` is not set, this job will restart from scratch if requeued, even though checkpoints are being saved elsewhere."
                )

        # Case 4: No checkpoint and no ModelCheckpoint callback - assume we are training without saving checkpoints
        elif ckpt_path is None and not is_mc_explicitly_configured:
            logging.info(
                "\t\t No Checkpointing: No `manager.ckpt_path` was provided and no `ModelCheckpoint` callback was found."
            )
            logging.info("\t\t The model will not be saved during this run.")
            if is_slurm_job:
                logging.error(
                    "\t\t CRITICAL SLURM WARNING: This job will lose all progress if it is preempted or requeued. It is highly recommended to configure checkpointing."
                )

    def _register_trainer(self, trainer):
        if type(trainer) is dict:
            trainer = OmegaConf.create(trainer)
        if type(trainer) is DictConfig:
            self.trainer: DictConfig = copy.deepcopy(trainer)
            logging.debug("\t● trainer config saved ✅")
        elif isinstance(trainer, pl.Trainer):
            self.trainer = trainer
            logging.debug("\t● trainer already instantiated ✅")
        else:
            raise ValueError(
                f"`trainer` must be a dict, DictConfig or pl.Trainer, not {type(trainer)}"
            )

    def _register_module(self, module):
        if type(module) is dict:
            module = OmegaConf.create(module)
        if type(module) is DictConfig:
            self.module: DictConfig = copy.deepcopy(module)
            logging.debug("\t● module config saved ✅")
        elif isinstance(module, pl.LightningModule):
            self.module = module
            logging.debug("\t● module already instantiated ✅")
        else:
            raise ValueError(
                f"`module` must be a dict, DictConfig or pl.LightningModule, not {type(module)}"
            )

    def _register_data(self, data):
        if type(data) is dict:
            data = OmegaConf.create(data)
        if type(data) is DictConfig:
            self.data: DictConfig = copy.deepcopy(data)
            logging.debug("\t● data config saved ✅")
        elif isinstance(data, pl.LightningDataModule):
            self.data = data
            logging.debug("\t● data already instantiated ✅")
        else:
            raise ValueError(
                f"`data` must be a dict, DictConfig or pl.LightningDataModule, not {type(data)}"
            )
