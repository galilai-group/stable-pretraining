import lightning.pytorch as pl
import torch
from lightning.pytorch import Callback
from loguru import logger as logging
from prettytable import PrettyTable
from lightning.pytorch.utilities import rank_zero_only
import os
from ..data.module import DataModule
from .utils import log_header


class ModuleSummary(pl.Callback):
    """Logs detailed module parameter statistics in a formatted table.

    This callback provides a comprehensive overview of all modules in the model,
    showing the number of trainable, non-trainable, uninitialized parameters,
    and buffers for each module. This helps understand model architecture and
    parameter distribution.

    The summary is displayed during the setup phase and includes:
    - Module name and hierarchy
    - Trainable parameter count
    - Non-trainable (frozen) parameter count
    - Uninitialized parameter count (for lazy modules)
    - Buffer count (non-parameter persistent state)
    """

    @rank_zero_only
    def setup(self, trainer, pl_module, stage):
        headers = [
            "Module",
            "Trainable parameters",
            "Non Trainable parameters",
            "Uninitialized parameters",
            "Buffers",
        ]
        table = PrettyTable()
        table.field_names = headers
        table.align["Module"] = "l"
        table.align["Trainable parameters"] = "r"
        table.align["Non Trainable parameters"] = "r"
        table.align["Uninitialized parameters"] = "r"
        table.align["Buffers"] = "r"
        log_header("PyTorch Modules")
        for name, module in pl_module.named_modules():
            num_trainable = 0
            num_nontrainable = 0
            num_buffer = 0
            num_uninitialized = 0
            for p in module.parameters():
                if isinstance(p, torch.nn.parameter.UninitializedParameter):
                    n = 0
                    num_uninitialized += 1
                else:
                    n = p.numel()
                if p.requires_grad:
                    num_trainable += n
                else:
                    num_nontrainable += n
            for p in module.buffers():
                if isinstance(p, torch.nn.parameter.UninitializedBuffer):
                    n = 0
                    num_uninitialized += 1
                else:
                    n = p.numel()
                num_buffer += n
            table.add_row(
                [name, num_trainable, num_nontrainable, num_uninitialized, num_buffer]
            )
        logging.info(f"\\n{table}")

        return super().setup(trainer, pl_module, stage)


class LoggingCallback(pl.Callback):
    """Displays validation metrics in a color-coded formatted table.

    This callback creates a visually appealing table of all validation metrics
    at the end of each validation epoch. Metrics are color-coded for better
    readability in terminal outputs.

    Features:
    - Automatic sorting of metrics by name
    - Color coding: blue for metric names, green for values
    - Filters out internal metrics (log, progress_bar)
    """

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                table.add_row(
                    [
                        "\033[0;34;40m" + key + "\033[0m",
                        "\033[0;32;40m" + str(metrics[key].item()) + "\033[0m",
                    ]
                )
        logging.info(f"\\n{table}")


class TrainerInfo(Callback):
    """Links the trainer to the DataModule for enhanced functionality.

    This callback establishes a bidirectional connection between the trainer
    and DataModule, enabling the DataModule to access trainer information
    such as device placement, distributed training state, and other runtime
    configurations.

    This is particularly useful for DataModules that need to adapt their
    behavior based on trainer configuration (e.g., device-aware data loading,
    distributed sampling adjustments).

    Note:
        Only works with DataModule instances that have a set_pl_trainer method.
        A warning is logged if using a custom DataModule without this method.
    """

    def setup(self, trainer, pl_module, stage):
        logging.info("  linking trainer to DataModule")
        if not isinstance(trainer.datamodule, DataModule):
            logging.warning("! Using a custom DataModule, won't have extra info")
            return
        try:
            trainer.datamodule.set_pl_trainer(trainer)
        except AttributeError as e:
            logging.error(
                "trainer's datamodule is of type"
                "stable_pretraining.data.DataModule but does"
                "not have a method `set_pl_trainer`..."
            )
            raise (e)
        return super().setup(trainer, pl_module, stage)


class SLURMInfo(Callback):
    """Links the trainer to the DataModule for enhanced functionality.

    This callback establishes a bidirectional connection between the trainer
    and DataModule, enabling the DataModule to access trainer information
    such as device placement, distributed training state, and other runtime
    configurations.

    This is particularly useful for DataModules that need to adapt their
    behavior based on trainer configuration (e.g., device-aware data loading,
    distributed sampling adjustments).

    Note:
        Only works with DataModule instances that have a set_pl_trainer method.
        A warning is logged if using a custom DataModule without this method.
    """

    def setup(self, trainer, pl_module, stage):
        log_header("SLURM")
        logging.info(f"  Job ID: {self._get_env_var('SLURM_JOB_ID')}")
        logging.info(f"  Task ID: {self._get_env_var('SLURM_ARRAY_TASK_ID')}")
        logging.info(f"  Job Name: {self._get_env_var('SLURM_JOB_NAME')}")
        logging.info(f"  Nodes: {self._get_env_var('SLURM_JOB_NUM_NODES')}")
        logging.info(f"  Tasks: {self._get_env_var('SLURM_NTASKS')}")
        logging.info(f"  CPUs per node: {self._get_env_var('SLURM_CPUS_ON_NODE')}")
        logging.info(f"  CPUs per task: {self._get_env_var('SLURM_CPUS_PER_TASK')}")
        logging.info(f"  Memory per node: {self._get_env_var('SLURM_MEM_PER_NODE')}")
        logging.info(f"  Memory per CPU: {self._get_env_var('SLURM_MEM_PER_CPU')}")
        logging.info(f"  Time limit: {self._get_env_var('SLURM_JOB_TIME')}")
        logging.info(f"  Partition: {self._get_env_var('SLURM_JOB_PARTITION')}")
        logging.info(f"  Node List: {self._get_env_var('SLURM_JOB_NODELIST')}")
        logging.info(f"  Submit Directory: {self._get_env_var('SLURM_SUBMIT_DIR')}")
        pl_module.save_hyperparameters(
            {
                **pl_module.hparams,
                "slurm.job_id": self._get_env_var("SLURM_JOB_ID"),
                "slurm.task_id": self._get_env_var("SLURM_ARRAY_TASK_ID"),
            },
        )

    def _get_env_var(self, var, default="N/A"):
        return os.environ.get(var, default)
