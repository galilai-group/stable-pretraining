import math
from loguru import logger
from lightning.pytorch import Callback, Trainer, LightningModule

from .registry import log as _spt_log
from .utils import log_header


class WeightDecayUpdater(Callback):
    """PyTorch Lightning Callback to update optimizer's weight decay per batch.

    - Supports multiple schedules: 'constant', 'linear', 'cosine', 'exponential'
    - Optionally specify which optimizer param group(s) to update (by index)
    - Infers total steps from Trainer config (max_steps or max_epochs + dataloader)
    - Checkpointable: state is saved/restored with Trainer checkpoints
    - Extensive Loguru logging
    Args:
        schedule_type (str): One of 'constant', 'linear', 'cosine', 'exponential'
        start_value (float): Initial weight decay value
        end_value (float): Final weight decay value (for non-constant schedules)
        param_group_indices (list[int] or None): List of param group indices to update. If None, updates all.
    """

    def __init__(
        self,
        schedule_type: str = "cosine",
        start_value: float = 0.01,
        end_value: float = 0.0,
        param_group_indices: list = None,
        opt_idx: int = None,
        verbose: bool = None,
    ):
        super().__init__()
        self.schedule_type = schedule_type
        self.start_value = start_value
        self.end_value = end_value
        self.param_group_indices = param_group_indices
        self.total_steps = None  # Will be set in on_fit_start
        self.opt_idx = opt_idx
        from .utils import resolve_verbose

        self.verbose = resolve_verbose(verbose)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        # Prefer max_steps if set
        self.total_steps = (
            trainer.estimated_stepping_batches * trainer.accumulate_grad_batches
        )
        log_header("WeightDecayUpdater")
        logger.info(f"  total_steps: {self.total_steps}")

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer
    ):
        optis = pl_module.optimizers()
        if self.opt_idx is not None and optimizer != optis[self.opt_idx].optimizer:
            return
        step = trainer.global_step // len(optis)
        accumulate_grad_batches = trainer.accumulate_grad_batches
        if (step + 1) % accumulate_grad_batches != 0:
            logger.debug("  step but accumulating grad, skipping step")
            return
        new_weight_decay = self._compute_weight_decay(step)
        indices = (
            self.param_group_indices
            if self.param_group_indices is not None
            else range(len(optimizer.param_groups))
        )
        for i in indices:
            param_group = optimizer.param_groups[i]
            old_wd = param_group.get("weight_decay", None)
            param_group["weight_decay"] = new_weight_decay
            logger.debug(
                f"  step {step}: param_group {i} weight_decay {old_wd} -> {new_weight_decay}"
            )
        if self.verbose:
            _spt_log(
                "hparams/weight_decay",
                new_weight_decay,
                on_step=True,
                on_epoch=False,
            )

    def _compute_weight_decay(self, step: int) -> float:
        progress = min(step, self.total_steps) / self.total_steps
        if self.schedule_type == "constant":
            return self.start_value
        elif self.schedule_type == "linear":
            return self.start_value + (self.end_value - self.start_value) * progress
        elif self.schedule_type == "cosine":
            return self.end_value + 0.5 * (self.start_value - self.end_value) * (
                1 + math.cos(math.pi * progress)
            )
        elif self.schedule_type == "exponential":
            # Exponential decay from start_value to end_value
            gamma = math.log(self.end_value / self.start_value) / self.total_steps
            return self.start_value * math.exp(gamma * step)
        else:
            logger.error(f"  unknown schedule_type: {self.schedule_type}")
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

    def state_dict(self):
        return {
            "schedule_type": self.schedule_type,
            "start_value": self.start_value,
            "end_value": self.end_value,
            "param_group_indices": self.param_group_indices,
            "total_steps": self.total_steps,
            "opt_idx": self.opt_idx,
        }

    def load_state_dict(self, state_dict):
        self.schedule_type = state_dict.get("schedule_type", self.schedule_type)
        self.start_value = state_dict.get("start_value", self.start_value)
        self.end_value = state_dict.get("end_value", self.end_value)
        self.opt_idx = state_dict.get("opt_idx", self.opt_idx)
        self.param_group_indices = state_dict.get(
            "param_group_indices", self.param_group_indices
        )
        self.total_steps = state_dict.get("total_steps", self.total_steps)
        logger.info(f"  state restored from checkpoint: {state_dict}")
