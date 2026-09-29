import lightning as pl
from typing import Union
from loguru import logger as logging
import numpy as np
import torchmetrics

from .registry import log as _spt_log
from .utils import log_header


def to_scalar(x):
    if isinstance(x, torchmetrics.Metric):
        return x.compute().item()
    return x.item() if hasattr(x, "item") else x


class EpochMilestones(pl.Callback):
    """PyTorch Lightning callback to stop training if a monitored metric does not meet specified thresholds at given epochs.

    This callback allows you to define "milestones"—specific epochs at which a metric must surpass (or fall below) a given value.
    If the metric fails to meet the requirement at the milestone epoch, training is stopped early.

    Args:
        metric_name (str):
            The name of the metric to monitor (as logged in `trainer.callback_metrics`).
        milestones (dict[int, float]):
            A dictionary mapping epoch numbers (int) to required metric values (float).
            At each specified epoch, the metric is checked against the corresponding value.
        direction (str, optional):
            One of "max" or "min".
            - "max": Training stops if the metric is less than or equal to the milestone value.
            - "min": Training stops if the metric is greater than or equal to the milestone value.
            Default is "max".
        after_validation (bool, optional):
            If True (default), the metric is checked after validation (`on_validation_end`).
            If False, the metric is checked after training (`on_training_end`).

    Raises:
        ValueError: If the specified metric is not found in `trainer.callback_metrics` at the milestone epoch.

    Example:
        >>> milestones = {10: 0.2, 20: 0.5}
        >>> callback = EpochMilestones(
        ...     metric_name="eva/accuracy",
        ...     milestones=milestones,
        ...     direction="max",
        ...     after_validation=True,
        ... )
        >>> trainer = pl.Trainer(callbacks=[callback])

    """

    def __init__(
        self,
        milestones: dict[int, float],
        monitor: Union[list[str], str] = None,
        contains: str = None,
        direction: str = "max",
        after_validation: bool = True,
        strict: bool = True,
        verbose: bool = None,
    ):
        if monitor is None and contains is None:
            raise ValueError("`monitor` and `contains` can't both be None")
        super().__init__()
        if type(monitor) is str:
            monitor = [monitor]
        if type(contains) is str:
            contains = [contains]

        self.monitor = monitor
        self.contains = contains
        self.strict = strict
        self.milestones = milestones
        self.direction = direction
        self.after_validation = after_validation
        from .utils import resolve_verbose

        self.verbose = resolve_verbose(verbose)

    def _check_condition(self, trainer):
        # Get the current epoch
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        # Select metrics by exact or substring match
        if self.monitor:
            matched = {m: metrics.get(m) for m in self.monitor}
        else:
            matched = {
                k: v
                for contain in self.contains
                for k, v in metrics.items()
                if contain in k
            }
        # Sanity check: verify presence
        if trainer.sanity_checking:
            log_header("EpochMilestones")
            logging.info("  sanity checking...")
            logging.info(f"  matched {len(matched)} metrics")
            if not matched:
                msg = f"No metrics found for monitor='{self.monitor}' contains='{self.contains}' in callback_metrics: {list(metrics.keys())}"
                if self.strict:
                    raise RuntimeError(msg)
                else:
                    logging.warning(f"! {msg}")
            logging.success(f"✓ sanity check passed, milestones: {self.milestones}")
            return
        if epoch not in self.milestones:
            logging.info(f"  epoch {epoch} is not in milestones, skipping")
            return
        logging.info(f"  epoch {epoch} is in milestones, checking condition")
        # Retrieve the metric from the logged metrics
        values = list(matched.values())
        # Stop training if the metric is not greater than min_value
        if self.direction == "max":
            final = np.max([to_scalar(x) for x in values])
            logging.info(f"  maximum value: {final}")
            if self.verbose:
                _spt_log(
                    "epoch_milestones/value",
                    float(final),
                    on_step=False,
                    on_epoch=True,
                )
                _spt_log(
                    "epoch_milestones/threshold",
                    float(self.milestones[epoch]),
                    on_step=False,
                    on_epoch=True,
                )
            if final < self.milestones[epoch]:
                logging.warning(
                    f"! value {final} below threshold"
                    f" {self.milestones[epoch]}, stopping"
                )
                trainer.should_stop = True
            else:
                logging.success(
                    f"✓ value {final} above threshold {self.milestones[epoch]}"
                )
        else:
            final = np.min([to_scalar(x) for x in values])
            logging.info(f"  minimum value: {final}")
            if self.verbose:
                _spt_log(
                    "epoch_milestones/value",
                    float(final),
                    on_step=False,
                    on_epoch=True,
                )
                _spt_log(
                    "epoch_milestones/threshold",
                    float(self.milestones[epoch]),
                    on_step=False,
                    on_epoch=True,
                )
            if final > self.milestones[epoch]:
                logging.warning(
                    f"! value {final} above threshold"
                    f" {self.milestones[epoch]}, stopping"
                )
                trainer.should_stop = True
            else:
                logging.success(
                    f"✓ value {final} below threshold {self.milestones[epoch]}"
                )

    def on_train_epoch_end(self, trainer, pl_module):
        if self.after_validation:
            return
        self._check_condition(trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.after_validation:
            return
        self._check_condition(trainer)
