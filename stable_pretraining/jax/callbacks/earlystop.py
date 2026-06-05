"""Early-stopping callback for the JAX backend.

JAX port of the monitor-and-stop logic in
:mod:`stable_pretraining.callbacks.earlystop`. Watches a metric in
``trainer.callback_metrics`` and sets ``trainer.should_stop`` (which the
:class:`~stable_pretraining.jax.Trainer` loop honors) when it stops improving.
"""

from loguru import logger as logging

from ..trainer import Callback, Trainer


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: Key in ``trainer.callback_metrics`` to watch (e.g.
            ``"val/linear_probe_acc"`` or ``"fit/loss"``).
        mode: ``"min"`` (lower is better) or ``"max"``. Default ``"min"``.
        patience: Epochs with no improvement before stopping. Default ``3``.
        min_delta: Minimum change counted as an improvement. Default ``0.0``.
    """

    def __init__(
        self, monitor: str, mode: str = "min", patience: int = 3, min_delta=0.0
    ):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'.")
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf") if mode == "min" else float("-inf")
        self.wait = 0

    def _improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def on_validation_epoch_end(self, trainer: Trainer, module):
        if self.monitor not in trainer.callback_metrics:
            return
        value = trainer.callback_metrics[self.monitor]
        if self._improved(value):
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logging.info(
                    f"  EarlyStopping: '{self.monitor}' did not improve for "
                    f"{self.patience} epochs (best={self.best:.4f}) — stopping."
                )
                trainer.should_stop = True
