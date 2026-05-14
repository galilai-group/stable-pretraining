import threading
import warnings
from typing import Optional, Dict, Any, List, Tuple

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

_lock = threading.Lock()
_MODULE_REGISTRY: Dict[str, LightningModule] = {}
_METRIC_BUFFER: Dict[str, List[Tuple[str, Any, Dict[str, Any]]]] = {}
_DICT_BUFFER: Dict[str, List[Tuple[tuple, Dict[str, Any]]]] = {}
_IN_STEP: Dict[str, bool] = {}


def get_module(name: str = "default") -> Optional[LightningModule]:
    """Retrieve a registered module."""
    return _MODULE_REGISTRY.get(name)


def _flush_buffer(module_name: str = "default") -> None:
    """Flush buffered metrics into the module's logger.

    Called at the start of each batch (train/val/test/predict) when the module
    is in a valid logging context.  Metrics that were logged outside of a step
    are replayed here so they are not lost.
    """
    module = _MODULE_REGISTRY.get(module_name)
    if module is None:
        return

    with _lock:
        metrics = _METRIC_BUFFER.pop(module_name, [])
        dict_metrics = _DICT_BUFFER.pop(module_name, [])

    for name, value, kwargs in metrics:
        try:
            module.log(name, value, **kwargs)
        except Exception:
            warnings.warn(
                f"Failed to flush buffered metric '{name}' — metric dropped",
                stacklevel=2,
            )

    for args, kwargs in dict_metrics:
        try:
            module.log_dict(*args, **kwargs)
        except Exception:
            warnings.warn(
                "Failed to flush buffered dict metrics — metrics dropped",
                stacklevel=2,
            )


def log(name: str, value: Any, module_name: str = "default", **kwargs) -> None:
    """Log a metric using the registered module.

    Safe to call from anywhere.  If no module is registered or the call happens
    outside a training/validation/test step, the metric is buffered and will be
    flushed at the start of the next step.
    """
    module = _MODULE_REGISTRY.get(module_name)
    if module is None:
        warnings.warn(
            f"log('{name}') called but no module registered — metric dropped",
            stacklevel=2,
        )
        return

    if _IN_STEP.get(module_name, False):
        module.log(name, value, **kwargs)
    else:
        with _lock:
            _METRIC_BUFFER.setdefault(module_name, []).append((name, value, kwargs))
        warnings.warn(
            f"log('{name}') called outside a training/validation step"
            " — metric buffered for next step",
            stacklevel=2,
        )


def log_dict(*args, module_name: str = "default", **kwargs) -> None:
    """Log a dict of metrics using the registered module.

    Same safety guarantees as :func:`log` — buffered when called outside a step.
    """
    module = _MODULE_REGISTRY.get(module_name)
    if module is None:
        warnings.warn(
            "log_dict() called but no module registered — metrics dropped",
            stacklevel=2,
        )
        return

    if _IN_STEP.get(module_name, False):
        module.log_dict(*args, **kwargs)
    else:
        with _lock:
            _DICT_BUFFER.setdefault(module_name, []).append((args, kwargs))
        warnings.warn(
            "log_dict() called outside a training/validation step"
            " — metrics buffered for next step",
            stacklevel=2,
        )


class ModuleRegistryCallback(Callback):
    """Callback that automatically registers the module for global logging access.

    Manages the lifecycle of the global module registry: registers the module
    on ``setup``, tracks valid-logging-step windows via batch hooks, flushes
    any buffered metrics at the start of each batch, and cleans everything up
    on ``teardown``.
    """

    def __init__(self, name: str = "default"):
        self.name = name

    # -- lifecycle ------------------------------------------------------------

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Register module at the start of any stage (fit, validate, test, predict)."""
        with _lock:
            _MODULE_REGISTRY[self.name] = pl_module

    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Clean up registry when done."""
        with _lock:
            dropped = _METRIC_BUFFER.pop(self.name, [])
            dropped_dict = _DICT_BUFFER.pop(self.name, [])
            _MODULE_REGISTRY.pop(self.name, None)
            _IN_STEP.pop(self.name, None)

        if dropped or dropped_dict:
            n = len(dropped) + len(dropped_dict)
            warnings.warn(
                f"{n} buffered metric(s) were dropped at teardown"
                " because no valid logging step occurred after they were buffered",
                stacklevel=2,
            )

    # -- step tracking --------------------------------------------------------

    def _enter_step(self, trainer, pl_module):
        _IN_STEP[self.name] = True
        _flush_buffer(self.name)

    def _exit_step(self, trainer, pl_module):
        _IN_STEP[self.name] = False

    # Training
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._enter_step(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Keep _IN_STEP=True through batch_end so other callbacks that log here
        # (e.g. TeacherStudentCallback writing the EMA coefficient) inline-log
        # with the correct trainer.global_step instead of buffering and being
        # flushed at the next batch with a stale step. Exit is moved to the
        # epoch boundary, where between-epoch logs are still legal.
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        self._exit_step(trainer, pl_module)

    # Validation
    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._enter_step(trainer, pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        pass  # see on_train_batch_end note

    def on_validation_epoch_end(self, trainer, pl_module):
        self._exit_step(trainer, pl_module)

    # Test
    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._enter_step(trainer, pl_module)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        self._exit_step(trainer, pl_module)

    # Predict
    def on_predict_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._enter_step(trainer, pl_module)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        pass

    def on_predict_epoch_end(self, trainer, pl_module):
        self._exit_step(trainer, pl_module)
