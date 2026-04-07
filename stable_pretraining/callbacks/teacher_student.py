"""Callback for automatic TeacherStudentWrapper EMA updates."""

import lightning as pl
from lightning.pytorch.callbacks import Callback
from loguru import logger as logging

from .registry import log as _spt_log
from .utils import log_header


class TeacherStudentCallback(Callback):
    """Automatically updates TeacherStudentWrapper instances during training.

    This callback handles the EMA (Exponential Moving Average) updates for any
    TeacherStudentWrapper instances found in the model. It updates both the teacher
    parameters and the EMA coefficient schedule.

    The callback automatically detects all TeacherStudentWrapper instances in the
    model hierarchy and updates them at the appropriate times during training.

    Args:
        update_frequency (int): How often to update the teacher (in batches).
            Default is 1 (every batch).
        update_after_backward (bool): If True, updates happen after backward pass.
            If False, updates happen after optimizer step. Default is True.

    Example:
        >>> backbone = ResNet18()
        >>> wrapped_backbone = TeacherStudentWrapper(backbone)
        >>> module = ssl.Module(backbone=wrapped_backbone, ...)
        >>> trainer = pl.Trainer(callbacks=[TeacherStudentCallback()])
    """

    def __init__(
        self,
        update_frequency: int = 1,
        update_after_backward: bool = False,
        verbose: bool = None,
    ):
        super().__init__()
        from .utils import resolve_verbose

        self.update_frequency = update_frequency
        self.update_after_backward = update_after_backward
        self.verbose = resolve_verbose(verbose)
        self._wrapper_found = False
        # Track optimizer-step progress and accumulation steps
        self._last_global_step = -1
        self._backward_calls = 0

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log if TeacherStudentWrapper instances are found."""
        # Reset counters at the start of fit
        self._last_global_step = -1
        self._backward_calls = 0
        wrapper_count = self._count_teacher_student_wrappers(pl_module)
        if wrapper_count > 0:
            self._wrapper_found = True
            log_header("TeacherStudentCallback")
            logging.info(
                f"  found {wrapper_count} TeacherStudentWrapper instance(s). "
                f"Updates every {self.update_frequency} batch(es)."
            )
        else:
            logging.warning(
                "! no TeacherStudentWrapper instances found in model. "
                "This callback will have no effect."
            )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Update teacher models after training batch if update_after_backward is False."""
        if not self.update_after_backward:
            # Only update after an optimizer step (global_step increments on optimizer step)
            current_step = trainer.global_step
            if current_step != self._last_global_step and self._should_update(
                current_step
            ):
                self._update_all_wrappers(trainer, pl_module)
                self._last_global_step = current_step

    def on_after_backward(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Update teacher models after backward pass if update_after_backward is True."""
        if self.update_after_backward:
            # Use an internal counter to respect update_frequency under gradient accumulation
            self._backward_calls += 1
            if self._should_update(self._backward_calls - 1):
                self._update_all_wrappers(trainer, pl_module)

    def _should_update(self, batch_idx: int) -> bool:
        """Check if we should update on this batch."""
        return (batch_idx + 1) % self.update_frequency == 0

    def _update_all_wrappers(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Find and update all TeacherStudentWrapper instances."""
        if not self._wrapper_found:
            return

        for module in pl_module.modules():
            # Use duck typing to support any module with these methods
            if hasattr(module, "update_teacher") and callable(
                getattr(module, "update_teacher")
            ):
                # Update EMA coefficient first (use current epoch's value), then update teacher parameters via EMA
                if hasattr(module, "update_ema_coefficient") and callable(
                    getattr(module, "update_ema_coefficient")
                ):
                    module.update_ema_coefficient(
                        trainer.current_epoch, trainer.max_epochs
                    )
                module.update_teacher()

                # Log EMA coefficient if available
                if self.verbose and hasattr(module, "ema_coefficient"):
                    _spt_log(
                        f"teacher_student/{getattr(module, 'name', 'ema')}_coefficient",
                        float(module.ema_coefficient),
                        on_step=True,
                        on_epoch=False,
                    )

                # Mark that updates are happening (for warning system)
                if hasattr(module, "_mark_updated"):
                    module._mark_updated()

    def _count_teacher_student_wrappers(self, pl_module: pl.LightningModule) -> int:
        """Count the number of TeacherStudentWrapper instances in the model."""
        count = 0
        for module in pl_module.modules():
            if hasattr(module, "update_teacher") and hasattr(module, "teacher"):
                count += 1
        return count
