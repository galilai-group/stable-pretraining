"""Callback for automatic TeacherStudentWrapper EMA updates."""

import lightning as pl
import torch
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

    Note:
        Order-sensitive. The EMA update fires inside ``on_train_batch_end``.
        Place this callback **after** any callback that reads the teacher's
        parameters in the same training step (e.g., ``OnlineProbe`` or
        ``OnlineKNN`` consuming teacher embeddings), or those callbacks
        will see the pre-update teacher state. Reading state from the
        *next* step is fine — the order rule only matters within a single
        step.

    Args:
        update_frequency: How often to update the teacher network, measured in
            optimizer steps. Default is ``1`` (every step).
        update_after_backward: If ``True``, the EMA update fires after the backward
            pass on an optimizer-step boundary (before the main optimizer step).
            Accumulation-only batches and callback-owned optimizers do not trigger
            additional updates. If ``False``, it fires after the optimizer step.
            Default is ``False``.
        verbose: If ``True``, log the EMA coefficient and update count each step.
            ``None`` inherits the global ``spt`` verbosity setting.

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

        if not isinstance(update_frequency, int) or update_frequency < 1:
            raise ValueError("update_frequency must be a positive integer")
        self.update_frequency = update_frequency
        self.update_after_backward = update_after_backward
        self.verbose = resolve_verbose(verbose)
        self._wrapper_found = False
        self._last_global_step = -1

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log if TeacherStudentWrapper instances are found."""
        # Reset counters at the start of fit
        self._last_global_step = trainer.global_step
        wrapper_count = self._count_teacher_student_wrappers(pl_module)
        self._wrapper_found = wrapper_count > 0
        if self._wrapper_found:
            log_header("TeacherStudentCallback")
            logging.info(
                f"  found {wrapper_count} TeacherStudentWrapper instance(s). "
                f"Updates every {self.update_frequency} optimizer step(s)."
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
            if (
                current_step > self._last_global_step
                and current_step % self.update_frequency == 0
            ):
                self._update_all_wrappers(trainer, pl_module)
                self._last_global_step = current_step

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Update before the main optimizer, after its accumulation window.

        Args:
            trainer: Trainer owning the optimizer-step counter.
            pl_module: Model containing the teacher/student wrappers.
            optimizer: Optimizer about to step.
        """
        if not self.update_after_backward:
            return
        # Backward hooks also fire on accumulation-only batches. Callback-owned
        # optimizers must not cause extra EMA updates either.
        optimizers = trainer.optimizers
        index = (
            pl_module._pick_global_step_ticker(optimizers)
            if hasattr(pl_module, "_pick_global_step_ticker")
            else 0
        )
        if optimizer is not optimizers[index]:
            return
        next_step = trainer.global_step + 1
        if (
            next_step > self._last_global_step
            and next_step % self.update_frequency == 0
        ):
            self._update_all_wrappers(trainer, pl_module)
            self._last_global_step = next_step

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
