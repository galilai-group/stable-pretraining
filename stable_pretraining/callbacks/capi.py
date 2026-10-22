"""Successful-step EMA updates for CAPI, including manual accumulation."""

import torch
from lightning.pytorch import Callback


class CAPIEMA(Callback):
    """Update the frozen encoder after each completed student optimizer step.

    Note:
        For manually assembled Modules, provide backbone, teacher, capi_updates
        (a registered scalar long buffer), total_steps, warmup_steps, and
        ema_momentum, and name the student optimizer ``student``. Progress lives
        in the Module state, so checkpoint restore cannot restart the schedule.
    """

    def __init__(self) -> None:
        super().__init__()
        self._handle = None

    def on_fit_start(self, trainer, pl_module) -> None:
        if sum(isinstance(cb, CAPIEMA) for cb in trainer.callbacks) != 1:
            raise ValueError("CAPI requires exactly one CAPIEMA callback")
        index = next(
            (
                i
                for i, name in pl_module._optimizer_index_to_name.items()
                if name == "student"
            ),
            None,
        )
        if index is None:
            raise ValueError("CAPIEMA requires a student optimizer")
        optimizer = trainer.optimizers[index]
        teacher_ids = {id(p) for p in pl_module.teacher.parameters()}
        student_ids = {
            id(p) for p in pl_module.backbone.parameters() if p.requires_grad
        }
        actual_ids = {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }
        if teacher_ids & actual_ids or not student_ids <= actual_ids:
            raise ValueError("Invalid CAPI student optimizer membership")
        pl_module.teacher.requires_grad_(False).eval()
        if self._handle is not None:
            self._handle.remove()
        self._handle = optimizer.register_step_post_hook(
            lambda *_: self._update(pl_module)
        )

    @torch.no_grad()
    def _update(self, module) -> None:
        from ..methods.capi import _schedule

        step = int(module.capi_updates)
        momentum = _schedule(
            step,
            module.total_steps,
            module.warmup_steps,
            module.ema_momentum,
            1,
            start=1,
        )
        for teacher, student in zip(
            module.teacher.parameters(), module.backbone.parameters(), strict=True
        ):
            teacher.lerp_(student.detach(), 1 - momentum)
        module.capi_updates.add_(1)
        module.log("train/ema_momentum", momentum, on_step=True, on_epoch=False)

    def teardown(self, trainer, pl_module, stage) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


__all__ = ["CAPIEMA"]
