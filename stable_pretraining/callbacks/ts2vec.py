"""Optimizer-step cumulative averaging for TS2Vec."""

from typing import Any

import lightning as pl
import torch


class TS2VecAveraging(pl.Callback):
    """Update ``averaged_backbone`` only when the backbone optimizer steps.

    Attach this callback when composing ``Module(forward=ts2vec, ...)`` with
    a frozen ``torch.optim.swa_utils.AveragedModel`` named
    ``averaged_backbone``. ``TS2Vec`` installs it automatically.

    Note:
        A post-step hook respects optimizer frequency and AMP-skipped steps,
        including fused optimizers that expose ``found_inf`` during the step.
        Callback-owned probe optimizers must not update the encoder average.
        Encoder parameters must all belong to exactly one optimizer.
    """

    def __init__(self) -> None:
        self._handle = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Register on the sole optimizer owning all trainable encoder weights."""
        self._remove_hook()
        backbone = pl_module.backbone
        average = pl_module.averaged_backbone
        wanted = {id(p) for p in backbone.parameters() if p.requires_grad}
        owners = []
        for optimizer in trainer.optimizers:
            owned = {id(p) for g in optimizer.param_groups for p in g["params"]}
            if wanted & owned:
                owners.append((optimizer, owned))
        if not wanted or len(owners) != 1 or not wanted <= owners[0][1]:
            raise ValueError(
                "TS2Vec averaging requires one optimizer for the entire backbone"
            )
        # Refresh the initial copy after distributed parameter synchronization.
        if average.n_averaged.item() <= 1:
            average.module.load_state_dict(backbone.state_dict())
            average.n_averaged.fill_(1)

        @torch.no_grad()
        def update(
            optimizer: torch.optim.Optimizer, args: tuple, kwargs: dict[str, Any]
        ) -> None:
            # GradScaler skips ordinary optimizer.step() calls on overflow. For
            # AMP-aware fused optimizers, it calls step() and supplies found_inf
            # so the fused kernel can skip its parameter update. The post-step
            # hook still runs in that path, so do not count the stale weights.
            found_inf = getattr(optimizer, "found_inf", None)
            if found_inf is not None and found_inf.item() != 0:
                return
            average.update_parameters(backbone)

        self._handle = owners[0][0].register_step_post_hook(update)

    def _remove_hook(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._remove_hook()

    def teardown(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        self._remove_hook()
