from typing import Any

import torch
import torch.distributed as dist
from lightning.pytorch import Callback, LightningModule, Trainer

from .utils import get_data_from_batch_or_outputs


class Sparsity(Callback):
    """Monitor normalized L0 and L1 sparsity of feature representations.

    For each row of an ``[N, D]`` feature tensor, L0 is the fraction of nonzero
    coordinates and L1 is ``(||z||_1 / (||z||_2 + 1e-12))**2 / D``. Both are
    zero for zero vectors, and lower values indicate greater sparsity.
    Statistics are computed in detached float32 and averaged over samples,
    including samples on other distributed ranks.

    Training is monitored at pre-update steps ``0, every_n_steps, ...`` and
    validation at every batch. Metrics are logged as ``fit/l0_{name}`` and
    ``fit/l1_{name}``, with a ``validate/`` prefix during validation. Epoch
    averages are weighted by the global number of monitored samples.

    Args:
        name: Name used in the logged metric keys.
        target: Key containing features in the outputs or batch dictionary.
            Outputs take precedence when both contain the key.
        every_n_steps: Positive training step interval between measurements.
        train_first_view_only: Monitor only the first view's rows during
            training. Requires ``batch["views"]`` to be a list or named
            dictionary of views containing ``"image"`` tensors, and features
            concatenated in the same view order. Validation always uses all rows.
    """

    def __init__(
        self,
        name: str,
        target: str = "projection",
        every_n_steps: int = 50,
        train_first_view_only: bool = False,
    ) -> None:
        super().__init__()
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive")

        self.name = name
        self.target = target
        self.every_n_steps = every_n_steps
        self.train_first_view_only = train_first_view_only
        self._log_this_batch = False

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Capture the logging decision before the optimizer advances the step."""
        self._log_this_batch = trainer.global_step % self.every_n_steps == 0

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Log sparsity for the selected training batches."""
        if self._log_this_batch:
            self._log_sparsity(pl_module, batch, outputs, "fit")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log sparsity for every validation batch."""
        self._log_sparsity(pl_module, batch, outputs, "validate")

    @torch.no_grad()
    def _log_sparsity(
        self,
        pl_module: LightningModule,
        batch: dict[str, Any],
        outputs: dict[str, Any],
        stage: str,
    ) -> None:
        features = get_data_from_batch_or_outputs(
            self.target, batch, outputs, caller_name=self.__class__.__name__
        )
        if stage == "fit" and self.train_first_view_only:
            views = batch["views"]
            first_view = (
                next(iter(views.values())) if isinstance(views, dict) else views[0]
            )
            features = features[: first_view["image"].shape[0]]
        features = features.detach().float()

        l0 = (features != 0).float().mean(dim=1)
        l1_norm = torch.linalg.vector_norm(features, ord=1, dim=1)
        l2_norm = torch.linalg.vector_norm(features, ord=2, dim=1)
        l1 = (l1_norm / (l2_norm + 1e-12)).square() / features.shape[1]
        totals = torch.stack(
            (l0.sum(), l1.sum(), features.new_tensor(features.shape[0]))
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        pl_module.log_dict(
            {
                f"{stage}/l0_{self.name}": totals[0] / totals[2],
                f"{stage}/l1_{self.name}": totals[1] / totals[2],
            },
            on_step=stage == "fit",
            on_epoch=True,
            sync_dist=False,
            batch_size=int(totals[2].item()),
        )
