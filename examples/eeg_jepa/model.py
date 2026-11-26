"""Reference-objective forward and the existing callback's EMA bridge."""

from functools import partial
from typing import Any

import torch
from torch import nn

import stable_pretraining as spt

from .config import Config, warmup_factor


class TeacherBridge(nn.Module):
    """Expose reference parameter-only, step-linear EMA to TeacherStudentCallback."""

    def __init__(self, model: nn.Module, config: Config) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.register_buffer("completed_steps", torch.zeros((), dtype=torch.long))

    @property
    def teacher(self) -> nn.Module:
        """Expose the existing frozen target without registering it twice."""
        return self.model.target_encoder

    def update_teacher(self) -> None:
        """Update after the optimizer, using the reference's pre-increment step."""
        step = int(self.completed_steps.item())
        self.model.update_target_encoder(self.config.momentum(step))
        self.completed_steps.add_(1)


def forward(self: spt.Module, batch: dict[str, Any], stage: str) -> dict[str, Any]:
    """Compute the unchanged hierarchical objective with trainer normalization.

    Args:
        self: Module containing the reference model bridge and loss callable.
        batch: Microvolt patches, context/target indices and valid-token grid.
        stage: Lightning stage; teacher mutation belongs to the callback.

    Returns:
        Predictions, detached teacher targets, objective terms and total loss.

    Note:
        Like the reference trainer, inputs are divided by 100 and the frozen
        teacher follows the model's train/eval mode, including dropout.
    """
    config = self.eeg.config
    step = int(self.eeg.completed_steps.item())
    x = batch["eeg"] / 100.0
    context, target = batch["context_indices"], batch["target_indices"]
    mask = self.binary_mask(target, x.shape[0], x.shape[1], config.patches)
    outputs = self.eeg.model(x, context, target, mask)
    valid = batch["valid_grid"].flatten(1)
    factor = warmup_factor(step, config.vcreg_warmup_steps)
    loss, prediction, consistency, stats = self.reference_loss(
        *outputs,
        context,
        target,
        patch_num=config.patches,
        lambda_ctx=config.lambda_ctx * warmup_factor(step, config.context_warmup_steps),
        ctx_valid_mask=valid.gather(1, context),
        tgt_valid_mask=valid.gather(1, target),
        lambda_var=config.lambda_var * factor,
        lambda_cov=config.lambda_cov * factor,
        vcreg_gamma=config.vcreg_gamma,
        vcreg_sample_size=config.vcreg_sample_size,
        return_stats=True,
    )
    return dict(
        loss=loss,
        context_predictions=outputs[0],
        target_predictions=outputs[1],
        context_targets=outputs[2],
        target_targets=outputs[3],
        prediction_loss=prediction,
        context_loss=consistency,
        stats=stats,
    )


def build_module(reference: Any, config: Config) -> spt.Module:
    """Wire the pinned model into Module without changing public library APIs.

    Args:
        reference: Verified modules returned by load_reference.
        config: Small synthetic model and optimizer-step schedule.

    Returns:
        Module ready for the existing TeacherStudentCallback and Manager.
    """
    return spt.Module(
        forward=forward,
        eeg=TeacherBridge(reference.model.EEGJEPA(**config.model_kwargs()), config),
        binary_mask=reference.model.indices_to_binary_mask,
        reference_loss=reference.model.compute_structured_latent_loss,
        optim=dict(
            optimizer=partial(
                torch.optim.AdamW, lr=config.lr, weight_decay=config.weight_decay
            ),
            scheduler=partial(
                torch.optim.lr_scheduler.LambdaLR, lr_lambda=config.lr_factor
            ),
        ),
    )
