"""Point-JEPA self-supervised pretraining on XYZ point clouds."""

from functools import partial
import math

import torch
from torch import nn
from timm.layers import DropPath

from stable_pretraining import Module
from stable_pretraining.backbone import FlexibleTransformer, TeacherStudentWrapper
from stable_pretraining.backbone.point_cloud import (
    PointJEPAMasking,
    PointSequencer,
    PointTokenizer,
    PointTokenTransformer,
)
from stable_pretraining.forward import point_jepa


class _PointJEPAEMA(TeacherStudentWrapper):
    """Reference linear EMA schedule driven by the SPT teacher callback."""

    def __init__(self, student: nn.Module, start: float, end: float) -> None:
        super().__init__(
            student,
            warm_init=False,
            base_ema_coefficient=start,
            final_ema_coefficient=end,
        )
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("ema_total_updates", torch.ones((), dtype=torch.long))

    @torch.no_grad()
    def update_ema_coefficient(self, epoch: int, total_epochs: int) -> None:
        """Use optimizer-update progress instead of the generic epoch cosine."""
        progress = min(self.ema_updates.item() / self.ema_total_updates.item(), 1.0)
        coefficient = self.base_ema_coefficient + progress * (
            self.final_ema_coefficient - self.base_ema_coefficient
        )
        self.ema_coefficient.copy_(coefficient if self.ema_updates.item() else 0.0)

    @torch.no_grad()
    def update_teacher(self) -> None:
        """Reproduce the reference's first update and initialization copies."""
        if not self.training:
            return
        if self.ema_updates.item() < 2:
            self.teacher.load_state_dict(self.student.state_dict())
        super().update_teacher()
        self.ema_updates.add_(1)


def _reference_lr_multiplier(
    step: int,
    *,
    updates_per_epoch: int,
    max_epochs: int,
    warmup_epochs: int,
    base_lr: float,
) -> float:
    """Express the reference's epoch warmup/cosine in SPT update units."""
    epoch = step // updates_per_epoch
    if epoch < warmup_epochs:
        return 1e-5 / base_lr + (1 - 1e-5 / base_lr) * epoch / max(warmup_epochs - 1, 1)
    progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
    return (
        1e-6 / base_lr + (1 - 1e-6 / base_lr) * (1 + math.cos(math.pi * progress)) / 2
    )


class PointJEPA(Module):
    """Point-JEPA: joint-embedding predictive learning for XYZ point clouds.

    Processes batches containing ``points`` tensors of shape [B,N,3]. A shared
    PointNet tokenizer embeds local patches, and a transformer predicts masked
    patch features from an EMA teacher using visible context and XYZ positions.

    Args:
        num_groups: FPS patch centers.
        group_size: Neighbors in each patch.
        embed_dim: Encoder token width.
        encoder_depth: Encoder transformer depth.
        encoder_heads: Encoder attention heads.
        predictor_dim: Predictor hidden width.
        predictor_depth: Predictor transformer depth.
        predictor_heads: Predictor attention heads.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        ema_decay_start: Initial EMA coefficient from the released config.
        ema_decay_end: Final EMA coefficient from the released config.
        accumulate_steps: Number of microbatches per optimizer step in SPT manual optimization.

    Note:
        Use ``TeacherStudentCallback`` in the Trainer to update the EMA teacher.
        The released configuration uses EMA 0.9995 to 0.99999; the paper
        supplement instead describes approximately 0.995 to 1.0.
    """

    def __init__(
        self,
        num_groups: int = 64,
        group_size: int = 32,
        embed_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        predictor_dim: int = 192,
        predictor_depth: int = 6,
        predictor_heads: int = 6,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        ema_decay_start: float = 0.9995,
        ema_decay_end: float = 0.99999,
        accumulate_steps: int = 1,
    ) -> None:
        encoder = PointTokenTransformer(embed_dim, encoder_depth, encoder_heads)
        super().__init__(
            forward=point_jepa,
            tokenizer=PointTokenizer(num_groups, group_size, embed_dim),
            sequencer=PointSequencer(),
            masking=PointJEPAMasking(),
            # use custom EMA scheduling to match official implementation
            encoder=_PointJEPAEMA(encoder, ema_decay_start, ema_decay_end),
            encoder_position=nn.Sequential(
                nn.Linear(3, 128), nn.GELU(), nn.Linear(128, embed_dim)
            ),
            predictor_position=nn.Sequential(
                nn.Linear(3, 128), nn.GELU(), nn.Linear(128, predictor_dim)
            ),
            predictor=FlexibleTransformer(
                input_dim=embed_dim,
                hidden_dim=predictor_dim,
                output_dim=embed_dim,
                num_patches=num_groups,
                depth=predictor_depth,
                num_heads=predictor_heads,
                mlp_ratio=4,
                self_attn=True,
                cross_attn=False,
                use_adaln=False,
                pos_embed_type="none",
                drop_path_rate=0.25,
                attn_drop=0.05,
                proj_drop=0.1,
                zero_init_output=False,
                num_prefix_tokens=0,
                add_mask_token=True,
            ),
            optim={
                "optimizer": partial(
                    torch.optim.AdamW, lr=lr, weight_decay=weight_decay
                ),
                "frequency": accumulate_steps,
            },
            hparams={
                "num_groups": num_groups,
                "group_size": group_size,
                "embed_dim": embed_dim,
                "encoder_depth": encoder_depth,
                "encoder_heads": encoder_heads,
                "predictor_dim": predictor_dim,
                "predictor_depth": predictor_depth,
                "predictor_heads": predictor_heads,
                "lr": lr,
                "weight_decay": weight_decay,
                "ema_decay_start": ema_decay_start,
                "ema_decay_end": ema_decay_end,
                "accumulate_steps": accumulate_steps,
            },
        )
        # The reference predictor uses constant drop path and a zero mask token.
        nn.init.zeros_(self.predictor.mask_token)
        for block in self.predictor.blocks:
            block.drop_path1 = DropPath(0.25)
            block.drop_path2 = DropPath(0.25)

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]]:
        """Use reference linear EMA and epoch warmup/cosine through SPT.

        Returns:
            SPT optimizer and scheduler configurations.
        """
        steps = self.trainer.max_steps
        if steps < 1:
            steps = math.ceil(
                self.trainer.estimated_stepping_batches / self.hparams.accumulate_steps
            )
        self.encoder.ema_total_updates.fill_(steps)
        epochs = self.trainer.max_epochs
        updates_per_epoch = 1 if self.trainer.max_steps > 0 else max(1, steps // epochs)
        if self.trainer.max_steps > 0:
            epochs = steps
        warmup_budget = round(epochs * 30 / 500) if self.trainer.max_steps > 0 else 30
        warmup_epochs = 0 if epochs == 1 else max(1, min(epochs - 1, warmup_budget))
        self.optim["scheduler"] = partial(
            torch.optim.lr_scheduler.LambdaLR,
            lr_lambda=partial(
                _reference_lr_multiplier,
                updates_per_epoch=updates_per_epoch,
                max_epochs=epochs,
                warmup_epochs=warmup_epochs,
                base_lr=self.hparams.lr,
            ),
        )
        return super().configure_optimizers()
