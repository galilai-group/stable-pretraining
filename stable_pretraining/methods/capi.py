# Copyright (c) Meta Platforms, Inc. and affiliates.
# Head and schedules adapted from facebookresearch/capi at
# 98b4fa17ee8eec8810c17022df9a27a44845368b, under Apache-2.0.
# See ../backbone/licenses/CAPI-Apache-2.0.txt.
"""CAPI through the shared dict-based Module training loop."""

import copy
import math
from functools import partial
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..module import Module
from ..backbone.capi import CAPIBackbone, capi_backbone
from ..losses.capi import CAPILoss


def _schedule(
    step: int,
    total: int,
    warmup: int,
    base: float,
    final: float,
    start: float = 0,
    truncate: float = 0.8,
) -> float:
    if step >= total:
        return final
    if step < warmup:
        return start + (base - start) * step / max(1, warmup - 1)
    return final + 0.5 * (base - final) * (
        1 + math.cos(math.pi * truncate * (step - warmup) / (total - warmup))
    )


def _student_optimizer(
    params: Any, *, named: list, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    selected = {id(p) for p in params}
    groups = {}
    for name, p in named:
        if id(p) not in selected or not p.requires_grad:
            continue
        lr_mult = 0.2 if "patch_embed" in name else 0 if "rope" in name else 1
        wd_mult = (
            0
            if name.endswith(".bias") or "gamma" in name
            else 0.1
            if "norm" in name and "weight" in name
            else 1
        )
        group = groups.setdefault(
            (lr_mult, wd_mult),
            {"params": [], "lr": lr * lr_mult, "weight_decay": weight_decay * wd_mult},
        )
        group["params"].append(p)
    return torch.optim.AdamW(list(groups.values()), lr=lr, betas=(0.9, 0.95))


class _NormalizedHead(nn.Module):
    """Reference normalized features and weight-normalized prediction head."""

    def __init__(self, dim: int, clusters: int) -> None:
        super().__init__()
        layer = nn.Linear(dim, clusters, bias=False)
        nn.init.trunc_normal_(layer.weight, std=0.02)
        self.last_layer = nn.utils.parametrizations.weight_norm(layer)
        with torch.no_grad():
            self.last_layer.parametrizations.weight.original0.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.last_layer(
            F.normalize(x, dim=-1, eps=1e-6 if x.dtype == torch.float16 else 1e-12)
        )


class CAPI(Module):
    """Cluster and predict latent patches, without labels or a global objective.

    Args:
        backbone: Initialized CAPI encoder/predictor; otherwise build from dimensions.
        embed_dim: Encoder/predictor width.
        depth: Encoder depth.
        num_heads: Attention head count.
        predictor_depth: Cross-attention predictor depth.
        patch_size: Patch side length.
        n_registers: Register count.
        drop_path_rate: Stochastic depth probability.
        num_clusters: Online cluster count.
        mask_ratio: Dropped patch fraction.
        prediction_subsampling: Fraction of dropped positions predicted.
        student_temp: Student prediction temperature.
        target_temp: Assignment temperature.
        clustering_temp: Clustering prediction temperature.
        lr: Student peak learning rate.
        clustering_lr: Clustering peak learning rate.
        weight_decay: Student weight decay; reference clustering decay is 0.01.
        total_steps: Completed optimizer updates in the schedule.
        warmup_steps: Linear warmup length, strictly below total_steps.
        ema_momentum: EMA value after warmup; warms from 1, then increases to 1.
        accumulation_steps: Optimizer frequency; batch used by Sinkhorn is unchanged.
        representation: Teacher (default) or student for full-image embeddings.
        kwargs: Additional Module attributes.

    Note:
        Defaults match the pinned author configuration. The method returns dicts
        and delegates optimization to Module. Its callback observes successful
        student optimizer steps and stores EMA progress in the model checkpoint.
    """

    def __init__(
        self,
        backbone: CAPIBackbone | None = None,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        predictor_depth: int = 12,
        patch_size: int = 14,
        n_registers: int = 16,
        drop_path_rate: float = 0.2,
        num_clusters: int = 16384,
        mask_ratio: float = 0.65,
        prediction_subsampling: float = 0.05,
        student_temp: float = 0.12,
        target_temp: float = 0.06,
        clustering_temp: float = 0.12,
        lr: float = 0.001,
        clustering_lr: float = 0.0005,
        weight_decay: float = 0.1,
        total_steps: int = 500000,
        warmup_steps: int = 50000,
        ema_momentum: float = 0.999,
        accumulation_steps: int = 1,
        representation: str = "teacher",
        **kwargs: Any,
    ) -> None:
        from ..forward import capi

        if not 0 <= warmup_steps < total_steps or accumulation_steps < 1:
            raise ValueError(
                "Require 0 <= warmup_steps < total_steps and positive accumulation"
            )
        if representation not in ("teacher", "student"):
            raise ValueError("representation must be teacher or student")
        if num_clusters < 2 or not 0 <= ema_momentum <= 1:
            raise ValueError("Require at least two clusters and EMA in [0,1]")
        if not 0 < mask_ratio < 1 or not 0 < prediction_subsampling <= 1:
            raise ValueError("Invalid mask ratio or prediction subsampling")
        backbone = (
            backbone
            if backbone is not None
            else capi_backbone(
                embed_dim,
                depth,
                num_heads,
                predictor_depth,
                patch_size,
                n_registers,
                drop_path_rate,
            )
        )
        teacher = copy.deepcopy(backbone).requires_grad_(False)
        teacher._capi_frozen = True
        teacher.eval()
        clustering = nn.Linear(backbone.embed_dim, num_clusters, bias=True)
        nn.init.normal_(clustering.weight, std=1)
        nn.init.zeros_(clustering.bias)
        metadata = dict(
            embed_dim=backbone.embed_dim,
            depth=backbone.encoder.n_blocks,
            num_heads=backbone.encoder.blocks[0].residual1.fn.num_heads,
            predictor_depth=backbone.decoder.n_blocks,
            patch_size=backbone.patch_size,
            n_registers=backbone.n_registers,
            drop_path_rate=1 - backbone.encoder.blocks[0].residual1.keep_prob,
            num_clusters=num_clusters,
            mask_ratio=mask_ratio,
            prediction_subsampling=prediction_subsampling,
            student_temp=student_temp,
            target_temp=target_temp,
            clustering_temp=clustering_temp,
            lr=lr,
            clustering_lr=clustering_lr,
            weight_decay=weight_decay,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            ema_momentum=ema_momentum,
            accumulation_steps=accumulation_steps,
            representation=representation,
        )
        metadata.update(kwargs.pop("hparams", {}) or {})
        super().__init__(
            forward=capi,
            hparams=metadata,
            backbone=backbone,
            teacher=teacher,
            student_head=_NormalizedHead(backbone.pred_dim, num_clusters),
            clustering=clustering,
            capi_loss=CAPILoss(student_temp, target_temp, clustering_temp),
            mask_ratio=mask_ratio,
            prediction_subsampling=prediction_subsampling,
            representation=representation,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            ema_momentum=ema_momentum,
            **kwargs,
        )
        self.register_buffer("capi_updates", torch.zeros((), dtype=torch.long))
        schedule = partial(
            torch.optim.lr_scheduler.LambdaLR,
            lr_lambda=partial(
                _schedule, total=total_steps, warmup=warmup_steps, base=1, final=0
            ),
        )
        self.optim = {
            "student": {
                "modules": r"^(backbone|student_head)(\.|$)",
                "optimizer": partial(
                    _student_optimizer,
                    named=list(self.named_parameters()),
                    lr=lr,
                    weight_decay=weight_decay,
                ),
                "scheduler": schedule,
                "interval": "step",
                "frequency": accumulation_steps,
            },
            "clustering": {
                "modules": r"^clustering(\.|$)",
                "optimizer": partial(
                    torch.optim.AdamW,
                    lr=clustering_lr,
                    betas=(0.9, 0.95),
                    weight_decay=0.01,
                ),
                "scheduler": schedule,
                "interval": "step",
                "frequency": accumulation_steps,
            },
        }

    def configure_callbacks(self) -> list:
        from ..callbacks.capi import CAPIEMA

        return [CAPIEMA()]


__all__ = ["CAPI"]
