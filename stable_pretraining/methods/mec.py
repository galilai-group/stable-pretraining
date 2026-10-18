"""MEC: Maximum Entropy Coding.

Self-supervised learning that maximizes the entropy of the representation,
estimated by the coding length of a lossy compression of the batch. The
architecture follows BYOL: the student has a backbone, projector, and
predictor; the teacher (EMA of student) has only a backbone and projector.
The loss is a Taylor-series estimate of ``log det(I + λ C)`` over the
cross-view correlation between student predictions and teacher projections.

References:
    Liu et al. "Self-Supervised Learning via Maximum Entropy Coding."
    NeurIPS 2022. https://arxiv.org/abs/2210.11464
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import TeacherStudentWrapper, from_timm
from stable_pretraining.losses import MECLoss
from stable_pretraining.utils import get_world_size


@dataclass
class MECOutput(ModelOutput):
    """Output from MEC forward pass.

    :ivar loss: Symmetric MEC loss (0 in eval mode)
    :ivar embedding: Teacher backbone features (always detached)
    :ivar prediction: Student predictor output [2B, P] (None in eval)
    :ivar target: Teacher projector output [2B, P] (None in eval, detached)
    :ivar entropy: Coding-length entropy estimate (None in eval, detached)
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    prediction: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None


def _projector(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    """MEC projector: 3-layer MLP with BN, no final ReLU (as in SimSiam)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim, bias=False),
        nn.BatchNorm1d(out_dim, affine=False),
    )


def _predictor(dim: int, hidden_dim: int) -> nn.Module:
    """MEC predictor: 2-layer MLP with bottleneck."""
    return nn.Sequential(
        nn.Linear(dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, dim),
    )


class MEC(Module):
    """MEC self-supervised learning with EMA target network.

    :param encoder_name: timm model name or pre-built ``nn.Module``.
    :param projector_dim: Projector hidden + output dim (default 2048).
    :param predictor_hidden_dim: Predictor bottleneck dim (default 512).
    :param eps: Coding distortion ``ε``; ``1/λ = m ε / d`` (default 64).
    :param taylor_order: Taylor expansion terms for ``log det`` (default 4).
    :param symmetric: Symmetrize the cross-correlation ``C`` so the Taylor
        series stays bounded; see :class:`~stable_pretraining.losses.MECLoss`.
        Recommended for small-scale training (default False, as in the paper).
    :param warmup_epochs: Epochs over which ``1/λ`` decays linearly from
        ``warmup_lambda_scale / λ`` to ``1/λ`` (default 10).
    :param warmup_lambda_scale: Starting multiplier on ``1/λ`` (default 8).
    :param ema_decay_start: Initial EMA decay (default 0.996).
    :param ema_decay_end: Final EMA decay (default 1.0).
    :param low_resolution: Adapt first conv for low-res input.
    :param pretrained: Load pretrained timm weights.

    Note:
        Use :class:`~stable_pretraining.callbacks.TeacherStudentCallback`
        to drive teacher EMA updates during training. The ``1/λ`` warm-up
        keeps the Taylor expansion convergent early in training; it steps
        once per epoch, whereas the reference implementation steps per
        iteration.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dim: int = 2048,
        predictor_hidden_dim: int = 512,
        eps: float = 64.0,
        taylor_order: int = 4,
        symmetric: bool = False,
        warmup_epochs: int = 10,
        warmup_lambda_scale: float = 8.0,
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 1.0,
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()

        if isinstance(encoder_name, str):
            base_backbone = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        else:
            base_backbone = encoder_name

        with torch.no_grad():
            embed_dim = base_backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.embed_dim = embed_dim

        self.backbone = TeacherStudentWrapper(
            base_backbone,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.projector = TeacherStudentWrapper(
            _projector(embed_dim, projector_dim, projector_dim),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.predictor = _predictor(projector_dim, predictor_hidden_dim)
        self.mec_loss = MECLoss(eps=eps, order=taylor_order, symmetric=symmetric)
        self.warmup_epochs = warmup_epochs
        self.warmup_lambda_scale = warmup_lambda_scale

    def _lambda_inv_scale(self) -> float:
        if self.current_epoch >= self.warmup_epochs:
            return 1.0
        progress = self.current_epoch / self.warmup_epochs
        return self.warmup_lambda_scale + progress * (1.0 - self.warmup_lambda_scale)

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> MECOutput:
        if view2 is None:
            with torch.no_grad():
                embedding = self.backbone.forward_teacher(view1)
            return MECOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding.detach(),
            )

        s1 = self.backbone.forward_student(view1)
        s2 = self.backbone.forward_student(view2)
        p1 = self.predictor(self.projector.forward_student(s1))
        p2 = self.predictor(self.projector.forward_student(s2))

        with torch.no_grad():
            t1 = self.backbone.forward_teacher(view1)
            t2 = self.backbone.forward_teacher(view2)
            zt1 = self.projector.forward_teacher(t1)
            zt2 = self.projector.forward_teacher(t2)

        scale = self._lambda_inv_scale()
        loss = (self.mec_loss(p1, zt2, scale) + self.mec_loss(p2, zt1, scale)) / 2

        batch_size = p1.size(0) * get_world_size()
        dim = p1.size(1)
        lambda_inv = self.mec_loss.lambda_inv(batch_size, dim) * scale
        entropy = -loss.detach() * (batch_size + dim) / 2 / lambda_inv

        return MECOutput(
            loss=loss,
            embedding=torch.cat([t1, t2], dim=0).detach(),
            prediction=torch.cat([p1, p2], dim=0),
            target=torch.cat([zt1, zt2], dim=0).detach(),
            entropy=entropy,
        )
