"""MoCo v3: Momentum Contrast for ViTs.

Symmetric InfoNCE between a student-predicted projection and a momentum
target's projection of the other view. No memory queue (the original MoCo
v3 paper showed it isn't needed for ViTs at modern batch sizes).

References:
    Chen, Xie, He. "An Empirical Study of Training Self-Supervised Vision
    Transformers." ICCV 2021. https://arxiv.org/abs/2104.02057
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import TeacherStudentWrapper, from_timm
from stable_pretraining.utils import all_gather


@dataclass
class MoCov3Output(ModelOutput):
    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    prediction: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None


def _mlp(dims: Sequence[int], last_bn: bool = True) -> nn.Module:
    """N-layer Linear-BN-ReLU MLP with optional final BN (no bias)."""
    layers = []
    for i in range(len(dims) - 1):
        is_last = i == len(dims) - 2
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
        if is_last:
            if last_bn:
                layers.append(nn.BatchNorm1d(dims[-1], affine=False))
        else:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _info_nce(p: torch.Tensor, t: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE — one direction; call twice and average for symmetry."""
    p = F.normalize(p, dim=-1)
    t = F.normalize(t, dim=-1)
    # gather targets across DDP for negatives
    t_all = torch.cat(all_gather(t), dim=0)
    logits = p @ t_all.T / temperature
    # positive index for each row is its own row offset by rank * batch
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    B = p.shape[0]
    targets = torch.arange(B, device=p.device) + rank * B
    return F.cross_entropy(logits, targets)


class MoCov3(Module):
    """MoCo v3: ViT-friendly momentum contrastive learning.

    Architecture:
        - Backbone (student) wrapped with EMA teacher.
        - Projector (student) wrapped with EMA teacher.
        - Predictor on the student side only.

    :param encoder_name: timm model or pre-built ``nn.Module``.
    :param projector_dims: 3-layer projector dims (default ``(4096, 4096, 256)``).
    :param predictor_hidden_dim: Predictor hidden dim (default 4096).
    :param temperature: InfoNCE temperature (default 0.2).
    :param ema_decay_start: Initial EMA (default 0.99).
    :param ema_decay_end: Final EMA (default 1.0).
    :param low_resolution: Adapt first conv for low-res input.
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (4096, 4096, 256),
        predictor_hidden_dim: int = 4096,
        temperature: float = 0.2,
        ema_decay_start: float = 0.99,
        ema_decay_end: float = 1.0,
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()

        if isinstance(encoder_name, str):
            base = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        else:
            base = encoder_name

        with torch.no_grad():
            embed_dim = base(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.embed_dim = embed_dim
        self.temperature = temperature

        proj_dims = (embed_dim, *projector_dims)
        self.backbone = TeacherStudentWrapper(
            base,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.projector = TeacherStudentWrapper(
            _mlp(list(proj_dims), last_bn=True),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.predictor = _mlp(
            [projector_dims[-1], predictor_hidden_dim, projector_dims[-1]],
            last_bn=False,
        )

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> MoCov3Output:
        if view2 is None:
            with torch.no_grad():
                emb = self.backbone.forward_teacher(view1)
            return MoCov3Output(
                loss=torch.zeros((), device=emb.device, dtype=emb.dtype),
                embedding=emb.detach(),
            )

        # Student path (gradients)
        s1 = self.backbone.forward_student(view1)
        s2 = self.backbone.forward_student(view2)
        zp1 = self.projector.forward_student(s1)
        zp2 = self.projector.forward_student(s2)
        p1 = self.predictor(zp1)
        p2 = self.predictor(zp2)

        # Teacher (no grad)
        with torch.no_grad():
            t1 = self.backbone.forward_teacher(view1)
            t2 = self.backbone.forward_teacher(view2)
            zt1 = self.projector.forward_teacher(t1)
            zt2 = self.projector.forward_teacher(t2)

        loss = (
            _info_nce(p1, zt2, self.temperature)
            + _info_nce(p2, zt1, self.temperature)
        ) / 2

        return MoCov3Output(
            loss=loss,
            embedding=torch.cat([t1, t2], dim=0).detach(),
            prediction=torch.cat([p1, p2], dim=0),
            target=torch.cat([zt1, zt2], dim=0).detach(),
        )
