"""CorInfoMax: Correlative Information Maximization.

Self-supervised learning by maximizing the correlative information between
the projections of two augmented views:
- **Big-bang** term: maximize the log-determinant of running (EMA) estimates
  of each view's projection covariance, which prevents dimensional collapse
- **Attraction** term: MSE between the L2-normalized projections

References:
    Ozsoy, Hamdan, Arik, Yuret, Erdogan. "Self-Supervised Learning with an
    Information Maximization Criterion." NeurIPS 2022.
    https://arxiv.org/abs/2209.07999

    Reference implementation: https://github.com/serdarozsoy/corinfomax-ssl
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm
from stable_pretraining.losses import CorInfoMaxLoss


@dataclass
class CorInfoMaxOutput(ModelOutput):
    """Output from CorInfoMax forward pass.

    :ivar loss: CorInfoMax loss (0 in eval mode)
    :ivar embedding: Backbone features [B, D] (eval) or [2B, D] (train)
    :ivar projection: Projector outputs [2B, P] (None in eval)
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None


def _build_corinfomax_projector(in_dim: int, hidden_dims: Sequence[int]) -> nn.Module:
    """Linear-ReLU MLP ending in a bias-free Linear, without batch norm.

    This matches the paper's projector, e.g. (2048, 2048, 64) on CIFAR-10.
    """
    if len(hidden_dims) < 1:
        raise ValueError("hidden_dims must contain at least one entry")
    layers = []
    prev = in_dim
    for i, dim in enumerate(hidden_dims):
        if i == len(hidden_dims) - 1:
            layers.append(nn.Linear(prev, dim, bias=False))
        else:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ReLU(inplace=True))
        prev = dim
    return nn.Sequential(*layers)


class CorInfoMax(Module):
    """CorInfoMax: correlative information maximization SSL.

    The paper's CIFAR-10 recipe uses a ResNet-18 with
    ``encoder_name="resnet18", low_resolution=True`` and the defaults below.

    :param encoder_name: timm model name or pre-built ``nn.Module``.
    :param projector_dims: Hidden + output dims for the projector.
        Default ``(2048, 2048, 64)`` matches the CIFAR-10 recipe.
    :param sim_coeff: Attraction term weight alpha (default 250.0).
    :param cov_coeff: Big-bang term weight (default 1.0).
    :param cov_momentum: Weight of the previous covariance estimate in the
        EMA (default 0.01).
    :param mean_momentum: Weight of the previous mean estimate in the EMA
        (default 0.01).
    :param init_scale: Initial covariance estimate ``init_scale * I``
        (default 1.0).
    :param epsilon: Diagonal perturbation before the log-determinant
        (default 1e-8).
    :param low_resolution: Adapt first conv for 32x32 inputs (CIFAR-style).
    :param pretrained: Load pretrained timm weights for the encoder.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 2048, 64),
        sim_coeff: float = 250.0,
        cov_coeff: float = 1.0,
        cov_momentum: float = 0.01,
        mean_momentum: float = 0.01,
        init_scale: float = 1.0,
        epsilon: float = 1e-8,
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()

        if isinstance(encoder_name, str):
            self.backbone = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        else:
            self.backbone = encoder_name

        with torch.no_grad():
            embed_dim = self.backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.embed_dim = embed_dim

        self.projector = _build_corinfomax_projector(embed_dim, list(projector_dims))
        self.corinfomax_loss = CorInfoMaxLoss(
            proj_dim=projector_dims[-1],
            sim_coeff=sim_coeff,
            cov_coeff=cov_coeff,
            cov_momentum=cov_momentum,
            mean_momentum=mean_momentum,
            init_scale=init_scale,
            epsilon=epsilon,
        )

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> CorInfoMaxOutput:
        if view2 is None:
            embedding = self.backbone(view1)
            return CorInfoMaxOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
                projection=None,
            )

        h1 = self.backbone(view1)
        h2 = self.backbone(view2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        loss = self.corinfomax_loss(z1, z2)
        return CorInfoMaxOutput(
            loss=loss,
            embedding=torch.cat([h1, h2], dim=0),
            projection=torch.cat([z1, z2], dim=0),
        )
