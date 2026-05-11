"""VICReg: Variance-Invariance-Covariance Regularization.

Self-supervised learning by enforcing three criteria on the projected
embeddings of two augmented views:
- **Invariance** to augmentations (MSE between views)
- **Variance** preservation (per-dimension std hinge loss)
- **Covariance** decorrelation (off-diagonal cross-covariance penalty)

References:
    Bardes, Ponce, LeCun. "VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning." ICLR 2022.
    https://arxiv.org/abs/2105.04906
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm
from stable_pretraining.losses import VICRegLoss


@dataclass
class VICRegOutput(ModelOutput):
    """Output from VICReg forward pass.

    :ivar loss: VICReg loss (0 in eval mode)
    :ivar embedding: Backbone features [B, D] (eval) or [2B, D] (train)
    :ivar projection: Projector outputs [2B, P] (None in eval)
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None


def _build_vicreg_projector(in_dim: int, hidden_dims: Sequence[int]) -> nn.Module:
    """3-layer Linear-BN-ReLU projector ending in a bias-free Linear.

    The original VICReg recipe uses (8192, 8192, 8192) with no final BN.
    """
    if len(hidden_dims) < 1:
        raise ValueError("hidden_dims must contain at least one entry")
    layers = []
    prev = in_dim
    for i, dim in enumerate(hidden_dims):
        is_last = i == len(hidden_dims) - 1
        if is_last:
            layers.append(nn.Linear(prev, dim, bias=False))
        else:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
        prev = dim
    return nn.Sequential(*layers)


class VICReg(Module):
    """VICReg: variance-invariance-covariance self-supervised learning.

    :param encoder_name: timm model name or pre-built ``nn.Module``.
    :param projector_dims: Hidden + output dims for the projector.
        Default ``(8192, 8192, 8192)`` matches the ResNet50 paper recipe.
    :param sim_coeff: Invariance term weight (default 25.0).
    :param std_coeff: Variance term weight (default 25.0).
    :param cov_coeff: Covariance term weight (default 1.0).
    :param low_resolution: Adapt first conv for 32x32 inputs (CIFAR-style).
    :param pretrained: Load pretrained timm weights for the encoder.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (8192, 8192, 8192),
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
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

        self.projector = _build_vicreg_projector(embed_dim, list(projector_dims))
        self.vicreg_loss = VICRegLoss(
            sim_coeff=sim_coeff, std_coeff=std_coeff, cov_coeff=cov_coeff
        )

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> VICRegOutput:
        if view2 is None:
            embedding = self.backbone(view1)
            return VICRegOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
                projection=None,
            )

        h1 = self.backbone(view1)
        h2 = self.backbone(view2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        loss = self.vicreg_loss(z1, z2)
        return VICRegOutput(
            loss=loss,
            embedding=torch.cat([h1, h2], dim=0),
            projection=torch.cat([z1, z2], dim=0),
        )
