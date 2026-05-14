"""Barlow Twins: Self-Supervised Learning via Redundancy Reduction.

Drives the cross-correlation matrix between projected embeddings of two
augmented views toward the identity, decorrelating features while staying
invariant to augmentations.

References:
    Zbontar, Jing, Misra, LeCun, Deny. "Barlow Twins: Self-Supervised
    Learning via Redundancy Reduction." ICML 2021.
    https://arxiv.org/abs/2103.03230
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm
from stable_pretraining.losses import BarlowTwinsLoss


@dataclass
class BarlowTwinsOutput(ModelOutput):
    """Output from Barlow Twins forward pass."""

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None


def _build_barlow_projector(in_dim: int, hidden_dims: Sequence[int]) -> nn.Module:
    """Linear(no-bias) -> BN -> ReLU stacked layers ending in a bias-free Linear."""
    if len(hidden_dims) < 1:
        raise ValueError("hidden_dims must contain at least one entry")
    layers = []
    prev = in_dim
    for i, dim in enumerate(hidden_dims):
        is_last = i == len(hidden_dims) - 1
        layers.append(nn.Linear(prev, dim, bias=False))
        if not is_last:
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
        prev = dim
    return nn.Sequential(*layers)


class BarlowTwins(Module):
    """Barlow Twins self-supervised learning.

    :param encoder_name: timm model name or pre-built ``nn.Module``.
    :param projector_dims: Hidden + output dims (default ``(8192, 8192, 8192)``
        matches the ResNet50 recipe).
    :param lambd: Off-diagonal weight in the cross-correlation loss
        (default ``5.1e-3`` from the paper).
    :param low_resolution: Adapt first conv for low-res input.
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (8192, 8192, 8192),
        lambd: float = 5.1e-3,
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

        self.projector = _build_barlow_projector(embed_dim, list(projector_dims))
        self.barlow_loss = BarlowTwinsLoss(lambd=lambd)

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> BarlowTwinsOutput:
        if view2 is None:
            embedding = self.backbone(view1)
            return BarlowTwinsOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
                projection=None,
            )

        h1 = self.backbone(view1)
        h2 = self.backbone(view2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        loss = self.barlow_loss(z1, z2)
        return BarlowTwinsOutput(
            loss=loss,
            embedding=torch.cat([h1, h2], dim=0),
            projection=torch.cat([z1, z2], dim=0),
        )
