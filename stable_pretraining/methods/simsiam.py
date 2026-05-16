"""SimSiam: Simple Siamese self-supervised learning.

Two-view siamese network with a stop-gradient on the target branch — no
momentum encoder, no negatives, no large batches. The student has a
backbone, projector, and predictor; the loss is the negative cosine
similarity between the predictor output and a stop-gradient projection
of the other view.

References:
    Chen, He. "Exploring Simple Siamese Representation Learning." CVPR 2021.
    https://arxiv.org/abs/2011.10566
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm


@dataclass
class SimSiamOutput(ModelOutput):
    """Structured output of the :class:`SimSiam` SSL method."""

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    prediction: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None


def _projector(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    """SimSiam projector: 3-layer MLP with BN, no final ReLU."""
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
    """SimSiam predictor: 2-layer MLP with bottleneck."""
    return nn.Sequential(
        nn.Linear(dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, dim),
    )


def _neg_cos_sim(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Symmetric negative cosine similarity loss with stop-gradient on z."""
    z = z.detach()
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return -(p * z).sum(dim=-1).mean()


class SimSiam(Module):
    """SimSiam: simple siamese SSL with stop-gradient.

    :param encoder_name: timm model name or pre-built ``nn.Module``.
    :param projector_dim: Projector hidden + output dim (default 2048).
    :param predictor_hidden_dim: Predictor bottleneck dim (default 512).
    :param low_resolution: Adapt first conv for 32x32.
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dim: int = 2048,
        predictor_hidden_dim: int = 512,
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

        self.projector = _projector(embed_dim, projector_dim, projector_dim)
        self.predictor = _predictor(projector_dim, predictor_hidden_dim)

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> SimSiamOutput:
        if view2 is None:
            embedding = self.backbone(view1)
            return SimSiamOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
                prediction=None,
                target=None,
            )

        h1 = self.backbone(view1)
        h2 = self.backbone(view2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        loss = (_neg_cos_sim(p1, z2) + _neg_cos_sim(p2, z1)) / 2
        return SimSiamOutput(
            loss=loss,
            embedding=torch.cat([h1, h2], dim=0),
            prediction=torch.cat([p1, p2], dim=0),
            target=torch.cat([z1, z2], dim=0).detach(),
        )
