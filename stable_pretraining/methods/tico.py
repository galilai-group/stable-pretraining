"""TiCO: Transformation Invariance and Covariance Contrast.

Joint-embedding SSL with two terms:
- **Invariance**: cosine similarity between paired views (maximised).
- **Covariance contrast**: penalise the off-diagonal of an EMA-tracked
  covariance of the projected features (decorrelation).

The EMA covariance acts as a soft memory bank; it stabilises the second
term against batch-size variation.

References:
    Zhu et al. "TiCo: Transformation Invariance and Covariance Contrast
    for Self-Supervised Visual Representation Learning." arXiv 2022.
    https://arxiv.org/abs/2206.10698
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm


@dataclass
class TiCOOutput(ModelOutput):
    """Structured output of the :class:`TiCO` SSL method."""

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None


def _projector(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class TiCO(Module):
    """TiCO joint-embedding SSL.

    :param encoder_name: timm model or pre-built ``nn.Module``.
    :param projector_dims: ``(hidden, output)`` (default ``(2048, 256)``).
    :param beta: EMA momentum for the covariance (default ``0.9``; paper).
    :param rho: Weight on the covariance-contrast term (default ``20.0``;
        paper used a sweep around 16-20).
    :param low_resolution: Adapt first conv for low-res input.
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 256),
        beta: float = 0.9,
        rho: float = 20.0,
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
        self.beta = beta
        self.rho = rho

        proj_hidden, proj_out = projector_dims
        self.projector = _projector(embed_dim, proj_hidden, proj_out)
        # EMA covariance buffer (D x D)
        self.register_buffer("C", torch.zeros(proj_out, proj_out))

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> TiCOOutput:
        if view2 is None:
            embedding = self.backbone(view1)
            return TiCOOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
            )

        h1 = self.backbone(view1)
        h2 = self.backbone(view2)
        # Always work on unit-norm projections so the running covariance is
        # bounded — without this, ``z^T C z`` can explode during training.
        z1n = F.normalize(self.projector(h1), dim=-1)
        z2n = F.normalize(self.projector(h2), dim=-1)

        # Update EMA covariance from both unit-norm views.
        with torch.no_grad():
            z_cat = torch.cat([z1n, z2n], dim=0).float()
            z_cat = z_cat - z_cat.mean(dim=0, keepdim=True)
            B = z_cat.shape[0]
            C_batch = (z_cat.T @ z_cat) / max(B - 1, 1)
            self.C.mul_(self.beta).add_(C_batch, alpha=1 - self.beta)

        # Invariance: 2 - 2·cos(z1, z2) ≥ 0
        inv = 2 - 2 * (z1n * z2n).sum(dim=-1).mean()

        # Covariance contrast — penalise alignment of student projections
        # with the principal directions of the running covariance.
        C = self.C.detach().to(z1n.dtype)
        cov_term = (z1n @ C * z1n).sum(dim=-1).mean()

        loss = inv + self.rho * cov_term
        return TiCOOutput(
            loss=loss,
            embedding=torch.cat([h1, h2], dim=0),
            projection=torch.cat([z1n, z2n], dim=0),
        )
