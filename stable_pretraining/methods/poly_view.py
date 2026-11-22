"""Poly-View Contrastive Learning.

Generalizes two-view contrastive learning to M > 2 augmented views per sample,
contrasting all views of the same instance jointly.

References:
    Shidani, Hjelm, Ramapuram, Webb, Dhekane, Busbridge. "Poly-View
    Contrastive Learning." ICLR 2024.
    https://arxiv.org/abs/2403.05490

Example::

    from stable_pretraining.methods import PolyView

    model = PolyView(encoder_name="vit_small_patch16_224", loss="geometric")
    views = [torch.randn(64, 3, 224, 224) for _ in range(4)]
    out = model(*views)
    out.loss.backward()
"""

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining.losses import PVCLoss, SufficientStatisticsLoss
from stable_pretraining.methods.simclr import SimCLR


@dataclass
class PolyViewOutput(ModelOutput):
    """Output from Poly-View forward pass.

    Attributes:
        loss: Poly-view contrastive loss (0 in eval mode).
        embedding: Backbone features, ``[B, D]`` in eval mode and ``[M * B, D]``
            (view-major) in train mode.
        projection: Projector outputs ``[M * B, P]`` (``None`` in eval mode).
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None


class PolyView(SimCLR):
    """Poly-View contrastive learning with M views per sample.

    Reuses SimCLR's backbone and projector and swaps NT-Xent for a poly-view
    loss. With two views every loss reduces to SimCLR's NT-Xent.

    Args:
        encoder_name: timm model name or a pre-instantiated ``nn.Module``
            returning a ``[B, D]`` tensor.
        projector_dims: Hidden + output dimensions of the MLP projector.
        temperature: Contrastive temperature.
        loss: ``"geometric"`` or ``"arithmetic"`` for :class:`PVCLoss` with that
            aggregation, or ``"sufficient_statistics"`` for
            :class:`SufficientStatisticsLoss`.
        low_resolution: Adapt the first conv for 32x32 inputs (CIFAR-style).
        pretrained: Load pretrained timm weights for the encoder.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 2048, 256),
        temperature: float = 0.1,
        loss: Literal["geometric", "arithmetic", "sufficient_statistics"] = "geometric",
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        super().__init__(
            encoder_name=encoder_name,
            projector_dims=projector_dims,
            temperature=temperature,
            low_resolution=low_resolution,
            pretrained=pretrained,
        )
        del self.simclr_loss
        if loss == "sufficient_statistics":
            self.poly_view_loss = SufficientStatisticsLoss(temperature=temperature)
        else:
            self.poly_view_loss = PVCLoss(temperature=temperature, aggregation=loss)

    def forward(self, *views: torch.Tensor) -> PolyViewOutput:
        """Encode M views and compute the poly-view contrastive loss.

        Args:
            *views: ``M`` augmented views, each ``[B, C, H, W]``. A single view
                returns only the backbone embedding (eval mode).

        Returns:
            PolyViewOutput: Loss, embeddings and projections.
        """
        if len(views) == 1:
            embedding = self.backbone(views[0])
            return PolyViewOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
                projection=None,
            )

        hs = [self.backbone(v) for v in views]
        zs = [self.projector(h) for h in hs]
        return PolyViewOutput(
            loss=self.poly_view_loss(zs),
            embedding=torch.cat(hs, dim=0),
            projection=torch.cat(zs, dim=0),
        )
