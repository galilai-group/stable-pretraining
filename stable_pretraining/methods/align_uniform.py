"""Alignment and Uniformity self-supervised learning.

Wang and Isola. "Understanding Contrastive Representation Learning through
Alignment and Uniformity on the Hypersphere." ICML 2020.
https://arxiv.org/abs/2005.10242
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm
from stable_pretraining.forward import align_uniform
from stable_pretraining.losses import AlignUniformLoss


@dataclass
class AlignUniformOutput(ModelOutput):
    """Output from Alignment and Uniformity forward pass."""

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None
    alignment: Optional[torch.Tensor] = None
    uniformity: Optional[torch.Tensor] = None


def _build_projector(in_dim: int, hidden_dims: Sequence[int]) -> nn.Module:
    """Build a SimCLR-style MLP projector ending in batch normalization.

    Args:
        in_dim: Dimension of backbone features.
        hidden_dims: Hidden and output dimensions of the projector.

    Returns:
        Projector module.
    """
    if len(hidden_dims) < 1:
        raise ValueError("projector_dims must contain at least one entry")
    layers = []
    prev = in_dim
    for i, dim in enumerate(hidden_dims):
        is_last = i == len(hidden_dims) - 1
        layers.append(nn.Linear(prev, dim, bias=False))
        layers.append(nn.BatchNorm1d(dim))
        if not is_last:
            layers.append(nn.ReLU(inplace=True))
        prev = dim
    return nn.Sequential(*layers)


class AlignUniform(Module):
    """Alignment and Uniformity self-supervised learning.

    Args:
        encoder_name: timm model name or a pre-built module returning ``[B, D]``.
        projector_dims: Hidden and output dimensions for the projector.
        alignment_weight: Weight applied to the positive-pair alignment term.
        uniformity_weight: Weight applied to the embedding uniformity term.
        alpha: Exponent applied to positive-pair distances.
        t: Temperature for the Gaussian potential in the uniformity term.
        low_resolution: Adapt the encoder stem for small images.
        pretrained: Load pretrained timm weights when ``encoder_name`` is a name.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 2048, 256),
        alignment_weight: float = 1.0,
        uniformity_weight: float = 1.0,
        alpha: float = 2.0,
        t: float = 2.0,
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

        self.projector = _build_projector(embed_dim, projector_dims)
        self.align_uniform_loss = AlignUniformLoss(
            alignment_weight=alignment_weight,
            uniformity_weight=uniformity_weight,
            alpha=alpha,
            t=t,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
    ) -> dict[str, torch.Tensor]:
        """Delegate the batch dict and stage to the composable forward function.

        Args:
            batch: Two-view batch or single-view evaluation batch.
            stage: Current training stage.

        Returns:
            Dictionary containing embeddings, loss, and optional objective terms.
        """
        return align_uniform(self, batch, stage)
