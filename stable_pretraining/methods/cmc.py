"""CMC: Contrastive Multiview Coding.

Self-supervised learning by contrasting different views of the same image,
where the two views are the L (luminance) and ab (chrominance) channels of
CIE Lab color space rather than two independently augmented crops.

References:
    Tian et al. "Contrastive Multiview Coding." ECCV 2020.
    https://arxiv.org/abs/1906.05849

Example::

    from stable_pretraining.methods import CMC
    import lightning as pl

    model = CMC(encoder_name="resnet18", low_resolution=True)

    trainer = pl.Trainer(max_epochs=300)
    trainer.fit(model, dataloader)
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import from_torchvision
from stable_pretraining.data.transforms import _rgb_to_lab
from stable_pretraining.losses import CLIPLoss


@dataclass
class CMCOutput(ModelOutput):
    """Output from CMC forward pass.

    :ivar loss: Bidirectional InfoNCE loss between L and ab projections
        (0 in eval mode)
    :ivar embedding: Concatenation of L and ab backbone features [B, 2*D]
    :ivar projection_l: L-view projector output [B, P] (None in eval mode)
    :ivar projection_ab: ab-view projector output [B, P] (None in eval mode)
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection_l: Optional[torch.Tensor] = None
    projection_ab: Optional[torch.Tensor] = None


def _build_projector(in_dim: int, hidden_dims: Sequence[int]) -> nn.Module:
    """Simple MLP projector: Linear -> BN -> ReLU -> ... -> Linear.

    :param in_dim: Backbone output dimension
    :param hidden_dims: Sequence of hidden + output dimensions, e.g. (128,)
    """
    if len(hidden_dims) < 1:
        raise ValueError("hidden_dims must contain at least one entry (the output dim)")
    layers = []
    prev = in_dim
    for i, dim in enumerate(hidden_dims):
        layers.append(nn.Linear(prev, dim))
        if i < len(hidden_dims) - 1:
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
        prev = dim
    return nn.Sequential(*layers)


class CMC(Module):
    """CMC: contrastive learning across CIE Lab color-space channels.

    Architecture:
        - **backbone_l**: feature extractor for the 1-channel L (luminance) view
        - **backbone_ab**: feature extractor for the 2-channel ab (chrominance) view
        - **projector_l** / **projector_ab**: MLPs mapping each backbone's
          features to a shared contrastive space
        - **Loss**: bidirectional InfoNCE (:class:`~stable_pretraining.losses.CLIPLoss`)
          between the two projections

    :param encoder_name: torchvision model name (e.g. ``"resnet18"``) used to
        build both ``backbone_l`` and ``backbone_ab`` with the appropriate
        ``in_channels``.
    :param projector_dims: Hidden + output dimensions of each MLP projector.
    :param temperature: Temperature for the bidirectional InfoNCE loss.
    :param low_resolution: Adapt first conv for 32x32 inputs (CIFAR-style).
    :param pretrained: Load pretrained torchvision weights for the encoders.

    Example::

        model = CMC(encoder_name="resnet18", low_resolution=True)

        l_view = torch.randn(64, 1, 32, 32)
        ab_view = torch.randn(64, 2, 32, 32)
        out = model(l_view, ab_view)
        out.loss.backward()

        # eval: single raw RGB view, no loss
        model.eval()
        rgb = torch.randn(64, 3, 32, 32)
        out = model(rgb)
        features = out.embedding  # [64, 2*embed_dim]
    """

    def __init__(
        self,
        encoder_name: str = "resnet18",
        projector_dims: Sequence[int] = (128,),
        temperature: float = 0.07,
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()

        weights = "DEFAULT" if pretrained else None
        self.backbone_l = from_torchvision(
            encoder_name,
            low_resolution=low_resolution,
            weights=weights,
            in_channels=1,
        )
        self.backbone_ab = from_torchvision(
            encoder_name,
            low_resolution=low_resolution,
            weights=weights,
            in_channels=2,
        )

        with torch.no_grad():
            size = 32 if low_resolution else 224
            embed_dim = self.backbone_l(torch.zeros(1, 1, size, size)).shape[-1]
        self.embed_dim = embed_dim

        self.projector_l = _build_projector(embed_dim, list(projector_dims))
        self.projector_ab = _build_projector(embed_dim, list(projector_dims))
        self.cmc_loss = CLIPLoss(temperature=temperature)

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> CMCOutput:
        """Forward pass.

        :param view1: L-channel view [B, 1, H, W] during training, or a raw
            RGB image [B, 3, H, W] at eval (``view2 is None``).
        :param view2: ab-channel view [B, 2, H, W]. If ``None``, ``view1`` is
            treated as a raw RGB image and internally split into L/ab.
        :return: :class:`CMCOutput`.
        """
        if view2 is None:
            lab = _rgb_to_lab(view1)
            l_input = lab[:, 0:1] / 50.0 - 1.0
            ab_input = lab[:, 1:3] / 110.0
            h_l = self.backbone_l(l_input)
            h_ab = self.backbone_ab(ab_input)
            embedding = torch.cat([h_l, h_ab], dim=-1)
            return CMCOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
            )

        h_l = self.backbone_l(view1)
        h_ab = self.backbone_ab(view2)
        z_l = self.projector_l(h_l)
        z_ab = self.projector_ab(h_ab)
        loss = self.cmc_loss(z_l, z_ab)
        return CMCOutput(
            loss=loss,
            embedding=torch.cat([h_l, h_ab], dim=-1),
            projection_l=z_l,
            projection_ab=z_ab,
        )
