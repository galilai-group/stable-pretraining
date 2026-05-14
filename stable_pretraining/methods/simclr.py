"""SimCLR: Simple Contrastive Learning of Representations.

Self-supervised learning via maximizing agreement between two augmented views
of the same image using NT-Xent contrastive loss.

References:
    Chen et al. "A Simple Framework for Contrastive Learning of Visual
    Representations." ICML 2020. https://arxiv.org/abs/2002.05709

Example::

    from stable_pretraining.methods import SimCLR
    import lightning as pl

    model = SimCLR(encoder_name="vit_small_patch16_224")

    trainer = pl.Trainer(max_epochs=300)
    trainer.fit(model, dataloader)
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import BatchNorm1dNoBias, from_timm
from stable_pretraining.losses import NTXEntLoss


@dataclass
class SimCLROutput(ModelOutput):
    """Output from SimCLR forward pass.

    :ivar loss: NT-Xent contrastive loss (0 in eval mode)
    :ivar embedding: Backbone features [B, D] in eval mode, [2B, D] in train mode
    :ivar projection: Projector outputs [2B, P] (None in eval mode)
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None


def _build_projector(
    in_dim: int,
    hidden_dims: Sequence[int],
    final_bn_no_bias: bool = True,
) -> nn.Module:
    """Standard SimCLR projector: Linear -> BN -> ReLU -> ... -> Linear -> BN(no bias).

    :param in_dim: Backbone output dimension
    :param hidden_dims: Sequence of hidden + output dimensions, e.g. (2048, 2048, 128)
    :param final_bn_no_bias: Use ``BatchNorm1dNoBias`` on the final layer (SimCLR original)
    """
    if len(hidden_dims) < 1:
        raise ValueError("hidden_dims must contain at least one entry (the output dim)")
    layers = []
    prev = in_dim
    for i, dim in enumerate(hidden_dims):
        is_last = i == len(hidden_dims) - 1
        layers.append(nn.Linear(prev, dim, bias=False))
        if is_last:
            layers.append(BatchNorm1dNoBias(dim) if final_bn_no_bias else nn.BatchNorm1d(dim))
        else:
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
        prev = dim
    return nn.Sequential(*layers)


class SimCLR(Module):
    """SimCLR: contrastive joint-embedding self-supervised learning.

    Architecture:
        - **Backbone**: any feature extractor producing a flat [B, D] embedding
          (timm ViT/ResNet with the head removed)
        - **Projector**: 2- or 3-layer MLP mapping features to the contrastive space
        - **Loss**: NT-Xent (normalised temperature-scaled cross entropy)

    :param encoder_name: timm model name (e.g. ``"vit_small_patch16_224"``,
        ``"resnet50"``) or a pre-instantiated ``nn.Module`` whose ``forward``
        returns a ``[B, D]`` tensor.
    :param projector_dims: Hidden + output dimensions of the MLP projector.
        ``(2048, 2048, 128)`` matches the original SimCLR ResNet50 recipe; for
        ViT backbones the input is taken from the encoder embed_dim.
    :param temperature: Temperature for NT-Xent (0.5 in original SimCLR; 0.1
        is common for harder/larger batches).
    :param low_resolution: Adapt first conv for 32x32 inputs (CIFAR-style).
    :param pretrained: Load pretrained timm weights for the encoder.

    Example::

        model = SimCLR(
            encoder_name="vit_small_patch16_224",
            projector_dims=(2048, 2048, 256),
            temperature=0.2,
        )

        v1 = torch.randn(64, 3, 224, 224)
        v2 = torch.randn(64, 3, 224, 224)
        out = model(v1, v2)
        out.loss.backward()

        # eval: single view, no loss
        model.eval()
        out = model(v1)
        features = out.embedding  # [64, embed_dim]
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 2048, 256),
        temperature: float = 0.5,
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

        # Detect embedding dimension by running a tiny dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            embed_dim = self.backbone(dummy).shape[-1]
        self.embed_dim = embed_dim

        self.projector = _build_projector(embed_dim, list(projector_dims))
        self.simclr_loss = NTXEntLoss(temperature=temperature)

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> SimCLROutput:
        """Forward pass.

        :param view1: First augmented view [B, C, H, W] (or single view at eval).
        :param view2: Second augmented view [B, C, H, W]. If ``None``, returns
            only the backbone embedding (eval mode).
        :return: :class:`SimCLROutput`.
        """
        if view2 is None:
            embedding = self.backbone(view1)
            return SimCLROutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
                projection=None,
            )

        h1 = self.backbone(view1)
        h2 = self.backbone(view2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        loss = self.simclr_loss(z1, z2)
        return SimCLROutput(
            loss=loss,
            embedding=torch.cat([h1, h2], dim=0),
            projection=torch.cat([z1, z2], dim=0),
        )
