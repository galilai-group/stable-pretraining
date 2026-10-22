"""Spectral Contrastive Learning.

Contrastive self-supervised learning with the spectral contrastive loss: positive
pairs are pulled together while squared inner products of independent pairs are
pushed to zero. It needs no temperature, momentum encoder, or stop-gradient.

References:
    HaoChen et al., "Provable Guarantees for Self-Supervised Deep Learning with
    Spectral Contrastive Loss", NeurIPS 2021. https://arxiv.org/abs/2106.04156

"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import from_timm
from stable_pretraining.forward import spectral
from stable_pretraining.losses import SpectralContrastiveLoss


@dataclass
class SpectralCLOutput(ModelOutput):
    """Output from the tensor interface of SpectralCL.

    Attributes:
        loss: Spectral contrastive loss for two views, or zero for one view.
        embedding: Backbone features of shape ``[2B, D]`` for two views, or
            ``[B, D]`` for one view.
        projection: Projector outputs of shape ``[2B, P]`` for two views, or
            ``None`` for one view.
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None


def _build_spectral_projector(in_dim: int, hidden_dims: Sequence[int]) -> nn.Module:
    """MLP projector: Linear -> BN -> ReLU -> ... -> Linear -> BN.

    Args:
        in_dim: Backbone output dimension.
        hidden_dims: Hidden and output dimensions, e.g. ``(2048, 1000)``.

    Returns:
        Sequential projector with BatchNorm after every linear layer.

    Raises:
        ValueError: If ``hidden_dims`` is empty.
    """
    if len(hidden_dims) < 1:
        raise ValueError("hidden_dims must contain at least one entry (the output dim)")
    layers = []
    prev = in_dim
    for i, dim in enumerate(hidden_dims):
        layers.append(nn.Linear(prev, dim, bias=False))
        layers.append(nn.BatchNorm1d(dim))
        if i < len(hidden_dims) - 1:
            layers.append(nn.ReLU(inplace=True))
        prev = dim
    return nn.Sequential(*layers)


class SpectralCL(Module):
    """Spectral Contrastive Learning.

    Architecture:
        - **Backbone**: any feature extractor producing a flat [B, D] embedding
          (timm ViT/ResNet with the head removed)
        - **Projector**: MLP with BatchNorm after every layer and ReLU after all
          but the last, as in the paper
        - **Loss**: spectral contrastive loss on projections confined to the
          ball of radius ``sqrt(mu)``

    Args:
        encoder_name: timm model name (e.g. ``"resnet18"`` or
            ``"vit_small_patch16_224"``), or an ``nn.Module`` whose ``forward``
            returns a ``[B, D]`` tensor.
        projector_dims: Hidden and output dimensions of the MLP projector. The
            default ``(2048, 1000)`` matches the authors' CIFAR-10 code; the
            paper text describes 1000 hidden units.
        mu: Squared radius of the ball the projections are confined to. The
            paper tunes it over ``{1, 3, 10}`` on CIFAR and uses 10 on ImageNet.
        low_resolution: Adapt the first convolution for 32x32 inputs.
        pretrained: Load pretrained timm weights for the encoder.
        optim: Optional optimizer configuration in the format of
            :class:`~stable_pretraining.Module`. When omitted, Module uses its
            default AdamW optimizer and cosine scheduler.

    The model supports two calling conventions:

    - **Tensors**: ``model(view1, view2)`` returns a :class:`SpectralCLOutput`.
    - **Batch dicts**: ``model(batch, stage=...)`` is the contract used by
      :class:`~stable_pretraining.Module`, so the model trains with
      :class:`~stable_pretraining.Manager` on batches from
      ``MultiViewTransform``. It returns a dict with ``"embedding"``,
      ``"label"`` when the views carry labels, and ``"loss"`` in training mode.

    Training with a supplied ``data`` DataModule whose training batches contain
    two view dicts under ``"views"`` (each with an ``"image"`` tensor)::

        import lightning as pl
        import stable_pretraining as spt

        model = spt.SpectralCL(encoder_name="resnet18", low_resolution=True)
        trainer = pl.Trainer(max_epochs=200)
        spt.Manager(trainer=trainer, module=model, data=data)()

    Using the tensor interface::

        import torch
        import stable_pretraining as spt

        model = spt.SpectralCL(
            encoder_name="resnet18",
            projector_dims=(2048, 1000),
            mu=1.0,
            low_resolution=True,
        )

        v1 = torch.randn(4, 3, 32, 32)
        v2 = torch.randn(4, 3, 32, 32)
        out = model(v1, v2)
        out.loss.backward()

        model.eval()
        with torch.no_grad():
            features = model(v1).embedding
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 1000),
        mu: float = 1.0,
        low_resolution: bool = False,
        pretrained: bool = False,
        optim: Optional[dict] = None,
    ) -> None:
        super().__init__(**({} if optim is None else {"optim": optim}))

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
            dummy = torch.zeros(1, 3, 224, 224)
            embed_dim = self.backbone(dummy).shape[-1]
        self.embed_dim = embed_dim

        self.projector = _build_spectral_projector(embed_dim, list(projector_dims))
        self.spectral_loss = SpectralContrastiveLoss(mu=mu)

    def forward(
        self,
        view1: Union[torch.Tensor, dict],
        view2: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
    ) -> Union[SpectralCLOutput, dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            view1: First augmented view of shape ``[B, C, H, W]``, or a batch
                dict, in which case ``view2`` must be ``None``.
            view2: Second augmented view of shape ``[B, C, H, W]``. If omitted
                for tensor input, returns backbone features, a zero loss, and
                no projection, regardless of training mode.
            stage: Lightning stage, used to name the logged loss for batch dicts.

        Returns:
            :class:`SpectralCLOutput` for tensors, or a dict with embeddings,
            labels when supplied, and a loss during training for batch dicts.
        """
        if isinstance(view1, dict):
            return self._forward_batch(view1, stage)

        if view2 is None:
            embedding = self.backbone(view1)
            return SpectralCLOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding,
                projection=None,
            )

        h1 = self.backbone(view1)
        h2 = self.backbone(view2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        loss = self.spectral_loss(z1, z2)
        return SpectralCLOutput(
            loss=loss,
            embedding=torch.cat([h1, h2], dim=0),
            projection=torch.cat([z1, z2], dim=0),
        )

    def _forward_batch(self, batch: dict, stage: Optional[str]) -> dict:
        """Forward a batch dict, as :class:`~stable_pretraining.Module` calls it.

        Args:
            batch: Two views during training, or a single ``{"image": ...}``
                batch or multiple views during evaluation.
            stage: Lightning stage for loss logging; defaults to ``"fit"``.

        Returns:
            Embeddings, optional labels, and a loss only during training.
        """
        return spectral(self, batch, stage or "fit")
