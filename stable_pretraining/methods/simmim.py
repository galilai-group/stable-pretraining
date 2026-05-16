"""SimMIM: Simple Framework for Masked Image Modeling.

Predicts raw pixel values for masked patches via a 1-layer linear decoder.
Unlike MAE, SimMIM passes both visible and mask tokens through the encoder
and uses a trivial decoder, making it cleaner to integrate with any ViT.

References:
    Xie et al. "SimMIM: A Simple Framework for Masked Image Modeling."
    CVPR 2022. https://arxiv.org/abs/2111.09886
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import patchify


@dataclass
class SimMIMOutput(ModelOutput):
    """Structured output of the :class:`SimMIM` SSL method."""

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    predictions: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None


class SimMIM(Module):
    """SimMIM masked image modeling.

    :param encoder_name: timm model name (default ``"vit_small_patch16_224"``).
    :param patch_size: Patch size (must match the encoder's).
    :param mask_ratio: Fraction of patches to mask (default 0.6, paper used 0.6).
    :param in_channels: Image channels (default 3).
    :param image_size: Input image size (default 224).
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        patch_size: int = 16,
        mask_ratio: float = 0.6,
        in_channels: int = 3,
        image_size: int = 224,
        pretrained: bool = False,
    ):
        super().__init__()

        if isinstance(encoder_name, str):
            import timm

            self.encoder = timm.create_model(
                encoder_name, num_classes=0, pretrained=pretrained
            )
        else:
            self.encoder = encoder_name

        with torch.no_grad():
            embed_dim = self.encoder(
                torch.zeros(1, in_channels, image_size, image_size)
            ).shape[-1]
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.in_channels = in_channels
        self.image_size = image_size

        # Learnable mask token (same dim as patch embedding input space)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # 1-layer linear decoder: embed_dim -> per-patch pixels
        self.decoder = nn.Linear(embed_dim, patch_size * patch_size * in_channels)

    def _random_mask(self, B: int, N: int, device) -> torch.Tensor:
        """Sample a 0/1 mask per (batch, patch) with given mask_ratio."""
        n_mask = int(round(N * self.mask_ratio))
        noise = torch.rand(B, N, device=device)
        order = noise.argsort(dim=1)
        mask = torch.zeros(B, N, device=device)
        mask.scatter_(1, order[:, :n_mask], 1.0)
        return mask  # 1 = masked

    def _encode_with_mask(
        self, images: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Patch-embed, replace masked tokens, then run encoder blocks.

        Works with timm ViTs that expose ``patch_embed``, ``cls_token`` (optional),
        ``pos_embed``, ``blocks``, ``norm``.
        """
        vit = self.encoder
        B = images.shape[0]
        x = vit.patch_embed(images)  # [B, N, D]
        # Replace masked positions with mask_token
        m = mask.unsqueeze(-1)  # [B, N, 1]
        x = x * (1 - m) + self.mask_token.expand_as(x) * m
        # Add CLS token if the model uses one
        if hasattr(vit, "cls_token") and vit.cls_token is not None:
            cls = vit.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
        # Positional embedding
        x = x + vit.pos_embed
        x = vit.pos_drop(x)
        x = vit.blocks(x)
        x = vit.norm(x)
        # Drop CLS for patch reconstruction
        if hasattr(vit, "cls_token") and vit.cls_token is not None:
            x = x[:, 1:]
        return x

    def forward(self, images: torch.Tensor) -> SimMIMOutput:
        """Forward pass.

        :param images: ``[B, C, H, W]`` images.
        :return: :class:`SimMIMOutput`.
        """
        B = images.shape[0]
        H = W = self.image_size
        p = self.patch_size
        N = (H // p) * (W // p)

        if not self.training:
            # Plain encoding for downstream tasks
            features = self.encoder.forward_features(images)
            cls = features[:, 0] if features.ndim == 3 else features
            return SimMIMOutput(
                loss=torch.zeros((), device=images.device, dtype=images.dtype),
                embedding=cls,
            )

        mask = self._random_mask(B, N, device=images.device)
        encoded = self._encode_with_mask(images, mask)  # [B, N, D]
        pred_pixels = self.decoder(encoded)  # [B, N, P]

        target = patchify(images, patch_size=(self.in_channels, p, p))  # [B, N, C*p*p]

        loss_per = F.l1_loss(pred_pixels, target, reduction="none").mean(
            dim=-1
        )  # [B, N]
        loss = (loss_per * mask).sum() / mask.sum().clamp(min=1.0)

        # Embedding for probes: mean of *all* patch tokens.
        embedding = encoded.mean(dim=1)

        return SimMIMOutput(
            loss=loss,
            embedding=embedding.detach(),
            predictions=pred_pixels,
            mask=mask,
        )
