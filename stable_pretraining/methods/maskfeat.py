"""MaskFeat: Masked feature prediction with HOG targets.

Replaces MAE's pixel reconstruction with the prediction of *Histograms of
Oriented Gradients* (HOG) at masked patch positions. HOG is a hand-crafted
feature with a strong inductive bias (scale + photometric invariance) that
keeps the encoder from learning low-level texture only.

References:
    Wei et al. "Masked Feature Prediction for Self-Supervised Visual
    Pre-Training." CVPR 2022. https://arxiv.org/abs/2112.09133
"""

from dataclasses import dataclass
from math import pi
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module


@dataclass
class MaskFeatOutput(ModelOutput):
    """Structured output of the :class:`MaskFeat` SSL method."""

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    predictions: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None


def _hog_per_patch(
    images: torch.Tensor, patch_size: int, n_bins: int = 9
) -> torch.Tensor:
    """Compute a per-patch HOG descriptor of shape ``[B, N, C * n_bins]``.

    Simple per-channel HOG using 1D Sobel-style gradient kernels.
    """
    B, C, H, W = images.shape
    p = patch_size
    assert H % p == 0 and W % p == 0
    # Channel-grouped Sobel-like gradients
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0]], device=images.device, dtype=images.dtype
    ).view(1, 1, 1, 3)
    ky = torch.tensor(
        [[-1.0], [0.0], [1.0]], device=images.device, dtype=images.dtype
    ).view(1, 1, 3, 1)
    kx = kx.expand(C, 1, 1, 3)
    ky = ky.expand(C, 1, 3, 1)
    gx = F.conv2d(images, kx, padding=(0, 1), groups=C)
    gy = F.conv2d(images, ky, padding=(1, 0), groups=C)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    ang = torch.atan2(gy, gx) % pi  # in [0, pi)
    bin_idx = torch.clamp((ang * n_bins / pi).long(), max=n_bins - 1)  # [B, C, H, W]

    # Per-patch histograms via scatter-add. Reshape to [B, C, gh, p, gw, p].
    gh, gw = H // p, W // p
    mag_p = mag.view(B, C, gh, p, gw, p)
    bin_p = bin_idx.view(B, C, gh, p, gw, p)
    # Permute so the patch interior is last: [B, C, gh, gw, p, p]
    mag_p = mag_p.permute(0, 1, 2, 4, 3, 5).contiguous()
    bin_p = bin_p.permute(0, 1, 2, 4, 3, 5).contiguous()
    mag_flat = mag_p.view(B, C, gh * gw, p * p)
    bin_flat = bin_p.view(B, C, gh * gw, p * p)
    hist = torch.zeros(
        B, C, gh * gw, n_bins, device=images.device, dtype=mag_flat.dtype
    )
    hist.scatter_add_(-1, bin_flat, mag_flat)
    # Standardise per-patch (zero mean, unit variance over the bin dim) — the
    # MaskFeat paper does this so the prediction target is well-conditioned
    # without being L2-flattened.
    mean = hist.mean(dim=-1, keepdim=True)
    var = hist.var(dim=-1, keepdim=True)
    hist = (hist - mean) / (var + 1e-6).sqrt()
    return hist.permute(0, 2, 1, 3).reshape(B, gh * gw, C * n_bins)


class MaskFeat(Module):
    """MaskFeat: predict per-patch HOG at masked positions.

    :param encoder_name: timm ViT name (default ``"vit_small_patch16_224"``).
    :param patch_size: Patch size (default 16, must match encoder).
    :param mask_ratio: Fraction of patches masked (default 0.4).
    :param n_hog_bins: HOG orientation bins (default 9).
    :param image_size: Input size (default 224).
    :param in_channels: Image channels (default 3).
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        patch_size: int = 16,
        mask_ratio: float = 0.4,
        n_hog_bins: int = 9,
        image_size: int = 224,
        in_channels: int = 3,
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
            seq = self.encoder.forward_features(
                torch.zeros(1, in_channels, image_size, image_size)
            )
        self._has_cls = (
            hasattr(self.encoder, "cls_token")
            and self.encoder.cls_token is not None
            and seq.shape[1] > 1
        )
        embed_dim = seq.shape[-1]
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.n_hog_bins = n_hog_bins
        self.image_size = image_size
        self.in_channels = in_channels

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.head = nn.Linear(embed_dim, in_channels * n_hog_bins)

    def _random_mask(self, B, N, device):
        n_mask = int(round(N * self.mask_ratio))
        noise = torch.rand(B, N, device=device)
        order = noise.argsort(dim=1)
        mask = torch.zeros(B, N, device=device)
        mask.scatter_(1, order[:, :n_mask], 1.0)
        return mask

    def _encode(self, images, mask):
        vit = self.encoder
        x = vit.patch_embed(images)
        m = mask.unsqueeze(-1)
        x = x * (1 - m) + self.mask_token.expand_as(x) * m
        if self._has_cls:
            cls = vit.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = x + vit.pos_embed
        x = vit.pos_drop(x)
        x = vit.blocks(x)
        x = vit.norm(x)
        if self._has_cls:
            x = x[:, 1:]
        return x

    def forward(self, images: torch.Tensor) -> MaskFeatOutput:
        B = images.shape[0]
        if not self.training:
            features = self.encoder.forward_features(images)
            cls = features[:, 0] if self._has_cls else features.mean(dim=1)
            return MaskFeatOutput(
                loss=torch.zeros((), device=images.device, dtype=images.dtype),
                embedding=cls,
            )

        with torch.no_grad():
            target = _hog_per_patch(
                images, patch_size=self.patch_size, n_bins=self.n_hog_bins
            )
        N = target.shape[1]
        mask = self._random_mask(B, N, device=images.device)

        encoded = self._encode(images, mask)
        pred = self.head(encoded)

        flat_mask = mask.bool().view(-1)
        flat_pred = pred.reshape(-1, pred.shape[-1])[flat_mask]
        flat_target = target.reshape(-1, target.shape[-1])[flat_mask]
        loss = F.mse_loss(flat_pred, flat_target)

        # Mean of all patch tokens (unmasked + masked) — using only unmasked
        # tokens biases the probe toward unmasked content, which damages it.
        embedding = encoded.mean(dim=1)

        return MaskFeatOutput(
            loss=loss,
            embedding=embedding.detach(),
            predictions=pred,
            targets=target,
            mask=mask,
        )
