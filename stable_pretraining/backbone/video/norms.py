"""Normalization layers shared across causal video encoders."""

from __future__ import annotations

import torch
import torch.nn as nn


class GroupNormPerFrame(nn.GroupNorm):
    """``GroupNorm`` whose statistics are computed independently per temporal frame.

    Standard ``nn.GroupNorm`` on a 5D input pools statistics across the
    ``(T, H, W)`` axes, which lets a perturbation at frame ``t = k+1`` shift
    the mean/variance used to normalize frame ``t = 0`` — i.e. it breaks
    temporal causality even when the surrounding convolutions are causal.

    Computing the statistics per frame fixes this and is the standard pattern
    used by every modern causal video VAE (Wan-VAE, Cosmos Tokenizer, the
    Stable Video Diffusion causal 3D VAE). On 4D inputs this layer behaves
    identically to ``nn.GroupNorm``.

    Example::

        norm = GroupNormPerFrame(num_groups=32, num_channels=128)
        x = torch.randn(2, 128, 16, 32, 32)
        y = norm(x)  # statistics computed per t ∈ [0, 15]
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            return super().forward(x)
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = super().forward(x)
        x = x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return x


def _fit_groups(target_groups: int, channels: int) -> int:
    """Return the largest divisor of ``channels`` that is ``<= target_groups``.

    ``nn.GroupNorm`` requires ``num_groups`` to divide ``num_channels`` exactly.
    Use this helper when channel counts come from a configurable width.
    """
    g = min(target_groups, channels)
    while channels % g != 0:
        g -= 1
    return g
