"""MAGVIT-v2 causal video encoder.

Reference
---------
Yu et al., "Language Model Beats Diffusion - Tokenizer is Key to Visual
Generation", ICLR 2024 (https://arxiv.org/abs/2310.05737). This module
implements the *encoder half* of MAGVIT-v2 (the causal 3D-ResNet that
maps a video clip to a spatially/temporally downsampled latent feature
map). The matching quantizers (FSQ / LFQ) and decoder live elsewhere
because plenty of downstream uses only need the encoder.

Design
------
- Pure causal 3D convolutions: every layer is a :class:`CausalConv3d`,
  so the receptive field at frame ``t`` is strictly ``[0, t]`` in time.
  This makes the encoder safe to drop into streaming / autoregressive
  pipelines and is verified by the ``no future leakage`` unit test.
- GroupNorm is applied **per frame** (statistics computed independently
  for each ``t``). Standard 5D ``nn.GroupNorm`` pools statistics across
  ``(T, H, W)``, which silently breaks causality even when the convs are
  causal. Every modern causal video VAE (Wan-VAE, Cosmos Tokenizer, the
  SVD causal-3D VAE) uses per-frame normalization for the same reason.
- Channels-first throughout. ``channels_last_3d`` interacts poorly with
  GroupNorm under ``torch.compile``, so we stay in NCDHW.
- Each stage is its own ``nn.Module`` so FSDP auto-wrap policies and
  ``accelerate.init_empty_weights()`` find natural sharding boundaries.
- Activation checkpointing is opt-in via ``use_checkpoint=True`` and is
  off by default (it slows down the small presets).

Scaling
-------
The factory functions (``magvit2_tiny`` ... ``magvit2_gigantic``) match
the ViT-family naming convention used elsewhere in this package. Width
is controlled by ``base_channels``, depth by ``n_res_blocks``.

Example::

    enc = magvit2_base()
    x = torch.randn(2, 3, 16, 256, 256)
    out = enc(x)
    out.feature_map.shape  # (2, 16, 4, 16, 16) — 4× temporal, 16× spatial
    out.pooled.shape  # (2, 16)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from transformers.utils import ModelOutput

from .causal_conv3d import CausalConv3d
from .norms import GroupNormPerFrame, _fit_groups


@dataclass
class MAGVIT2Output(ModelOutput):
    """Structured output of :class:`MAGVIT2Encoder`.

    :param feature_map: ``(B, latent_dim, T', H', W')`` — the encoder's
        spatiotemporal latent grid.
    :param pooled: ``(B, latent_dim)`` — global-average-pooled feature
        when ``global_pool != ''``, else ``None``.
    """

    feature_map: torch.Tensor = None
    pooled: Optional[torch.Tensor] = None


class _ResBlock3D(nn.Module):
    """Pre-norm causal 3D residual block: ``GN -> SiLU -> Conv -> GN -> SiLU -> Conv (+ skip)``.

    Optional 1×1×1 projection on the skip path when ``in_channels != out_channels``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 32,
    ):
        super().__init__()
        g_in = _fit_groups(groups, in_channels)
        g_out = _fit_groups(groups, out_channels)

        self.norm1 = GroupNormPerFrame(g_in, in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=kernel_size)
        self.norm2 = GroupNormPerFrame(g_out, out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=kernel_size)
        self.act = nn.SiLU(inplace=False)

        if in_channels != out_channels:
            self.skip = CausalConv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class _Downsample(nn.Module):
    """Strided causal 3D conv with stride ``(st, 2, 2)``.

    Spatial dims always halve; ``st`` is 1 (no temporal downsample) or 2
    (also halve time).
    """

    def __init__(self, channels: int, temporal: bool):
        super().__init__()
        st = 2 if temporal else 1
        self.conv = CausalConv3d(channels, channels, kernel_size=3, stride=(st, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Stage(nn.Module):
    """A MAGVIT-v2 encoder stage.

    ``n_res_blocks`` residual blocks followed by an optional downsample.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_res_blocks: int,
        downsample: bool,
        temporal_downsample: bool,
        groups: int = 32,
    ):
        super().__init__()
        blocks = []
        ch = in_channels
        for _ in range(n_res_blocks):
            blocks.append(_ResBlock3D(ch, out_channels, groups=groups))
            ch = out_channels
        self.blocks = nn.ModuleList(blocks)
        if downsample:
            self.downsample = _Downsample(out_channels, temporal=temporal_downsample)
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.downsample(x)


class MAGVIT2Encoder(nn.Module):
    """MAGVIT-v2 causal video encoder.

    The encoder is a 4-stage causal 3D ResNet. Spatial resolution is halved
    at every stage (16× total). Temporal resolution is halved at the second
    and third stages (4× total), matching the original paper's compression
    ratio.

    :param in_channels: Input channels (3 for RGB).
    :param base_channels: Channel width of the stem. Per-stage channel counts
        are ``base_channels * channel_multipliers[i]``.
    :param channel_multipliers: Per-stage channel multipliers, length 4.
    :param n_res_blocks: Residual blocks per stage (uniform across stages).
    :param latent_dim: Channels of the encoder's output feature map.
    :param temporal_downsample_stages: 0-indexed stage IDs where time is also
        halved. Default ``(1, 2)`` gives 4× temporal compression.
    :param groups: Target group count for ``GroupNorm``. Actual group count
        per layer is clamped to a divisor of the channel count.
    :param global_pool: Either ``'avg'`` (global-average-pool the feature map
        to ``(B, latent_dim)`` and return it in ``output.pooled``) or ``''``
        (return only the feature map).
    :param use_checkpoint: If True, wrap each stage in
        ``torch.utils.checkpoint``. Off by default — on small presets the
        recompute overhead is a net slowdown.

    Example::

        # 8 frames at 128×128, MAGVIT2-small.
        enc = magvit2_small()
        out = enc(torch.randn(1, 3, 8, 128, 128))
        out.feature_map.shape  # (1, 16, 2, 8, 8)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 2, 4),
        n_res_blocks: int = 2,
        latent_dim: int = 16,
        temporal_downsample_stages: Tuple[int, ...] = (1, 2),
        groups: int = 32,
        global_pool: str = "avg",
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if len(channel_multipliers) < 1:
            raise ValueError("channel_multipliers must be non-empty")
        if global_pool not in ("avg", ""):
            raise ValueError(f"global_pool must be 'avg' or '', got {global_pool!r}")

        self.global_pool = global_pool
        self.use_checkpoint = use_checkpoint
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        # Stem: lift RGB to base_channels (causal so the very first frame
        # remains a function of itself only).
        self.stem = CausalConv3d(in_channels, base_channels, kernel_size=3)

        stage_channels = [base_channels * m for m in channel_multipliers]
        self.stages = nn.ModuleList()
        prev_ch = base_channels
        n_stages = len(channel_multipliers)
        for i, ch in enumerate(stage_channels):
            is_last = i == n_stages - 1
            self.stages.append(
                _Stage(
                    in_channels=prev_ch,
                    out_channels=ch,
                    n_res_blocks=n_res_blocks,
                    downsample=not is_last,
                    temporal_downsample=(i in set(temporal_downsample_stages)),
                    groups=groups,
                )
            )
            prev_ch = ch

        out_ch = stage_channels[-1]
        g_out = _fit_groups(groups, out_ch)
        self.norm_out = GroupNormPerFrame(g_out, out_ch)
        self.act_out = nn.SiLU(inplace=False)
        self.proj_out = CausalConv3d(out_ch, latent_dim, kernel_size=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder and return the latent feature map (no pooling).

        :param x: ``(B, C, T, H, W)``.
        :return: ``(B, latent_dim, T', H', W')``.
        """
        x = self.stem(x)
        for stage in self.stages:
            if self.use_checkpoint and self.training:
                x = ckpt.checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        x = self.proj_out(self.act_out(self.norm_out(x)))
        return x

    def forward(self, x: torch.Tensor) -> MAGVIT2Output:
        feat = self.forward_features(x)
        pooled = feat.mean(dim=(2, 3, 4)) if self.global_pool == "avg" else None
        return MAGVIT2Output(feature_map=feat, pooled=pooled)


# -----------------------------------------------------------------------------
# Scaling presets — width via ``base_channels``, depth via ``n_res_blocks``.
# Naming mirrors the ViT family in this package (tiny → gigantic).
# -----------------------------------------------------------------------------


def magvit2_tiny(**kwargs) -> MAGVIT2Encoder:
    """MAGVIT-v2 Tiny. ``base_channels=32, n_res_blocks=2`` (~5M params)."""
    return MAGVIT2Encoder(base_channels=32, n_res_blocks=2, latent_dim=8, **kwargs)


def magvit2_small(**kwargs) -> MAGVIT2Encoder:
    """MAGVIT-v2 Small. ``base_channels=64, n_res_blocks=2`` (~20M params)."""
    return MAGVIT2Encoder(base_channels=64, n_res_blocks=2, latent_dim=16, **kwargs)


def magvit2_base(**kwargs) -> MAGVIT2Encoder:
    """MAGVIT-v2 Base. ``base_channels=128, n_res_blocks=2`` (~80M params).

    Closest to the original paper's reference configuration.
    """
    return MAGVIT2Encoder(base_channels=128, n_res_blocks=2, latent_dim=16, **kwargs)


def magvit2_large(**kwargs) -> MAGVIT2Encoder:
    """MAGVIT-v2 Large. ``base_channels=192, n_res_blocks=3`` (~180M params)."""
    return MAGVIT2Encoder(base_channels=192, n_res_blocks=3, latent_dim=32, **kwargs)


def magvit2_huge(**kwargs) -> MAGVIT2Encoder:
    """MAGVIT-v2 Huge. ``base_channels=256, n_res_blocks=4`` (~370M params)."""
    return MAGVIT2Encoder(base_channels=256, n_res_blocks=4, latent_dim=32, **kwargs)


def magvit2_giant(**kwargs) -> MAGVIT2Encoder:
    """MAGVIT-v2 Giant. ``base_channels=384, n_res_blocks=4`` (~850M params).

    Scaling experiment territory — no published reference at this size.
    """
    return MAGVIT2Encoder(base_channels=384, n_res_blocks=4, latent_dim=64, **kwargs)


def magvit2_gigantic(**kwargs) -> MAGVIT2Encoder:
    """MAGVIT-v2 Gigantic. ``base_channels=512, n_res_blocks=6`` (~1.8B params).

    Scaling experiment territory — no published reference at this size.
    """
    return MAGVIT2Encoder(base_channels=512, n_res_blocks=6, latent_dim=64, **kwargs)
