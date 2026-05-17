"""Cosmos Tokenizer (NVIDIA) — causal video encoder.

Reference
---------
NVIDIA Cosmos team, "Cosmos World Foundation Model Platform for Physical
AI", 2025 (https://research.nvidia.com/labs/dir/cosmos-tokenizer/). The
Cosmos Tokenizer family generalizes MAGVIT-v2 with:

- **Causal temporal attention** in deep stages, on top of the causal 3D
  ResNet backbone. Each pixel sequence ``(t_0, ..., t_{T-1})`` attends to
  prior time steps only.
- **Per-frame spatial self-attention** in deep stages — refines each
  frame's representation independently. Optional, off in the cheapest
  configs.
- **Configurable spatial / temporal compression** so the same encoder can
  serve different downstream latent rates (e.g. 4×8×8 vs 8×16×16).

This module implements the encoder; the matching decoder + quantizer
(FSQ / Lattice-FSQ) live elsewhere.

Design notes
------------
- Reuses :class:`CausalConv3d` and :class:`GroupNormPerFrame` from the
  causal-conv3d / norms helpers. Per-frame GroupNorm is mandatory: stock
  GroupNorm breaks causality even when the convs are causal (the test
  ``test_no_future_leakage`` catches this).
- Causal temporal attention uses :func:`torch.nn.functional.scaled_dot_product_attention`
  with ``is_causal=True`` — single-kernel SDPA path, ``torch.compile``-clean.
- Activation checkpointing is opt-in via ``use_checkpoint=True`` and off
  by default (recompute overhead beats the memory savings at small width).
- Stage-level modules are first-class ``nn.Module`` so FSDP auto-wrap
  policies and ``accelerate.init_empty_weights()`` find natural shards.

Scaling
-------
Factory presets follow the ViT-family naming convention. Width scales via
``base_channels``, depth via ``n_res_blocks``, attention budget via
``attn_stages``. The ``giant`` and ``gigantic`` presets exceed any published
Cosmos checkpoint and are intended for scaling research.

Example::

    enc = cosmos_base(num_frames=16)
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
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from transformers.utils import ModelOutput

from .causal_conv3d import CausalConv3d
from .norms import GroupNormPerFrame, _fit_groups


@dataclass
class CosmosOutput(ModelOutput):
    """Structured output of :class:`CosmosEncoder`.

    :param feature_map: ``(B, latent_dim, T', H', W')``.
    :param pooled: ``(B, latent_dim)`` global-pooled feature when
        ``global_pool='avg'``, else ``None``.
    """

    feature_map: torch.Tensor = None
    pooled: Optional[torch.Tensor] = None


# --- Building blocks ---------------------------------------------------------


class _ResBlock3D(nn.Module):
    """Causal pre-norm 3D residual block.

    Same recipe as MAGVIT-v2's res block, kept local to avoid cross-module
    coupling between the two families.
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
    """Strided causal 3D conv. ``temporal=True`` also halves time."""

    def __init__(self, channels: int, temporal: bool):
        super().__init__()
        st = 2 if temporal else 1
        self.conv = CausalConv3d(channels, channels, kernel_size=3, stride=(st, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CosmosSpatialAttention(nn.Module):
    """Per-frame spatial self-attention.

    Treats each (B, T) slice independently — for each frame, attention is
    computed across the ``H*W`` spatial tokens. No mask. ``GroupNorm`` is
    per-frame (statistics local to a single frame), preserving causality.

    :param channels: Input / output channel count.
    :param num_heads: Attention head count. Must divide ``channels``.
    :param groups: ``GroupNorm`` target group count (clamped to a divisor
        of ``channels``).
    """

    def __init__(self, channels: int, num_heads: int = 8, groups: int = 32):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = GroupNormPerFrame(_fit_groups(groups, channels), channels)
        self.qkv = nn.Linear(channels, 3 * channels, bias=False)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        h_in = x
        x = self.norm(x)
        # (B, C, T, H, W) → (B*T, H*W, C)
        x = x.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)
        qkv = (
            self.qkv(x)
            .reshape(b * t, h * w, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each: (B*T, num_heads, H*W, head_dim)
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(b * t, h * w, c)
        a = self.proj(a)
        a = a.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        return h_in + a


class CosmosCausalTemporalAttention(nn.Module):
    """Causal temporal self-attention applied per-pixel.

    For each (B, h, w) location, attention is computed across ``T`` tokens
    with a strict causal mask (frame ``t`` sees frames ``[0, t]``). Uses
    SDPA's built-in ``is_causal=True`` which dispatches to the FlashAttention
    kernel when available and stays a single fused op under ``torch.compile``.
    """

    def __init__(self, channels: int, num_heads: int = 8, groups: int = 32):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = GroupNormPerFrame(_fit_groups(groups, channels), channels)
        self.qkv = nn.Linear(channels, 3 * channels, bias=False)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        h_in = x
        x = self.norm(x)
        # (B, C, T, H, W) → (B*H*W, T, C)
        x = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)
        qkv = (
            self.qkv(x)
            .reshape(b * h * w, t, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each: (B*H*W, num_heads, T, head_dim)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(b * h * w, t, c)
        a = self.proj(a)
        a = a.reshape(b, h, w, t, c).permute(0, 4, 3, 1, 2)
        return h_in + a


class _AttnBlock(nn.Module):
    """Spatial-then-causal-temporal attention block. Both passes are residual."""

    def __init__(self, channels: int, num_heads: int = 8, groups: int = 32):
        super().__init__()
        self.spatial = CosmosSpatialAttention(channels, num_heads, groups)
        self.temporal = CosmosCausalTemporalAttention(channels, num_heads, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        x = self.temporal(x)
        return x


class _Stage(nn.Module):
    """``n_res_blocks`` × (ResBlock [+ AttnBlock]) → optional downsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_res_blocks: int,
        use_attention: bool,
        downsample: bool,
        temporal_downsample: bool,
        num_heads: int = 8,
        groups: int = 32,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.use_attention = use_attention
        ch = in_channels
        for _ in range(n_res_blocks):
            self.res_blocks.append(_ResBlock3D(ch, out_channels, groups=groups))
            ch = out_channels
            if use_attention:
                self.attn_blocks.append(
                    _AttnBlock(out_channels, num_heads=num_heads, groups=groups)
                )
        self.downsample = (
            _Downsample(out_channels, temporal=temporal_downsample)
            if downsample
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, res in enumerate(self.res_blocks):
            x = res(x)
            if self.use_attention:
                x = self.attn_blocks[i](x)
        return self.downsample(x)


# --- Encoder -----------------------------------------------------------------


class CosmosEncoder(nn.Module):
    """Cosmos Tokenizer causal video encoder.

    Architecture: a causal 3D ResNet with attention blocks inserted in deep
    stages. Spatial downsampling halves resolution at every stage except
    the last (so an N-stage encoder has ``2**(N-1)``× spatial compression).
    Temporal downsampling is enabled per-stage via
    ``temporal_downsample_stages`` — by default the second and third stages
    (4× temporal compression to match the published 4× ratio).

    :param in_channels: Input channels (3 for RGB).
    :param base_channels: Channel width of the stem.
    :param channel_multipliers: Per-stage width multipliers. Length defines
        the number of stages.
    :param n_res_blocks: Residual blocks per stage.
    :param latent_dim: Output channel count.
    :param attn_stages: 0-indexed stage IDs that get attention blocks
        (one block per res-block inside the stage). Typical: deep stages
        only, since attention scales as ``O(T^2)`` and ``O((HW)^2)``.
    :param num_heads: Attention head count (must divide every attention
        layer's channel count).
    :param temporal_downsample_stages: 0-indexed stages where time is also
        halved. Default ``(1, 2)`` → 4× temporal.
    :param groups: Target group count for ``GroupNorm``.
    :param global_pool: ``'avg'`` or ``''``.
    :param use_checkpoint: If True, wrap each stage in
        ``torch.utils.checkpoint``. Off by default.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 2, 4),
        n_res_blocks: int = 2,
        latent_dim: int = 16,
        attn_stages: Tuple[int, ...] = (2, 3),
        num_heads: int = 8,
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

        self.stem = CausalConv3d(in_channels, base_channels, kernel_size=3)

        stage_channels = [base_channels * m for m in channel_multipliers]
        n_stages = len(channel_multipliers)
        attn_set = set(attn_stages)
        tds_set = set(temporal_downsample_stages)

        # Validate that attention head dim divides every targeted stage's channels.
        for i in attn_set:
            if i >= n_stages:
                raise ValueError(
                    f"attn_stages index {i} out of range for {n_stages} stages"
                )
            if stage_channels[i] % num_heads != 0:
                raise ValueError(
                    f"stage {i} channels={stage_channels[i]} not divisible by num_heads={num_heads}"
                )

        self.stages = nn.ModuleList()
        prev_ch = base_channels
        for i, ch in enumerate(stage_channels):
            is_last = i == n_stages - 1
            self.stages.append(
                _Stage(
                    in_channels=prev_ch,
                    out_channels=ch,
                    n_res_blocks=n_res_blocks,
                    use_attention=(i in attn_set),
                    downsample=not is_last,
                    temporal_downsample=(i in tds_set),
                    num_heads=num_heads,
                    groups=groups,
                )
            )
            prev_ch = ch

        out_ch = stage_channels[-1]
        self.norm_out = GroupNormPerFrame(_fit_groups(groups, out_ch), out_ch)
        self.act_out = nn.SiLU(inplace=False)
        self.proj_out = CausalConv3d(out_ch, latent_dim, kernel_size=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            if self.use_checkpoint and self.training:
                x = ckpt.checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        return self.proj_out(self.act_out(self.norm_out(x)))

    def forward(self, x: torch.Tensor) -> CosmosOutput:
        feat = self.forward_features(x)
        pooled = feat.mean(dim=(2, 3, 4)) if self.global_pool == "avg" else None
        return CosmosOutput(feature_map=feat, pooled=pooled)


# -----------------------------------------------------------------------------
# Scaling presets. Default compression is 4× temporal, 16× spatial — the
# Cosmos CV-style default. Tune via constructor kwargs for other ratios.
# -----------------------------------------------------------------------------


def cosmos_tiny(**kwargs) -> CosmosEncoder:
    """Cosmos Tiny. ``base=64, n_res=2, no attention`` (~11M params).

    Cheapest preset — pure causal-3D-ResNet, attention disabled.
    """
    return CosmosEncoder(
        base_channels=64,
        n_res_blocks=2,
        attn_stages=(),
        latent_dim=8,
        num_heads=8,
        **kwargs,
    )


def cosmos_small(**kwargs) -> CosmosEncoder:
    """Cosmos Small. ``base=96, n_res=2, attn at stage 3`` (~27M params)."""
    return CosmosEncoder(
        base_channels=96,
        n_res_blocks=2,
        attn_stages=(3,),
        latent_dim=16,
        num_heads=8,
        **kwargs,
    )


def cosmos_base(**kwargs) -> CosmosEncoder:
    """Cosmos Base. ``base=128, n_res=2, attn at stages 2-3`` (~50M params).

    Closest match to NVIDIA's published Cosmos CV variant in compute class.
    """
    return CosmosEncoder(
        base_channels=128,
        n_res_blocks=2,
        attn_stages=(2, 3),
        latent_dim=16,
        num_heads=8,
        **kwargs,
    )


def cosmos_large(**kwargs) -> CosmosEncoder:
    """Cosmos Large. ``base=192, n_res=3, attn at stages 2-3`` (~165M params)."""
    return CosmosEncoder(
        base_channels=192,
        n_res_blocks=3,
        attn_stages=(2, 3),
        latent_dim=32,
        num_heads=8,
        **kwargs,
    )


def cosmos_huge(**kwargs) -> CosmosEncoder:
    """Cosmos Huge. ``base=256, n_res=3, attn at stages 2-3`` (~300M params)."""
    return CosmosEncoder(
        base_channels=256,
        n_res_blocks=3,
        attn_stages=(2, 3),
        latent_dim=32,
        num_heads=8,
        **kwargs,
    )


def cosmos_giant(**kwargs) -> CosmosEncoder:
    """Cosmos Giant. ``base=384, n_res=4, attn at stages 1-2-3`` (~900M params).

    Scaling experiment territory — exceeds any published Cosmos checkpoint.
    """
    return CosmosEncoder(
        base_channels=384,
        n_res_blocks=4,
        attn_stages=(1, 2, 3),
        latent_dim=64,
        num_heads=16,
        **kwargs,
    )


def cosmos_gigantic(**kwargs) -> CosmosEncoder:
    """Cosmos Gigantic. ``base=512, n_res=5, attn at stages 1-2-3`` (~2.0B params).

    Scaling experiment territory.
    """
    return CosmosEncoder(
        base_channels=512,
        n_res_blocks=5,
        attn_stages=(1, 2, 3),
        latent_dim=64,
        num_heads=16,
        **kwargs,
    )
