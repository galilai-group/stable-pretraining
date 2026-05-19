"""Recurrent ViT — per-frame ViT spatial pass + GRU temporal mixer.

Reference
---------
Folded in from the V-JEPA-style ``RecurrentEncoder`` of the *deepstats* video
sweeps. The architecture is a per-frame ViT (CLS-pooled) followed by a GRU
on the per-frame CLS sequence — temporal recurrence happens on the **pooled
D-vector** rather than per-spatial-position, which makes it the cheapest
per-epoch encoder in our CALVIN sweeps.

Design
------
- Per-frame patch embed: ``Conv2d(in_channels, embed_dim, kernel=stride=patch_size)``
  flattens to ``(P, D)`` with ``P = (img_size / patch_size)²``.
- Learnable spatial positional embedding ``(1, P, D)``. No RoPE — the
  spatial stack uses an additive learned pos embed.
- Per-frame CLS token prepended; the full ``(1+P, D)`` sequence goes through
  ``spatial_depth`` :class:`~stable_pretraining.backbone.vit.TransformerBlock`
  layers (``self_attn=True, use_rope=None, mlp_type='gelu'``). QK-norm is
  off by default and only enabled on the ``huge`` preset.
- ``LayerNorm`` (``norm_s``) then split: CLS → ``(B, T, D)``, patches →
  ``(B, T, P, D)``.
- ``nn.GRU(D, D, num_layers=gru_layers, batch_first=True)`` walks the CLS
  sequence causally — frame ``t``'s output sees only frames ``0..t`` by the
  GRU's construction, so no attention mask plumbing is needed.
- ``LayerNorm`` (``norm_t``) on the GRU output produces ``tokens``.
- ``feature_map`` is the per-frame patch grid reshaped to
  ``(B, embed_dim, T, grid, grid)``; ``pooled`` is its global mean.

Causality
---------
The spatial branch has no temporal mixing (per-frame ViT runs frames
independently), and the GRU is causal by construction. Together they
guarantee that outputs at time ``t`` depend only on inputs at times
``0..t`` — verified by ``test_no_future_leakage`` on both ``feature_map``
and ``tokens``.

Scaling
-------
Factory presets follow the standard ViT family (tiny / small / base /
large / huge). No ``giant`` or ``gigantic`` preset — backprop-through-time
through a GRU at that scale is an open research problem (same justification
PredRNN-v2 uses).

Example::

    enc = recurrent_vit_tiny()
    x = torch.randn(2, 3, 8, 64, 64)
    out = enc(x)
    out.feature_map.shape  # (2, 192, 8, 8, 8)
    out.pooled.shape  # (2, 192)
    out.tokens.shape  # (2, 8, 192)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from ..vit import TransformerBlock


@dataclass
class RecurrentViTOutput(ModelOutput):
    """Structured output of :class:`RecurrentViT`.

    :param feature_map: ``(B, embed_dim, T, grid, grid)`` — per-frame patch
        tokens from the spatial ViT, reshaped into a regular grid. No
        temporal mixing on this view (spatial ViT runs each frame
        independently).
    :param pooled: ``(B, embed_dim)`` global average of ``feature_map`` over
        ``(T, grid, grid)`` when ``global_pool='avg'``, else ``None``.
    :param tokens: ``(B, T, embed_dim)`` — post-GRU per-frame CLS sequence.
        This is the temporally mixed view suitable for JEPA-style training
        that consumes per-frame embeddings directly.
    """

    feature_map: torch.Tensor = None
    pooled: Optional[torch.Tensor] = None
    tokens: Optional[torch.Tensor] = None


class _PatchEmbed(nn.Module):
    """Per-frame patchify via ``Conv2d``. Returns ``(N, P, D)``."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, dim: int):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size {img_size} not divisible by patch_size {patch_size}"
            )
        self.proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        self.grid = img_size // patch_size
        self.P = self.grid * self.grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (N, P, D)


class RecurrentViT(nn.Module):
    """Per-frame ViT + GRU temporal mixer.

    :param img_size: Spatial side length of the input (square). The spatial
        positional embedding is sized for this; runtime input must match.
    :param patch_size: Patch side length. Must divide ``img_size``.
    :param in_channels: Input channel count (3 for RGB).
    :param embed_dim: ViT hidden dimension ``D``.
    :param spatial_depth: Number of :class:`TransformerBlock` layers in the
        per-frame ViT.
    :param num_heads: Attention head count for the spatial blocks. Must
        divide ``embed_dim``.
    :param gru_layers: Number of stacked GRU layers on the temporal axis.
    :param mlp_ratio: MLP expansion ratio inside each transformer block.
    :param use_qk_norm: Apply RMSNorm to Q/K before attention. Off by
        default — only meaningfully improves stability at huge scale, so
        the ``huge`` preset enables it.
    :param max_cams: When ``>1``, allocate a learnable per-cam embedding
        ``(max_cams, 1, embed_dim)`` and optionally accept a ``cam_id``
        tensor in :meth:`forward`. With the default ``max_cams=1`` the
        embedding is skipped entirely — callers that want multi-cam SSL
        (CALVIN, DROID) typically tile cams along the ``W`` axis before
        passing the video in.
    :param global_pool: ``'avg'`` (global mean of ``feature_map``) or ``''``
        (skip pooling and return ``pooled=None``).
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 384,
        spatial_depth: int = 12,
        num_heads: int = 6,
        gru_layers: int = 1,
        mlp_ratio: float = 4.0,
        use_qk_norm: bool = False,
        max_cams: int = 1,
        global_pool: str = "avg",
    ):
        super().__init__()
        if global_pool not in ("avg", ""):
            raise ValueError(f"global_pool must be 'avg' or '', got {global_pool!r}")
        if max_cams < 1:
            raise ValueError(f"max_cams must be >= 1, got {max_cams}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.spatial_depth = spatial_depth
        self.num_heads = num_heads
        self.gru_layers = gru_layers
        self.max_cams = max_cams
        self.global_pool = global_pool

        self.patch_embed = _PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.grid = self.patch_embed.grid
        self.P = self.patch_embed.P

        self.pos_s = nn.Parameter(torch.zeros(1, self.P, embed_dim))
        nn.init.trunc_normal_(self.pos_s, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if max_cams > 1:
            self.cam_embed = nn.Parameter(torch.zeros(max_cams, 1, embed_dim))
            nn.init.trunc_normal_(self.cam_embed, std=0.02)
        else:
            self.register_parameter("cam_embed", None)

        self.spatial_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    self_attn=True,
                    cross_attn=False,
                    use_rope=None,
                    use_qk_norm=use_qk_norm,
                    mlp_type="gelu",
                )
                for _ in range(spatial_depth)
            ]
        )
        self.norm_s = nn.LayerNorm(embed_dim)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.norm_t = nn.LayerNorm(embed_dim)

    def forward(
        self,
        video: torch.Tensor,
        cam_id: Optional[torch.Tensor] = None,
    ) -> RecurrentViTOutput:
        """Encode a video clip.

        :param video: ``(B, C, T, H, W)`` tensor. ``H`` and ``W`` must equal
            the ``img_size`` the model was built with.
        :param cam_id: Optional ``(B,)`` int64 tensor naming the camera
            each clip came from. Used only when ``max_cams > 1``; ignored
            otherwise.
        :return: :class:`RecurrentViTOutput`.
        """
        if video.dim() != 5:
            raise ValueError(
                f"video must be (B, C, T, H, W); got shape {tuple(video.shape)}"
            )
        B, C, T, H, W = video.shape
        if H != self.img_size or W != self.img_size:
            raise ValueError(
                f"spatial size ({H}, {W}) does not match img_size={self.img_size}"
            )

        # (B, C, T, H, W) -> (B*T, C, H, W) so the per-frame ViT runs each
        # frame independently. The spatial branch has no temporal mixing —
        # that's the GRU's job.
        x = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.patch_embed(x)  # (B*T, P, D)
        x = x + self.pos_s

        if self.cam_embed is not None:
            if cam_id is None:
                cam_id = torch.zeros(B, dtype=torch.long, device=x.device)
            if cam_id.shape != (B,):
                raise ValueError(
                    f"cam_id must be shape (B,) = ({B},); got {tuple(cam_id.shape)}"
                )
            # (B, 1, D) -> broadcast to (B*T, 1, D) so each frame of clip i
            # carries clip i's cam embedding.
            ce = self.cam_embed[cam_id]
            ce = (
                ce.unsqueeze(1)
                .expand(B, T, 1, self.embed_dim)
                .reshape(B * T, 1, self.embed_dim)
            )
            x = x + ce

        cls = self.cls_token.expand(B * T, 1, self.embed_dim)
        x = torch.cat([cls, x], dim=1)  # (B*T, 1+P, D)

        for blk in self.spatial_blocks:
            x = blk(x)
        x = self.norm_s(x)

        cls_seq = x[:, 0, :].reshape(B, T, self.embed_dim)
        patches = x[:, 1:, :]  # (B*T, P, D)

        gru_out, _ = self.gru(cls_seq)
        tokens = self.norm_t(gru_out)  # (B, T, D)

        # Patches → feature_map. The per-patch reshape lays them onto a
        # regular (grid, grid) tile that matches the rest of the video zoo's
        # (B, D, T', H', W') contract.
        feature_map = (
            patches.reshape(B, T, self.grid, self.grid, self.embed_dim)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )
        pooled = feature_map.mean(dim=(2, 3, 4)) if self.global_pool == "avg" else None

        return RecurrentViTOutput(feature_map=feature_map, pooled=pooled, tokens=tokens)


# -----------------------------------------------------------------------------
# Scaling presets — ViT-aligned (tiny / small / base / large / huge). No
# giant / gigantic preset: training backprop-through-time through a GRU at
# that scale is an open research problem (same justification PredRNN-v2 uses).
# -----------------------------------------------------------------------------


def recurrent_vit_tiny(**kwargs) -> RecurrentViT:
    """RecurrentViT Tiny. ``embed_dim=192, depth=12, heads=3, gru=1`` (~5M params)."""
    return RecurrentViT(
        embed_dim=192, spatial_depth=12, num_heads=3, gru_layers=1, **kwargs
    )


def recurrent_vit_small(**kwargs) -> RecurrentViT:
    """RecurrentViT Small. ``embed_dim=384, depth=12, heads=6, gru=1`` (~20M params)."""
    return RecurrentViT(
        embed_dim=384, spatial_depth=12, num_heads=6, gru_layers=1, **kwargs
    )


def recurrent_vit_base(**kwargs) -> RecurrentViT:
    """RecurrentViT Base. ``embed_dim=768, depth=12, heads=12, gru=2`` (~80M params)."""
    return RecurrentViT(
        embed_dim=768, spatial_depth=12, num_heads=12, gru_layers=2, **kwargs
    )


def recurrent_vit_large(**kwargs) -> RecurrentViT:
    """RecurrentViT Large. ``embed_dim=1024, depth=24, heads=16, gru=2`` (~290M params)."""
    return RecurrentViT(
        embed_dim=1024, spatial_depth=24, num_heads=16, gru_layers=2, **kwargs
    )


def recurrent_vit_huge(**kwargs) -> RecurrentViT:
    """RecurrentViT Huge. ``embed_dim=1280, depth=32, heads=16, gru=2`` (~600M params).

    QK-norm is enabled here (and only here): at this width the deep
    attention stack starts to need the Q/K stability guard. Scaling
    experiment territory — BPTT through a GRU at this width is an open
    research problem.
    """
    kwargs.setdefault("use_qk_norm", True)
    return RecurrentViT(
        embed_dim=1280, spatial_depth=32, num_heads=16, gru_layers=2, **kwargs
    )
