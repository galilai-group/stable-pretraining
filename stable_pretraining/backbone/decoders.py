"""Image decoders for online reconstruction probes.

Two decoder families that map representations back to images:

* :class:`CNNImageDecoder` — flat vector ``(B, D)`` to image
  ``(B, C, H, W)``. LDM / Stable-Diffusion VAE-decoder-style: nearest
  upsampling + 3x3 conv (no ``ConvTranspose2d``, which is prone to
  checkerboard artifacts), ``GroupNorm`` + ``SiLU``, residual blocks. Works
  for any ``out_chans`` (3 for RGB, 1 for depth, etc.).

* :class:`ViTImageDecoder` — token grid ``(B, P, D)`` to image
  ``(B, C, H, W)``. MAE-style pixel decoder for the full sequence (no mask
  reasoning): linear projection + learned positional embedding + a few
  transformer blocks + per-token pixel head + unpatchify.

Both decoders validate inputs on the forward pass and raise informative
errors if a caller passes the wrong rank or feature dim — this matters in
practice because the most common bug is feeding token grids into the CNN
decoder (or vice versa).

References:
----------
* Rombach et al., 2022 — High-Resolution Image Synthesis with Latent
  Diffusion Models. (decoder block pattern: GroupNorm + SiLU + Conv with
  residual + nearest upsample)
* He et al., 2022 — Masked Autoencoders Are Scalable Vision Learners.
  (ViT pixel decoder, unpatchify head)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as logging


# ---------------------------------------------------------------------------
# CNN decoder (LDM / VQ-VAE style)
# ---------------------------------------------------------------------------


def _group_norm(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(
        num_groups=min(num_groups, channels), num_channels=channels, eps=1e-6
    )


class _ResBlock(nn.Module):
    """LDM-style residual block: GN -> SiLU -> Conv -> GN -> SiLU -> Conv + skip."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv2d(in_ch, out_ch, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class _Upsample(nn.Module):
    """Nearest 2x upsample + 3x3 conv — avoids transpose-conv checkerboard."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class CNNImageDecoder(nn.Module):
    """Flat embedding ``(B, D)`` -> image ``(B, out_chans, H, W)``.

    Architecture follows the LDM / Stable Diffusion VAE decoder pattern:
    a linear projection lifts the embedding to a small spatial grid, then a
    cascade of ``ResBlock x 2 + Upsample(nearest + Conv)`` stages doubles the
    spatial resolution until the target ``img_size`` is reached. A final
    ``GroupNorm + SiLU + Conv`` projects to ``out_chans``.

    Suitable for any output modality with arbitrary channel count (RGB,
    depth, segmentation logits, etc.) — pick ``out_chans`` accordingly.

    Parameters
    ----------
    embed_dim
        Input feature dimension ``D``.
    img_size
        Target spatial resolution. Must be a power-of-two multiple of
        ``start_size`` (default 4): e.g. 32, 64, 96, 128, 256.
    out_chans
        Number of output channels (3 for RGB, 1 for depth, ...).
    base_channels
        Channel count at the lowest spatial resolution; halved at each
        upsampling stage down to a minimum of ``min_channels``.
    min_channels
        Floor for the channel ladder.
    start_size
        Spatial size of the initial feature map after the linear lift.
        ``img_size / start_size`` must be a power of two; the number of
        ``2x`` upsample stages is ``log2(img_size / start_size)``.
    num_res_blocks
        Number of residual blocks per stage (LDM default: 2).
    """

    def __init__(
        self,
        embed_dim: int,
        img_size: int,
        out_chans: int = 3,
        base_channels: int = 512,
        min_channels: int = 32,
        start_size: int = 4,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        if img_size < start_size or img_size % start_size != 0:
            raise ValueError(
                f"CNNImageDecoder: img_size ({img_size}) must be a multiple of "
                f"start_size ({start_size})."
            )
        ratio = img_size // start_size
        n_stages = int(round(math.log2(ratio)))
        if 2**n_stages != ratio:
            raise ValueError(
                f"CNNImageDecoder: img_size / start_size ({ratio}) must be a "
                f"power of two; got img_size={img_size}, start_size={start_size}."
            )

        self.embed_dim = embed_dim
        self.img_size = img_size
        self.out_chans = out_chans
        self.start_size = start_size
        self.num_stages = n_stages

        channels = [
            max(base_channels // (2**i), min_channels) for i in range(n_stages + 1)
        ]
        self.channels = channels  # for introspection / debugging

        # Lift embedding to (C0, start_size, start_size).
        self.fc = nn.Linear(embed_dim, channels[0] * start_size * start_size)
        self.conv_in = nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1)

        stages = []
        for i in range(n_stages):
            c_in, c_out = channels[i], channels[i + 1]
            stage = nn.ModuleList()
            stage.append(_ResBlock(c_in, c_out))
            for _ in range(num_res_blocks - 1):
                stage.append(_ResBlock(c_out, c_out))
            stage.append(_Upsample(c_out))
            stages.append(stage)
        self.stages = nn.ModuleList(stages)

        self.norm_out = _group_norm(channels[-1])
        self.conv_out = nn.Conv2d(channels[-1], out_chans, kernel_size=3, padding=1)

        logging.info(
            f"CNNImageDecoder: D={embed_dim} -> ({out_chans}, {img_size}, {img_size}) | "
            f"stages={n_stages}, channels={channels}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(
                f"CNNImageDecoder expects a 2-D input (B, D); got shape {tuple(z.shape)}. "
                f"If your features are token grids (B, P, D) use ViTImageDecoder instead, "
                f"or pool over the patch dim before passing in."
            )
        if z.shape[1] != self.embed_dim:
            raise ValueError(
                f"CNNImageDecoder expects D={self.embed_dim} on dim 1; got {z.shape[1]}. "
                f"Pass the matching embed_dim at construction."
            )
        h = self.fc(z).view(-1, self.channels[0], self.start_size, self.start_size)
        h = self.conv_in(h)
        for stage in self.stages:
            for block in stage:
                h = block(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


# ---------------------------------------------------------------------------
# ViT decoder (MAE-style pixel head over full token sequence)
# ---------------------------------------------------------------------------


class ViTImageDecoder(nn.Module):
    """Token grid ``(B, P, D)`` -> image ``(B, out_chans, H, W)``.

    MAE-style pixel decoder for the full (unmasked) sequence: project to a
    decoder width, add learned positional embeddings, run a small transformer
    stack, then a per-token linear head emits ``patch_size**2 * out_chans``
    values per token which are unpatchified into the final image.

    Use this when your encoder exposes a patch token grid (e.g. ViT pre-pool
    features) and you want pixel-level reconstruction. The number of tokens
    ``P`` must equal ``(img_size / patch_size)**2``.

    Parameters
    ----------
    embed_dim
        Per-token input dimension ``D``.
    img_size
        Reconstruction spatial size. Must be divisible by ``patch_size``.
    patch_size
        Patch side length. Must satisfy ``img_size % patch_size == 0``.
    out_chans
        Number of output channels.
    decoder_dim
        Internal transformer width. MAE default 512; we use 256 to stay light
        for online use.
    depth
        Number of transformer blocks. MAE default 8; default here 4 for speed.
    num_heads
        Attention heads.
    mlp_ratio
        MLP expansion ratio.
    """

    def __init__(
        self,
        embed_dim: int,
        img_size: int,
        patch_size: int,
        out_chans: int = 3,
        decoder_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"ViTImageDecoder: img_size ({img_size}) must be divisible by "
                f"patch_size ({patch_size})."
            )
        # Local import: TransformerBlock lives in the same backbone package and
        # importing at module top-level would create a circular import via
        # backbone/__init__.py.
        from .vit import TransformerBlock

        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid

        self.proj_in = nn.Linear(embed_dim, decoder_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=decoder_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    self_attn=True,
                    cross_attn=False,
                    use_qk_norm=True,
                    mlp_type="gelu",
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.pixel_head = nn.Linear(decoder_dim, patch_size * patch_size * out_chans)

        logging.info(
            f"ViTImageDecoder: (P={self.num_patches}, D={embed_dim}) -> "
            f"({out_chans}, {img_size}, {img_size}) | depth={depth}, "
            f"decoder_dim={decoder_dim}, patch_size={patch_size}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                f"ViTImageDecoder expects a 3-D input (B, P, D); got shape "
                f"{tuple(tokens.shape)}. If your features are pooled vectors "
                f"(B, D) use CNNImageDecoder."
            )
        B, P, D = tokens.shape
        if D != self.embed_dim:
            raise ValueError(
                f"ViTImageDecoder expects D={self.embed_dim} on dim 2; got {D}."
            )
        if P != self.num_patches:
            raise ValueError(
                f"ViTImageDecoder expects P={self.num_patches} tokens "
                f"(grid={self.grid}x{self.grid} from img_size={self.img_size}, "
                f"patch_size={self.patch_size}); got P={P}."
            )
        x = self.proj_in(tokens) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.pixel_head(x)
        # (B, P, ps*ps*C) -> (B, C, H, W)
        ps, g, C = self.patch_size, self.grid, self.out_chans
        x = x.reshape(B, g, g, ps, ps, C)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, g * ps, g * ps)
        return x


# ---------------------------------------------------------------------------
# Auto-selection helper
# ---------------------------------------------------------------------------


def build_image_decoder(
    embed_dim: int,
    image_shape: Tuple[int, int, int],
    kind: str = "auto",
    patch_size: Optional[int] = None,
    decoder_kwargs: Optional[dict] = None,
) -> nn.Module:
    """Construct a CNN or ViT image decoder from a compact spec.

    Parameters
    ----------
    embed_dim
        Encoder feature dimension ``D``.
    image_shape
        Target image shape ``(out_chans, height, width)``. ``height`` must
        equal ``width`` (square images only — current decoders are square).
    kind
        ``"cnn"``, ``"vit"``, or ``"auto"``. ``"auto"`` resolves to ``"vit"``
        if ``patch_size`` is provided, else ``"cnn"``.
    patch_size
        Required for ``vit``. Must divide ``image_shape[1]``.
    decoder_kwargs
        Extra kwargs forwarded to the chosen decoder constructor.
    """
    decoder_kwargs = dict(decoder_kwargs or {})
    if len(image_shape) != 3:
        raise ValueError(f"image_shape must be (C, H, W); got {image_shape}.")
    C, H, W = image_shape
    if H != W:
        raise ValueError(
            f"Image decoders currently require square images; got H={H}, W={W}."
        )
    resolved = kind
    if kind == "auto":
        resolved = "vit" if patch_size is not None else "cnn"
    if resolved == "cnn":
        if patch_size is not None:
            logging.warning(
                "build_image_decoder: patch_size is ignored for kind='cnn'."
            )
        return CNNImageDecoder(
            embed_dim=embed_dim, img_size=H, out_chans=C, **decoder_kwargs
        )
    if resolved == "vit":
        if patch_size is None:
            raise ValueError("kind='vit' requires a patch_size.")
        return ViTImageDecoder(
            embed_dim=embed_dim,
            img_size=H,
            patch_size=patch_size,
            out_chans=C,
            **decoder_kwargs,
        )
    raise ValueError(f"Unknown kind={kind!r}; expected one of 'auto', 'cnn', 'vit'.")
