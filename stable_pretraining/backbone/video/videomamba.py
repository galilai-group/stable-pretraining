"""VideoMamba (state-space video encoder).

Reference
---------
Li et al., "VideoMamba: State Space Model for Efficient Video
Understanding", ICML 2024 (https://arxiv.org/abs/2403.06977). Stacks
bidirectional Mamba (S6) blocks over a spatiotemporal token sequence —
gives linear-time compute in the number of tokens, scaling to long
clips where attention becomes prohibitive.

Design
------
This module ships a **pure-PyTorch reference implementation** of the
Mamba S6 selective state-space scan. It is correct on CPU and GPU,
compiles cleanly under ``torch.compile``, and is the path tested in CI.
The reference scan is *sequential* over the token sequence (a Python
loop over ``L``) — slow at large ``L``. For production-scale training on
CUDA, swap :class:`MambaSSMBlock` for the fast kernel from the
``mamba-ssm`` package (``selective_scan_cuda``); the surrounding code is
unchanged.

Token ordering
--------------
Tokens are flattened **spatial-first** — for each frame, all spatial
tokens come before any token of the next frame (i.e. ordering
``(t, h, w)`` with ``t`` outermost). With ``causal=True`` this guarantees
that any token of frame ``t`` cannot influence tokens of frame ``t' < t``
through the forward-only scan, so the encoder is **strictly causal in
time** — verified by ``test_no_future_leakage``.

With ``causal=False`` (the paper default for understanding tasks), each
block runs both a forward and a backward scan and sums them — better
representations, but no temporal causality.

Scaling
-------
Width via ``embed_dim``, depth via ``depth``. ``tiny`` / ``small`` /
``base`` match the original paper's published configs (in causal mode);
``large`` / ``huge`` / ``giant`` / ``gigantic`` extend the ladder into
scaling-experiment territory.

Example::

    enc = videomamba_tiny(causal=True, num_frames=16, img_size=224)
    out = enc(torch.randn(2, 3, 16, 224, 224))
    out.feature_map.shape  # (2, num_tokens, 192)
    out.pooled.shape  # (2, 192)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from transformers.utils import ModelOutput

from ..pos_embed import get_3d_sincos_pos_embed


@dataclass
class VideoMambaOutput(ModelOutput):
    """Structured output of :class:`VideoMamba`.

    :param feature_map: ``(B, embed_dim, T', H', W')`` — patch tokens
        reshaped to a 5D feature map. This matches the output convention
        of every other video encoder in this subpackage (MAGVIT-v2,
        Cosmos, PredRNN-v2) so the four families are interchangeable
        downstream. CLS / prefix tokens are dropped from this view —
        access them via ``tokens``.
    :param tokens: ``(B, num_prefix + T'*H'*W', embed_dim)`` — the raw
        post-norm token sequence including any CLS / register tokens.
        Use this when you need the CLS feature explicitly or want to
        feed an attention-style decoder.
    :param pooled: ``(B, embed_dim)`` (or ``(B, num_classes)`` when a
        head is configured) when ``global_pool != ''``, else ``None``.
    """

    feature_map: torch.Tensor = None
    tokens: Optional[torch.Tensor] = None
    pooled: Optional[torch.Tensor] = None


# --- Mamba S6 selective state-space block (pure PyTorch reference) -----------


class MambaSSMBlock(nn.Module):
    """S6 selective state-space block (forward scan only, strictly causal).

    Faithful PyTorch port of the Mamba block (Gu & Dao, 2023). Parameter
    layout matches the official ``mamba-ssm`` implementation so weights
    can be loaded across the two — useful when training in pure-PyTorch
    on CPU then deploying with the fast kernel on CUDA.

    :param d_model: Token dimension.
    :param d_state: SSM state size (the ``N`` in the paper). 16 is the
        published default.
    :param d_conv: Causal 1D conv kernel size applied to the inner
        projection. 4 is the published default.
    :param expand: Inner dimension multiplier (``d_inner = expand * d_model``).
        2 is the published default.
    :param dt_rank: Rank of the low-rank ``delta`` projection. ``None``
        uses ``ceil(d_model / 16)``, matching the paper.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
    ):
        super().__init__()
        d_inner = expand * d_model
        if dt_rank is None:
            dt_rank = math.ceil(d_model / 16)

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_inner
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        # Depthwise causal 1D conv along the sequence axis.
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=0,
            bias=True,
        )
        # x_proj produces (delta_low_rank, B, C) all in one Linear.
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        # dt_proj lifts the low-rank delta to d_inner with a learned bias
        # whose initialization controls the initial timescale (per Mamba).
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -(dt_rank**-0.5), dt_rank**-0.5)

        # A_log is stored in log space; A = -exp(A_log) so eigenvalues are
        # strictly in (-inf, 0), giving a stable continuous-time system.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SSM block.

        :param x: ``(B, L, d_model)``.
        :return: Same shape.
        """
        b, seq_len, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        u, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal 1D conv along sequence dim. F.pad with (k-1, 0) makes it strict.
        u_t = u.transpose(1, 2)  # (B, d_inner, L)
        u_t = F.pad(u_t, (self.d_conv - 1, 0))
        u_t = self.conv1d(u_t)  # (B, d_inner, L)
        u = u_t.transpose(1, 2)
        u = F.silu(u)

        x_dbl = self.x_proj(u)
        delta, B_p, C_p = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # Discretize (zero-order hold).
        # deltaA: (B, L, d_inner, d_state), deltaB_u: (B, L, d_inner, d_state)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_u = delta.unsqueeze(-1) * B_p.unsqueeze(2) * u.unsqueeze(-1)

        # Sequential selective scan. Slow on CPU; for CUDA scale, swap in
        # ``mamba_ssm.ops.selective_scan_fn`` — parameter layout matches.
        state = u.new_zeros(b, self.d_inner, self.d_state)
        ys = []
        for i in range(seq_len):
            state = deltaA[:, i] * state + deltaB_u[:, i]
            # y_i = state @ C_p[:, i]
            ys.append(torch.einsum("bdn,bn->bd", state, C_p[:, i]))
        y = torch.stack(ys, dim=1).to(x.dtype)  # (B, L, d_inner)

        y = y + u * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


# --- Video-level Mamba blocks ------------------------------------------------


class CausalMambaBlock(nn.Module):
    """Pre-norm Mamba block, causal (forward scan only)."""

    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaSSMBlock(
            d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class BiMambaBlock(nn.Module):
    """Pre-norm bidirectional Mamba block (forward + backward scans, summed).

    The two SSM blocks are independent parameters — twice the params of a
    causal block but the standard configuration used in the VideoMamba
    paper for understanding tasks.
    """

    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = MambaSSMBlock(
            d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.mamba_bwd = MambaSSMBlock(
            d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        f = self.mamba_fwd(h)
        b = self.mamba_bwd(h.flip(1)).flip(1)
        return x + f + b


# --- VideoMamba encoder ------------------------------------------------------


def _to_triple(x: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if isinstance(x, int):
        return (x, x, x)
    return tuple(x)


class VideoMamba(nn.Module):
    """VideoMamba encoder.

    :param img_size: Spatial input size (int or (H, W)).
    :param num_frames: Temporal input size ``T``. Pinning this keeps the
        flattened sequence length static under ``torch.compile``.
    :param patch_size: Tubelet kernel/stride as ``(t, h, w)`` or single int.
        Default ``(1, 16, 16)`` — 1-frame tubelets, 16×16 spatial patches.
    :param in_chans: Input channels.
    :param embed_dim: Token dimension.
    :param depth: Number of Mamba blocks.
    :param d_state: SSM state size.
    :param d_conv: Causal 1D conv kernel inside each Mamba block.
    :param expand: Inner-dim multiplier inside each Mamba block.
    :param causal: If True, use forward-only (causal) Mamba blocks. If
        False, use bidirectional blocks (paper default; not causal).
    :param class_token: Prepend a learnable CLS token if True.
    :param pos_embed_type: ``'learned'`` (default, matches the paper),
        ``'sincos_3d'`` (requires ``embed_dim`` divisible by 6), or
        ``'none'``.
    :param num_classes: If >0, add a linear classification head on the
        pooled feature. ``0`` returns features only.
    :param global_pool: ``'token'`` (CLS — requires ``class_token=True``),
        ``'avg'`` (mean of patch tokens), or ``''`` (return token sequence
        unchanged).
    :param use_checkpoint: If True, wrap each block in
        ``torch.utils.checkpoint``. Off by default — the sequential scan
        is already memory-light vs attention.

    Example::

        m = videomamba_tiny(causal=True, num_frames=8, img_size=64)
        out = m(torch.randn(1, 3, 8, 64, 64))
        out.pooled.shape  # (1, 192)
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        num_frames: int = 16,
        patch_size: Union[int, Tuple[int, int, int]] = (1, 16, 16),
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 24,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        causal: bool = True,
        class_token: bool = True,
        pos_embed_type: str = "learned",
        num_classes: int = 0,
        global_pool: str = "token",
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        pt, ph, pw = _to_triple(patch_size)
        if num_frames % pt != 0 or img_size[0] % ph != 0 or img_size[1] % pw != 0:
            raise ValueError(
                f"patch_size {(pt, ph, pw)} must divide (T, H, W) = ({num_frames}, {img_size[0]}, {img_size[1]})"
            )
        if global_pool not in ("token", "avg", ""):
            raise ValueError(
                f"global_pool must be 'token', 'avg', or '', got {global_pool!r}"
            )
        if global_pool == "token" and not class_token:
            raise ValueError("global_pool='token' requires class_token=True")

        self.img_size = img_size
        self.num_frames = num_frames
        self.patch_size = (pt, ph, pw)
        self.embed_dim = embed_dim
        self.causal = causal
        self.global_pool = global_pool
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes

        self.grid_size = (
            num_frames // pt,
            img_size[0] // ph,
            img_size[1] // pw,
        )
        n_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.patch_embed = nn.Conv3d(
            in_chans, embed_dim, kernel_size=(pt, ph, pw), stride=(pt, ph, pw)
        )

        if class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            num_prefix = 1
        else:
            self.register_parameter("cls_token", None)
            num_prefix = 0
        self.num_prefix = num_prefix

        total_tokens = num_prefix + n_patches
        if pos_embed_type == "sincos_3d":
            pe = get_3d_sincos_pos_embed(
                embed_dim, self.grid_size, cls_token=(num_prefix > 0)
            )
            self.register_buffer("pos_embed", pe.unsqueeze(0))
        elif pos_embed_type == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed_type == "none":
            self.register_parameter("pos_embed", None)
        else:
            raise ValueError(
                f"pos_embed_type must be 'sincos_3d', 'learned', or 'none', got {pos_embed_type!r}"
            )

        block_cls = CausalMambaBlock if causal else BiMambaBlock
        self.blocks = nn.ModuleList(
            [
                block_cls(embed_dim, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        if num_classes > 0:
            self.head = nn.Linear(embed_dim, num_classes)
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)
        else:
            self.head = nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a clip to a token sequence (post-norm).

        :param x: ``(B, C, T, H, W)``.
        :return: ``(B, num_prefix + N, embed_dim)``.
        """
        b = x.size(0)
        x = self.patch_embed(x)  # (B, embed_dim, T', H', W')
        # Spatial-first flatten — temporal axis is outermost, so causal
        # forward scan over the token sequence is causal in time.
        x = x.flatten(2).transpose(1, 2)  # (B, T'*H'*W', embed_dim)

        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = ckpt.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        return self.norm(x)

    def forward(self, x: torch.Tensor) -> VideoMambaOutput:
        tokens = self.forward_features(x)
        if self.global_pool == "token":
            pooled = tokens[:, 0]
        elif self.global_pool == "avg":
            pooled = tokens[:, self.num_prefix :].mean(dim=1)
        else:
            pooled = None

        if pooled is not None and self.num_classes > 0:
            pooled = self.head(pooled)

        # Reshape patch tokens to a 5D (B, C, T', H', W') feature map so
        # the output format matches the other video encoders.
        b = tokens.size(0)
        patch_tokens = tokens[:, self.num_prefix :]
        t, h, w = self.grid_size
        feature_map = (
            patch_tokens.transpose(1, 2)
            .reshape(b, self.embed_dim, t, h, w)
            .contiguous()
        )

        return VideoMambaOutput(feature_map=feature_map, tokens=tokens, pooled=pooled)


# -----------------------------------------------------------------------------
# Scaling presets. ``causal=True`` is the default — the user-facing motivation
# for this module is causal video encoding. Pass ``causal=False`` to get the
# paper's bidirectional configuration.
# -----------------------------------------------------------------------------


def videomamba_tiny(**kwargs) -> VideoMamba:
    """VideoMamba Tiny. ``embed_dim=192, depth=24`` (~7M causal / ~12M bi)."""
    return VideoMamba(embed_dim=192, depth=24, **kwargs)


def videomamba_small(**kwargs) -> VideoMamba:
    """VideoMamba Small. ``embed_dim=384, depth=24`` (~26M causal / ~46M bi)."""
    return VideoMamba(embed_dim=384, depth=24, **kwargs)


def videomamba_base(**kwargs) -> VideoMamba:
    """VideoMamba Base. ``embed_dim=576, depth=32`` (~74M causal / ~140M bi).

    Matches the paper's ``VideoMamba-M`` (middle) when ``causal=False``.
    """
    return VideoMamba(embed_dim=576, depth=32, **kwargs)


def videomamba_large(**kwargs) -> VideoMamba:
    """VideoMamba Large. ``embed_dim=1024, depth=32`` (~230M causal)."""
    return VideoMamba(embed_dim=1024, depth=32, **kwargs)


def videomamba_huge(**kwargs) -> VideoMamba:
    """VideoMamba Huge. ``embed_dim=1280, depth=48`` (~530M causal).

    Scaling experiment territory.
    """
    return VideoMamba(embed_dim=1280, depth=48, **kwargs)


def videomamba_giant(**kwargs) -> VideoMamba:
    """VideoMamba Giant. ``embed_dim=1664, depth=48`` (~900M causal).

    Scaling experiment territory.
    """
    return VideoMamba(embed_dim=1664, depth=48, **kwargs)


def videomamba_gigantic(**kwargs) -> VideoMamba:
    """VideoMamba Gigantic. ``embed_dim=2048, depth=64`` (~1.8B causal).

    Scaling experiment territory.
    """
    return VideoMamba(embed_dim=2048, depth=64, **kwargs)
