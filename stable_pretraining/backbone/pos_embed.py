"""Positional embedding utilities for vision transformers."""

import torch
import torch.nn.functional as F
from typing import Literal, Tuple
import math
from torch import nn

__all__ = [
    "get_sincos_pos_embed",
    "get_1d_sincos_pos_embed",
    "get_2d_sincos_pos_embed",
    "interpolate_pos_embed",
    "get_timestep_embed",
]


def get_timestep_embed(
    t: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    """Generate sinusoidal embeddings for continuous timesteps.

    Unlike positional embeddings for sequences, this embeds scalar timestep values.
    Used for diffusion/flow matching time conditioning.
    :param t: Timestep values (B,) or (B, 1), typically in [0, 1]
    :param dim: Embedding dimension
    :param max_period: Maximum period for frequency scaling
    :return: Timestep embeddings of shape (B, dim)
    """
    t = t.view(-1).float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / half
    )
    args = t[:, None] * freqs[None, :]
    embedding = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def get_1d_sincos_pos_embed(
    embed_dim: int,
    length: int,
    cls_token: bool = False,
) -> torch.Tensor:
    """Generate 1D sinusoidal positional embeddings.

    :param embed_dim: Embedding dimension
    :param length: Sequence length (number of positions)
    :param cls_token: If True, prepend a zero embedding for CLS token
    :return: Positional embeddings of shape (length, embed_dim) or
             (length + 1, embed_dim) if cls_token=True
    """
    if embed_dim <= 0:
        raise ValueError(f"embed_dim must be positive, got {embed_dim}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    dim = torch.arange(0, embed_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (dim / embed_dim))

    pe = torch.zeros(length, embed_dim)
    pe[:, 0::2] = torch.sin(pos * inv_freq)
    pe[:, 1::2] = torch.cos(pos * inv_freq[: embed_dim // 2])

    if cls_token:
        pe = torch.cat([torch.zeros(1, embed_dim), pe], dim=0)
    return pe


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int | tuple[int, int],
    cls_token: bool = False,
) -> torch.Tensor:
    """Generate 2D sinusoidal positional embeddings for image patches.

    :param embed_dim: Embedding dimension (must be divisible by 4)
    :param grid_size: Grid height/width as int (square) or (height, width) tuple
    :param cls_token: If True, prepend a zero embedding for CLS token
    :return: Positional embeddings of shape (H*W, embed_dim) or
             (H*W + 1, embed_dim) if cls_token=True
    """
    if embed_dim <= 0 or embed_dim % 4 != 0:
        raise ValueError(
            f"embed_dim must be positive and divisible by 4, got {embed_dim}"
        )

    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        grid_h, grid_w = grid_size

    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"grid dimensions must be positive, got ({grid_h}, {grid_w})")

    grid_y = torch.arange(grid_h, dtype=torch.float32)
    grid_x = torch.arange(grid_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid = torch.stack(grid, dim=-1).reshape(-1, 2)

    dim = embed_dim // 4
    omega = torch.arange(dim, dtype=torch.float32) / dim
    omega = 1.0 / (10000**omega)

    out_h = grid[:, 0:1] @ omega.unsqueeze(0)
    out_w = grid[:, 1:2] @ omega.unsqueeze(0)

    pe = torch.cat(
        [torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w)],
        dim=1,
    )

    if cls_token:
        pe = torch.cat([torch.zeros(1, embed_dim), pe], dim=0)
    return pe


def get_sincos_pos_embed(
    embed_dim: int,
    num_patches: int,
    mode: Literal["1d", "2d"] = "1d",
    grid_size: int | tuple[int, int] | None = None,
    cls_token: bool = False,
) -> torch.Tensor:
    """Unified interface for generating sinusoidal positional embeddings.

    :param embed_dim: Embedding dimension
    :param num_patches: Total number of patches (used for 1d mode)
    :param mode: Embedding type - '1d' for sequence, '2d' for image grid
    :param grid_size: Required for '2d' mode
    :param cls_token: If True, prepend a zero embedding for CLS token
    :return: Positional embeddings tensor
    """
    if mode == "2d":
        if grid_size is None:
            raise ValueError("grid_size is required for 2d mode")
        return get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token)
    return get_1d_sincos_pos_embed(embed_dim, num_patches, cls_token)


def interpolate_pos_embed(
    pos_embed: torch.Tensor,
    src_size: tuple[int, int],
    tgt_size: tuple[int, int],
    num_prefix_tokens: int = 0,
    mode: str = "bicubic",
) -> torch.Tensor:
    """Interpolate positional embeddings to a new grid size.

    :param pos_embed: Original positional embeddings of shape
                      (1, num_prefix + src_h*src_w, embed_dim) or
                      (num_prefix + src_h*src_w, embed_dim)
    :param src_size: Source grid size as (height, width)
    :param tgt_size: Target grid size as (height, width)
    :param num_prefix_tokens: Number of prefix tokens (CLS, registers) to preserve
    :param mode: Interpolation mode ('nearest', 'bilinear', 'bicubic', 'area')
    :return: Interpolated positional embeddings

    Example::

        old_pos = model.pos_embed  # (1, 197, 768) = 1 + 14*14
        new_pos = interpolate_pos_embed(
            old_pos, src_size=(14, 14), tgt_size=(16, 16), num_prefix_tokens=1
        )  # (1, 257, 768) = 1 + 16*16
    """
    if pos_embed.dim() not in (2, 3):
        raise ValueError(f"pos_embed must be 2D or 3D, got {pos_embed.dim()}D")

    src_h, src_w = src_size
    tgt_h, tgt_w = tgt_size

    if src_h <= 0 or src_w <= 0 or tgt_h <= 0 or tgt_w <= 0:
        raise ValueError(
            f"All grid dims must be positive, src={src_size}, tgt={tgt_size}"
        )

    squeeze_output = False
    if pos_embed.dim() == 2:
        pos_embed = pos_embed.unsqueeze(0)
        squeeze_output = True

    expected_src_len = num_prefix_tokens + src_h * src_w
    if pos_embed.shape[1] != expected_src_len:
        raise ValueError(
            f"pos_embed length {pos_embed.shape[1]} doesn't match expected {expected_src_len}"
        )

    if src_h == tgt_h and src_w == tgt_w:
        return pos_embed.squeeze(0) if squeeze_output else pos_embed

    prefix_pos = pos_embed[:, :num_prefix_tokens, :]
    patch_pos = pos_embed[:, num_prefix_tokens:, :]

    embed_dim = patch_pos.shape[-1]
    patch_pos = patch_pos.reshape(1, src_h, src_w, embed_dim).permute(0, 3, 1, 2)

    patch_pos = F.interpolate(
        patch_pos,
        size=(tgt_h, tgt_w),
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
    )

    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, tgt_h * tgt_w, embed_dim)
    result = torch.cat([prefix_pos, patch_pos], dim=1)

    return result.squeeze(0) if squeeze_output else result


# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to input tensor.

    :param x: Input tensor [..., seq_len, dim]
    :param cos: Cosine frequencies [seq_len, dim] or [1, 1, seq_len, dim]
    :param sin: Sine frequencies [seq_len, dim] or [1, 1, seq_len, dim]
    :return: Rotated tensor
    """
    # Split into pairs and rotate
    x1, x2 = x[..., ::2], x[..., 1::2]

    # Ensure cos/sin have right shape for broadcasting
    if cos.dim() == 2:
        cos = cos[..., ::2]  # [seq_len, dim//2]
        sin = sin[..., ::2]
    else:
        cos = cos[..., ::2]
        sin = sin[..., ::2]

    # Apply rotation
    out = torch.stack(
        [
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ],
        dim=-1,
    ).flatten(-2)

    return out


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding (RoPE) for vision transformers.

    Encodes relative 2D positions via complex rotations in attention,
    improving generalization across varying image sizes. Uses separate
    frequencies for height and width dimensions.
    :param head_dim: Dimension per attention head
    :param max_grid_size: Maximum grid size for precomputed frequencies
    :param base: Base for frequency computation (default: 10000.0)
    Example::
        rope = RotaryPositionEmbedding2D(head_dim=64, max_grid_size=32)
        # In attention forward:
        q, k = rope(q, k, grid_h=14, grid_w=14)
        # Or get frequencies and apply manually:
        cos, sin = rope.get_freqs(grid_h=14, grid_w=14, device=q.device)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin).
    """

    def __init__(
        self,
        head_dim: int,
        max_grid_size: int = 32,
        base: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_grid_size = max_grid_size
        self.base = base
        # Each 2D axis gets head_dim // 4 frequency pairs
        # Total: (head_dim // 4) * 2 for height + (head_dim // 4) * 2 for width = head_dim
        dim_per_axis = head_dim // 4
        if dim_per_axis <= 0:
            raise ValueError(f"head_dim must be >= 4, got {head_dim}")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim_per_axis, 2).float() / dim_per_axis)
        )
        self.register_buffer("inv_freq", inv_freq)
        # Cache for current grid size
        self._cached_grid_h = 0
        self._cached_grid_w = 0

    def _build_cache(
        self,
        grid_h: int,
        grid_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Build and cache sin/cos frequencies for given grid size."""
        # Height frequencies
        pos_h = torch.arange(grid_h, device=device, dtype=dtype)
        freqs_h = torch.outer(pos_h, self.inv_freq.to(device=device, dtype=dtype))
        # Width frequencies
        pos_w = torch.arange(grid_w, device=device, dtype=dtype)
        freqs_w = torch.outer(pos_w, self.inv_freq.to(device=device, dtype=dtype))
        # Expand to full grid [H, W, dim_per_axis]
        freqs_h = freqs_h.unsqueeze(1).expand(-1, grid_w, -1)  # [H, W, dim//4]
        freqs_w = freqs_w.unsqueeze(0).expand(grid_h, -1, -1)  # [H, W, dim//4]
        # Flatten to [H*W, dim_per_axis] and duplicate for sin/cos pairs
        freqs_h = freqs_h.reshape(-1, freqs_h.shape[-1])  # [H*W, dim//4]
        freqs_w = freqs_w.reshape(-1, freqs_w.shape[-1])  # [H*W, dim//4]
        # Combine: [H*W, head_dim] with interleaved h/w frequencies
        freqs = torch.cat(
            [
                freqs_h,
                freqs_h,  # height (for sin/cos pairs)
                freqs_w,
                freqs_w,  # width (for sin/cos pairs)
            ],
            dim=-1,
        )
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
        self._cached_grid_h = grid_h
        self._cached_grid_w = grid_w

    def get_freqs(
        self,
        grid_h: int,
        grid_w: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin frequencies for given grid size.

        :param grid_h: Grid height
        :param grid_w: Grid width
        :param device: Target device
        :param dtype: Target dtype
        :return: (cos, sin) tensors of shape [H*W, head_dim].
        """
        if grid_h != self._cached_grid_h or grid_w != self._cached_grid_w:
            self._build_cache(grid_h, grid_w, device, dtype)
        seq_len = grid_h * grid_w
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 2D rotary embeddings to query and key tensors.

        :param q: Query tensor [B, num_heads, seq_len, head_dim]
        :param k: Key tensor [B, num_heads, seq_len, head_dim]
        :param grid_h: Patch grid height
        :param grid_w: Patch grid width
        :return: (rotated_q, rotated_k).
        """
        cos, sin = self.get_freqs(grid_h, grid_w, q.device, q.dtype)

        q_rot = apply_rotary_emb(q, cos, sin)
        k_rot = apply_rotary_emb(k, cos, sin)
        return q_rot, k_rot

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, max_grid_size={self.max_grid_size}, base={self.base}"
