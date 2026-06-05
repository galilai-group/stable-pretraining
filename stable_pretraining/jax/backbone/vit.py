"""Native Flax-NNX Vision Transformer backbone (channels-last / NHWC).

A compact ViT mirroring the torch backbone's role: patch-embed, prepend a
learnable CLS token, add learnable positional embeddings, run pre-norm
transformer blocks, and return the final CLS-token embedding ``[B, embed_dim]``.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class _Block(nnx.Module):
    """Pre-norm transformer block: MHSA + MLP with residual connections."""

    def __init__(self, dim, num_heads, mlp_ratio, rngs, dtype=None):
        # LayerNorm kept in float32 (compute dtype unset), mirroring torch AMP.
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=dim, decode=False, dtype=dtype, rngs=rngs
        )
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(dim, dim * mlp_ratio, dtype=dtype, rngs=rngs),
            jax.nn.gelu,
            nnx.Linear(dim * mlp_ratio, dim, dtype=dtype, rngs=rngs),
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nnx.Module):
    """Vision Transformer returning the CLS-token embedding ``[B, embed_dim]``.

    Args:
        rngs: NNX RNG collection.
        img_size: Input image side length (square). Default ``224``.
        patch_size: Patch side length. Default ``16``.
        in_channels: Input channels. Default ``3``.
        embed_dim: Token/embedding dimension. Default ``384`` (ViT-Small).
        depth: Number of transformer blocks. Default ``12``.
        num_heads: Attention heads. Default ``6``.
        mlp_ratio: MLP hidden expansion factor. Default ``4``.

    Attributes:
        embed_dim: Output feature dimension.
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: int = 4,
        dtype=None,
    ):
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size.")
        self.patch_embed = nnx.Conv(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            dtype=dtype,
            rngs=rngs,
        )
        n_patches = (img_size // patch_size) ** 2
        self.cls_token = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, embed_dim)) * 0.02
        )
        self.pos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, n_patches + 1, embed_dim)) * 0.02
        )
        self.blocks = nnx.List(
            [_Block(embed_dim, num_heads, mlp_ratio, rngs, dtype) for _ in range(depth)]
        )
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.embed_dim = embed_dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run the ViT.

        Args:
            x: Input images in NHWC layout, ``[B, H, W, C]``.

        Returns:
            jnp.ndarray: CLS-token embedding ``[B, embed_dim]``.
        """
        b = x.shape[0]
        x = self.patch_embed(x)  # [B, H', W', dim]
        x = x.reshape(b, -1, x.shape[-1])  # [B, n_patches, dim]
        cls = jnp.broadcast_to(self.cls_token[...], (b, 1, x.shape[-1]))
        x = jnp.concatenate([cls, x], axis=1) + self.pos_embed[...]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token


def vit_small(*, rngs: nnx.Rngs, **kwargs) -> ViT:
    """ViT-Small/16: ``embed_dim=384, depth=12, num_heads=6``."""
    kwargs.setdefault("embed_dim", 384)
    kwargs.setdefault("depth", 12)
    kwargs.setdefault("num_heads", 6)
    return ViT(rngs=rngs, **kwargs)


def vit_tiny(*, rngs: nnx.Rngs, **kwargs) -> ViT:
    """ViT-Tiny/16: ``embed_dim=192, depth=12, num_heads=3``."""
    kwargs.setdefault("embed_dim", 192)
    kwargs.setdefault("depth", 12)
    kwargs.setdefault("num_heads", 3)
    return ViT(rngs=rngs, **kwargs)


def vit_base(*, rngs: nnx.Rngs, **kwargs) -> ViT:
    """ViT-Base/16: ``embed_dim=768, depth=12, num_heads=12``."""
    kwargs.setdefault("embed_dim", 768)
    kwargs.setdefault("depth", 12)
    kwargs.setdefault("num_heads", 12)
    return ViT(rngs=rngs, **kwargs)


def vit_large(*, rngs: nnx.Rngs, **kwargs) -> ViT:
    """ViT-Large/16: ``embed_dim=1024, depth=24, num_heads=16``."""
    kwargs.setdefault("embed_dim", 1024)
    kwargs.setdefault("depth", 24)
    kwargs.setdefault("num_heads", 16)
    return ViT(rngs=rngs, **kwargs)


__all__ = ["ViT", "vit_small", "vit_tiny", "vit_base", "vit_large"]
