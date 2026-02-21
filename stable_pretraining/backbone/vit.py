import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal, Union
import timm
from timm.layers import DropPath, Mlp, trunc_normal_
from .patch_masking import PatchMasking
from dataclasses import dataclass
from .pos_embed import (
    get_sincos_pos_embed,
    get_timestep_embed,
    interpolate_pos_embed,
    RotaryPositionEmbedding2D,
)


class SwiGLU(nn.Module):
    """SwiGLU: Gated Linear Unit with Swish activation.

    A parameter-efficient gated activation that combines the benefits of
    gating mechanisms with the smooth, non-monotonic Swish activation.
    Empirically improves transformer performance over standard GeLU MLPs.
    Architecture
    ------------
    Standard MLP::
        x → Linear → GeLU → Linear → out
        Parameters: 2 * d * h
    SwiGLU::
        x → Linear(W₁) → SiLU ─┐
                               ├─ element-wise multiply → Linear(W₃) → out
        x → Linear(W₂) ────────┘
        Parameters: 3 * d * h'  (where h' = 2h/3 to match param count)
    The hidden dimension is scaled to ``2/3 * hidden_features`` so that
    total parameter count matches a standard 2-layer MLP:
    ``3 * d * (2h/3) = 2 * d * h``
    Performance Benefits
    --------------------
    - **Better gradient flow**: Gating provides multiplicative paths
    - **Smoother optimization**: SiLU (Swish) is smooth and non-monotonic
    - **Quality**: Consistently outperforms GeLU in language and vision models
    :param in_features: Input dimension
    :param hidden_features: Nominal hidden dimension. Actual hidden size is
        ``int(2 * hidden_features / 3)`` to maintain parameter parity with
        standard MLPs.
    :param out_features: Output dimension. Defaults to ``in_features``.
    :param bias: If True, use bias in linear layers. Default False following
        LLaMA/PaLM convention for better training stability.
    :param drop: Dropout probability applied after gating.
    Example::
        # Replace standard MLP in transformer
        # Old: mlp = Mlp(768, 3072, 768)
        # New:
        mlp = SwiGLU(768, 3072, 768)
        x = torch.randn(4, 196, 768)
        out = mlp(x)  # [4, 196, 768]
        # Parameter count comparison
        standard_mlp = nn.Sequential(
            nn.Linear(768, 3072), nn.GELU(), nn.Linear(3072, 768)
        )
        swiglu = SwiGLU(768, 3072, 768)
        print(sum(p.numel() for p in standard_mlp.parameters()))  # 4,722,432
        print(sum(p.numel() for p in swiglu.parameters()))  # 4,722,432 (same!)

    Note:
        For best results, combine SwiGLU with:
        - **LayerScale**: Stabilizes residual connections
        - **QK-Norm**: Prevents attention explosion
        - **RoPE**: Better positional generalization
        This combination is used in LLaMA, PaLM, and modern vision transformers.

    References:
        - Shazeer, "GLU Variants Improve Transformer" (2020)
        - Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = False,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        # Scale hidden dim to maintain param parity with standard MLP
        hidden_features = int(2 * hidden_features / 3)
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation."""
        # SwiGLU: (SiLU(xW₁) ⊙ xW₂) W₃
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.w1.in_features}, "
            f"hidden_features={self.w1.out_features}, "
            f"out_features={self.w3.out_features}"
        )


class QKNorm(nn.Module):
    """Query-Key Normalization for attention stabilization.

    Applies LayerNorm (without learnable parameters) independently to query
    and key tensors before computing attention scores. This simple technique
    dramatically improves training stability in deep transformers.
    Why QK-Norm Works
    -----------------
    In deep transformers, attention logits (Q·Kᵀ) can grow unboundedly large,
    causing:
    - **Gradient explosion**: Large logits → extreme softmax → tiny gradients
    - **Attention collapse**: All mass on single token
    - **Training instability**: Requires very small learning rates
    QK-Norm bounds the attention logits by normalizing Q and K to unit variance:
    - Attention logits become bounded: ``|q·k| ≤ ||q|| ||k|| = O(√d)``
    - Gradients remain stable throughout training
    - Enables larger learning rates and faster convergence
    Implementation Details
    ----------------------
    - Uses **LayerNorm without learnable parameters** (γ=1, β=0)
    - Normalizes per-head: applied to ``[..., head_dim]`` dimension
    - Zero computational overhead in modern frameworks (fused with attention)
    :param head_dim: Dimension per attention head. Each head is normalized
        independently to preserve multi-head diversity.
    Example::
        # In attention forward pass
        qk_norm = QKNorm(head_dim=64)
        # q, k shape: [B, num_heads, seq_len, head_dim]
        q, k = qk_norm(q, k)
        # Now safe to compute attention
        attn = (q @ k.transpose(-2, -1)) * scale
    Example integration with Attention::
        class Attention(nn.Module):
            def __init__(self, dim, num_heads, use_qk_norm=True):
                ...
                if use_qk_norm:
                    self.qk_norm = QKNorm(dim // num_heads)

            def forward(self, x):
                q, k, v = self.qkv(x).chunk(3, dim=-1)
                if self.use_qk_norm:
                    q, k = self.qk_norm(q, k)
                ...

    Note:
        QK-Norm is especially important when combined with:
        - **SwiGLU**: Gated activations can amplify hidden states
        - **LayerScale**: Small initial residual scale needs stable attention
        - **Deep networks**: Logit growth compounds with depth
        Without QK-Norm, these combinations often fail to train or require
        extensive hyperparameter tuning.

    References:
        - Henry et al., "Query-Key Normalization for Transformers" (EMNLP 2020)
        - Dehghani et al., "Scaling Vision Transformers to 22B Parameters" (2023)
        - Wortsman et al., "Small-scale proxies for large-scale Transformer training" (2023).
    """

    def __init__(self, head_dim: int):
        super().__init__()
        # LayerNorm without learnable parameters (elementwise_affine=False)
        self.q_norm = nn.LayerNorm(head_dim, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(head_dim, elementwise_affine=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize query and key tensors independently.

        :param q: Query tensor of shape ``[B, num_heads, seq_len, head_dim]``
            or any shape with last dimension = head_dim
        :param k: Key tensor of same shape as q
        :return: Tuple of (normalized_q, normalized_k) with same shapes
        Note:
            Normalization is applied to the last dimension (head_dim).
            Each head is normalized independently, preserving multi-head
            representation diversity.
        """
        return self.q_norm(q), self.k_norm(k)

    def extra_repr(self) -> str:
        return f"head_dim={self.q_norm.normalized_shape[0]}"


@dataclass
class MaskedEncoderOutput:
    """Output from MaskedEncoder forward pass.

    :ivar encoded: Encoded token representations (B, num_prefix + N_visible, D)
    :ivar mask: Binary mask where 1 = masked, 0 = visible (B, N_patches)
    :ivar ids_keep: Indices of visible patches (B, N_visible)
    :ivar grid_size: Patch grid dimensions (height, width)
    """

    encoded: torch.Tensor
    mask: torch.Tensor
    ids_keep: torch.Tensor
    grid_size: Tuple[int, int]


class MaskedEncoder(nn.Module):
    """Vision Transformer encoder with optional masking support.

    Wraps a timm ViT model and adds flexible masking via :class:`PatchMasking`.
    Handles all ViT internals: patch embedding, positional embeddings, prefix
    tokens (CLS, registers), and transformer blocks.
    :param model_or_model_name: timm model name string or pre-instantiated nn.Module
    :param masking: PatchMasking instance. If None, no masking is applied.
    :param pretrained: Load pretrained weights (only when model_or_model_name is str)
    :param img_size: Override default image size
    :param patch_size: Override default patch size (will reinitialize patch_embed)
    :param dynamic_img_size: Enable dynamic image size support with pos_embed interpolation
    Example::
        from spt.backbone import PatchMasking, MaskedEncoder

        masking = PatchMasking(mask_ratio=0.75, block_size=4)
        encoder = MaskedEncoder(
            model_or_model_name="vit_base_patch16_224",
            masking=masking,
            pretrained=True,
        )
        images = torch.randn(4, 3, 224, 224)
        output = encoder(images)
        print(output.encoded.shape)  # (4, 1 + 49, 768) with 75% masking
        print(output.mask.shape)  # (4, 196)
        print(output.ids_keep.shape)  # (4, 49)
    """

    def __init__(
        self,
        model_or_model_name: Union[str, nn.Module] = "vit_base_patch16_224",
        masking: Optional[PatchMasking] = None,
        pretrained: bool = False,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        dynamic_img_size: bool = False,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.dynamic_img_size = dynamic_img_size
        self.masking = masking
        # === Load or use provided encoder ===
        if isinstance(model_or_model_name, str):
            create_kwargs = {
                "pretrained": pretrained,
                "num_classes": 0,
                "dynamic_img_size": dynamic_img_size,
            }
            if img_size is not None:
                create_kwargs["img_size"] = img_size
            if patch_size is not None:
                create_kwargs["patch_size"] = patch_size
                if pretrained:
                    print(
                        f"Warning: Changing patch_size to {patch_size} will reinitialize "
                        f"patch_embed weights. Pretrained weights won't fully apply."
                    )
            if norm_layer is not None:
                create_kwargs["norm_layer"] = norm_layer

            self.vit = timm.create_model(model_or_model_name, **create_kwargs)
        else:
            self.vit = model_or_model_name
            if patch_size is not None:
                self._rebuild_patch_embed(patch_size, img_size)
            # Remove classification head if present
            if hasattr(self.vit, "head") and hasattr(self.vit.head, "in_features"):
                self.vit.head = nn.Identity()
        # === Cache encoder properties ===
        self.embed_dim = self.vit.embed_dim
        self.patch_embed = self.vit.patch_embed
        ps = self.patch_embed.patch_size
        self.patch_size_h, self.patch_size_w = (ps, ps) if isinstance(ps, int) else ps
        gs = self.patch_embed.grid_size
        self.default_grid_h, self.default_grid_w = (
            (gs, gs) if isinstance(gs, int) else gs
        )
        self.num_prefix_tokens = getattr(self.vit, "num_prefix_tokens", 1)
        self.has_class_token = getattr(self.vit, "has_class_token", True)
        self.num_reg_tokens = getattr(self.vit, "num_reg_tokens", 0)
        self.no_embed_class = getattr(self.vit, "no_embed_class", False)

    def _rebuild_patch_embed(
        self,
        patch_size: Union[int, Tuple[int, int]],
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> None:
        """Rebuild patch embedding with new patch size."""
        from timm.layers import PatchEmbed

        old = self.vit.patch_embed
        if img_size is None:
            og, op = old.grid_size, old.patch_size
            img_size = (
                (og[0] * op[0], og[1] * op[1]) if isinstance(og, tuple) else og * op
            )
        self.vit.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=old.proj.in_channels,
            embed_dim=old.proj.out_channels,
        )
        if old.num_patches != self.vit.patch_embed.num_patches:
            self._resize_pos_embed(self.vit.patch_embed.grid_size)

    def _resize_pos_embed(self, new_grid_size: Tuple[int, int]) -> None:
        """Resize positional embeddings to new grid size."""
        old_pos = self.vit.pos_embed
        num_prefix = self.num_prefix_tokens if not self.no_embed_class else 0
        src_patches = old_pos.shape[1] - num_prefix
        src_size = int(src_patches**0.5)
        new_pos = interpolate_pos_embed(
            old_pos, (src_size, src_size), new_grid_size, num_prefix
        )
        self.vit.pos_embed = nn.Parameter(new_pos)

    def _get_grid_size(self, images: torch.Tensor) -> Tuple[int, int]:
        """Compute patch grid size from image dimensions."""
        H, W = images.shape[-2:]
        return H // self.patch_size_h, W // self.patch_size_w

    def _get_pos_embed(
        self, grid_h: int, grid_w: int
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Get positional embeddings, interpolating if needed for dynamic size."""
        pos_embed = self.vit.pos_embed
        num_prefix = self.num_prefix_tokens if not self.no_embed_class else 0
        if self.dynamic_img_size and (
            grid_h != self.default_grid_h or grid_w != self.default_grid_w
        ):
            src_patches = pos_embed.shape[1] - num_prefix
            src_size = int(src_patches**0.5)
            pos_embed = interpolate_pos_embed(
                pos_embed, (src_size, src_size), (grid_h, grid_w), num_prefix
            )
        if self.no_embed_class:
            return None, pos_embed
        return (
            pos_embed[:, : self.num_prefix_tokens],
            pos_embed[:, self.num_prefix_tokens :],
        )

    def _get_prefix_tokens(self, B: int) -> Optional[torch.Tensor]:
        """Get CLS and register tokens expanded to batch size."""
        tokens = []
        if self.has_class_token:
            tokens.append(self.vit.cls_token.expand(B, -1, -1))
        if self.num_reg_tokens > 0:
            tokens.append(self.vit.reg_token.expand(B, -1, -1))
        return torch.cat(tokens, dim=1) if tokens else None

    def forward(
        self, images: torch.Tensor, apply_mask: bool | None = None
    ) -> MaskedEncoderOutput:
        """Encode images with optional masking.

        :param images: Input images (B, C, H, W)
        :return: MaskedEncoderOutput with encoded tokens and mask info
        """
        B = images.shape[0]
        device = images.device

        grid_h, grid_w = self._get_grid_size(images)
        num_patches = grid_h * grid_w

        # Patch embed + positional embed
        x = self.patch_embed(images)
        prefix_pos, patch_pos = self._get_pos_embed(grid_h, grid_w)
        x = x + patch_pos

        if apply_mask is None:
            apply_mask = self.training

        # Apply masking (training only)
        if apply_mask and self.masking is not None:
            mask_out = self.masking(x, grid_h, grid_w)
            x = mask_out.visible
            mask = mask_out.mask
            ids_keep = mask_out.ids_keep
        else:
            mask = torch.zeros(B, num_patches, device=device)
            ids_keep = (
                torch.arange(num_patches, device=device).unsqueeze(0).expand(B, -1)
            )
        # Prepend prefix tokens
        prefix = self._get_prefix_tokens(B)
        if prefix is not None:
            if prefix_pos is not None and not self.no_embed_class:
                prefix = prefix + prefix_pos
            x = torch.cat([prefix, x], dim=1)
        # Transformer blocks
        x = self.vit.pos_drop(x)
        x = self.vit.blocks(x) if hasattr(self.vit, "blocks") else self.vit.layers(x)
        x = self.vit.norm(x)
        return MaskedEncoderOutput(
            encoded=x,
            mask=mask,
            ids_keep=ids_keep,
            grid_size=(grid_h, grid_w),
        )

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Encode without masking (for inference)."""
        with torch.no_grad():
            output = self.forward(images, apply_mask=False)
        return output.encoded

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"patch_size=({self.patch_size_h}, {self.patch_size_w}), "
            f"num_prefix_tokens={self.num_prefix_tokens}, "
            f"has_masking={self.masking is not None}"
        )


class Attention(nn.Module):
    """Multi-head self-attention with efficient SDPA backend.

    Supports modern transformer features including Rotary Position Embeddings
    (RoPE) and Query-Key Normalization (QK-Norm) for improved training stability
    and positional generalization.

    Uses ``F.scaled_dot_product_attention`` which automatically selects the
    optimal backend:

    - **Flash Attention** (when available, fastest)
    - **Memory-efficient attention** (xformers-style)
    - **Math fallback**

    Architecture Features
    ---------------------
    **RoPE (Rotary Position Embedding)**:
        Encodes relative 2D positions via complex rotations applied to Q and K.
        Unlike additive positional embeddings, RoPE:

        - Naturally captures relative positions
        - Generalizes to unseen sequence lengths
        - Requires no extra parameters

        Enable with ``use_rope=True``. Requires ``grid_size`` in forward().

    **QK-Norm (Query-Key Normalization)**:
        Applies LayerNorm (without learnable params) to Q and K before
        computing attention scores. Benefits:

        - Prevents attention logit explosion in deep networks
        - Stabilizes training without extra hyperparameter tuning
        - Essential when combined with SwiGLU/LayerScale

        Enable with ``use_qk_norm=True``.

    Attention Masking
    -----------------
    Supports flexible attention patterns via ``attn_mask``:

    - **Causal (autoregressive)**: ``torch.triu(ones, diagonal=1)``
    - **Bidirectional**: ``attn_mask=None``
    - **Block sparse**: Custom boolean masks
    - **Leave-one-out**: ``torch.eye(N)`` (each token ignores itself)

    Mask convention: ``True`` = blocked (cannot attend), ``False`` = allowed.

    :param dim: Input/output embedding dimension
    :param num_heads: Number of parallel attention heads. Must divide ``dim``.
    :param qkv_bias: If True, add learnable bias to Q, K, V projections.
        Default True following ViT convention.
    :param attn_drop: Dropout probability on attention weights. Applied only
        during training.
    :param proj_drop: Dropout probability on output projection.
    :param use_rope: Enable 2D Rotary Position Embedding. When True, position
        information is encoded via rotation in attention rather than additive
        embeddings. Requires ``grid_size`` parameter in forward().
    :param use_qk_norm: Enable Query-Key normalization. Applies LayerNorm
        (without learnable parameters) to Q and K tensors before attention.
        Recommended for deep networks or when using SwiGLU.
    :param max_grid_size: Maximum spatial grid size for RoPE frequency cache.
        Only used when ``use_rope=True``. Set to largest expected grid dimension.

    Example::

        # Standard attention
        attn = Attention(dim=768, num_heads=12)
        out = attn(x)  # [B, N, 768]

        # With RoPE for vision (requires grid_size)
        attn = Attention(dim=768, num_heads=12, use_rope=True)
        out = attn(x, grid_size=(14, 14))

        # With QK-Norm for training stability
        attn = Attention(dim=768, num_heads=12, use_qk_norm=True)
        out = attn(x)

        # Causal attention (autoregressive)
        N = x.shape[1]
        causal_mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        out = attn(x, attn_mask=causal_mask)

        # NEPA-style: RoPE + QK-Norm + causal
        attn = Attention(dim=768, num_heads=12, use_rope=True, use_qk_norm=True)
        causal_mask = torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=x.device), diagonal=1
        )
        out = attn(x, attn_mask=causal_mask, grid_size=(14, 14))

    Note:
        When ``use_rope=True``, do NOT add positional embeddings to input tokens.
        RoPE encodes positions internally via Q/K rotation.

    References:
        - RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary
          Position Embedding" (2021)
        - QK-Norm: Henry et al., "Query-Key Normalization for Transformers" (2020)
        - Flash Attention: Dao et al., "FlashAttention: Fast and Memory-Efficient
          Exact Attention" (2022)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rope: bool = False,
        use_qk_norm: bool = False,
        max_grid_size: int = 32,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if use_rope:
            self.rope = RotaryPositionEmbedding2D(self.head_dim, max_grid_size)

        if use_qk_norm:
            self.qk_norm = QKNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Compute multi-head self-attention.

        :param x: Input tensor of shape ``[B, N, D]`` where B is batch size,
            N is sequence length, and D is embedding dimension.
        :param attn_mask: Optional attention mask. Supported shapes:

            - ``[N, N]``: Same mask for all batches and heads
            - ``[B, N, N]``: Per-batch mask, broadcast over heads
            - ``[B, H, N, N]``: Full per-batch, per-head mask

            Values: ``True`` = blocked (cannot attend), ``False`` = allowed.

            Common patterns:

            - Causal: ``torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)``
            - Leave-one-out: ``torch.eye(N, dtype=torch.bool)``

        :param grid_size: Spatial grid dimensions as ``(height, width)``.
            **Required when** ``use_rope=True``. Used to compute 2D rotary
            position embeddings. For a 224x224 image with patch_size=16,
            use ``grid_size=(14, 14)``.

        :return: Output tensor of shape ``[B, N, D]``

        :raises ValueError: If ``use_rope=True`` but ``grid_size`` is None
        """
        B, N, C = x.shape

        # Fused QKV projection
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # Each: [B, num_heads, N, head_dim]

        # Apply QK-Norm before RoPE (order matters for stability)
        if self.use_qk_norm:
            q, k = self.qk_norm(q, k)

        # Apply 2D Rotary Position Embedding
        if self.use_rope:
            if grid_size is None:
                # Attempt to infer square grid
                grid_h = grid_w = int(N**0.5)
                if grid_h * grid_w != N:
                    raise ValueError(
                        f"use_rope=True requires grid_size parameter for non-square "
                        f"sequences. Got N={N} which is not a perfect square."
                    )
            else:
                grid_h, grid_w = grid_size
            q, k = self.rope(q, k, grid_h, grid_w)

        # Convert boolean mask to SDPA-compatible float mask
        attn_mask_sdpa = self._prepare_attn_mask(attn_mask, q.dtype, q.device)

        # Efficient attention via SDPA
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask_sdpa,
            dropout_p=self.attn_drop if self.training else 0.0,
        )

        # Reshape and project output
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def _prepare_attn_mask(
        self,
        attn_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Convert boolean attention mask to SDPA-compatible float mask.

        SDPA expects additive mask where ``-inf`` blocks attention.
        Our convention: ``True`` = blocked, ``False`` = allowed.
        """
        if attn_mask is None:
            return None

        if attn_mask.dtype == torch.bool:
            mask = torch.zeros_like(attn_mask, dtype=dtype, device=device)
            mask = mask.masked_fill(attn_mask, float("-inf"))
        else:
            mask = attn_mask.to(dtype=dtype, device=device)

        # Expand to [B, H, N, N] for SDPA broadcasting
        while mask.dim() < 4:
            mask = mask.unsqueeze(0 if mask.dim() == 2 else 1)

        return mask

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"use_rope={self.use_rope}, use_qk_norm={self.use_qk_norm}"
        )


class CrossAttention(nn.Module):
    """Multi-head cross-attention with efficient SDPA backend.

    Queries attend to key-value pairs from a separate context sequence.
    Supports attention masking to block specific query-key interactions.

    :param dim: Query dimension
    :param context_dim: Context dimension (defaults to dim)
    :param num_heads: Number of attention heads
    :param qkv_bias: Add bias to projections
    :param attn_drop: Attention dropout rate
    :param proj_drop: Output projection dropout rate

    Example::

        cross_attn = CrossAttention(dim=768, context_dim=1024, num_heads=12)

        # Standard cross-attention
        out = cross_attn(queries, context)  # [B, N, 768]

        # Masked cross-attention: block certain query-context pairs
        # mask[i, j] = True means query i cannot attend to context j
        mask = torch.zeros(N, M, dtype=torch.bool)
        mask[:, :10] = True  # block attention to first 10 context tokens
        out = cross_attn(queries, context, attn_mask=mask)
    """

    def __init__(
        self,
        dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        context_dim = context_dim or dim
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        :param x: Query tensor [B, N, D]
        :param context: Key-value tensor [B, M, context_dim]
        :param attn_mask: Cross-attention mask. Can be one of:
            - [N, M]: Same mask for all batches and heads
            - [B, N, M]: Per-batch mask, broadcast over heads
            - [B, H, N, M]: Full per-batch, per-head mask
            Mask values: True = blocked (cannot attend), False = allowed.
        :return: Output tensor [B, N, D]
        """
        B, N, C = x.shape
        M = context.shape[1]

        # Query projection: [B, N, D] -> [B, H, N, head_dim]
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # KV projection: [B, M, D] -> [B, H, M, head_dim] x2
        kv = (
            self.kv(context)
            .reshape(B, M, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)

        # Convert mask to SDPA format if provided
        attn_mask_sdpa = self._prepare_attn_mask(attn_mask, q.dtype, q.device)

        # Efficient attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask_sdpa,
            dropout_p=self.attn_drop if self.training else 0.0,
        )

        # Reshape back
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _prepare_attn_mask(
        self,
        attn_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Convert boolean attention mask to SDPA-compatible float mask.

        SDPA expects additive mask where -inf blocks attention.
        Our convention: True = blocked, False = allowed.

        :param attn_mask: Boolean mask [N, M], [B, N, M], or [B, H, N, M]
        :param dtype: Target dtype for the mask
        :param device: Target device for the mask
        :return: Float mask suitable for SDPA, or None
        """
        if attn_mask is None:
            return None

        # Convert bool mask to float: True -> -inf, False -> 0
        if attn_mask.dtype == torch.bool:
            mask = torch.zeros_like(attn_mask, dtype=dtype, device=device)
            mask = mask.masked_fill(attn_mask, float("-inf"))
        else:
            mask = attn_mask.to(dtype=dtype, device=device)

        # Expand dimensions for broadcasting: need [B, H, N, M] for SDPA
        # [N, M] -> [1, 1, N, M]
        # [B, N, M] -> [B, 1, N, M]
        # [B, H, N, M] -> unchanged
        while mask.dim() < 4:
            mask = mask.unsqueeze(0 if mask.dim() == 2 else 1)

        return mask


# =============================================================================
# Transformer Block
# =============================================================================
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale) + shift


class TransformerBlock(nn.Module):
    """Unified transformer block supporting multiple architectures.

    Configurable for various modern transformer designs:

    +------------------+----------+----------+--------+------------+----------+
    | Architecture     | RoPE     | QK-Norm  | MLP    | LayerScale | AdaLN    |
    +==================+==========+==========+========+============+==========+
    | Standard ViT     | ✗        | ✗        | gelu   | ✗          | ✗        |
    | DINOv2 / Modern  | ✓        | ✓        | swiglu | ✓          | ✗        |
    | NEPA             | ✓        | ✓        | swiglu | ✓          | ✗        |
    | DiT / Flow       | ✗        | ✗        | gelu   | ✗          | ✓        |
    +------------------+----------+----------+--------+------------+----------+

    Attention Modes
    ---------------
    **Mode 1: Self-Attention Only** (``self_attn=True, cross_attn=False``)
        Standard encoder block. Used for NEPA, ViT encoder, etc.

    **Mode 2: Cross-Attention Only** (``self_attn=False, cross_attn=True``)
        Queries attend to context only. Lightweight decoder.

    **Mode 3: Full Decoder** (``self_attn=True, cross_attn=True``)
        Self-attention on queries, then cross-attention to context.

    Modern Components
    -----------------
    **RoPE** (``use_rope=True``):
        2D Rotary Position Embedding. Encodes positions via Q/K rotation.
        Requires ``grid_size`` in forward(). Don't use additive pos_embed.

    **QK-Norm** (``use_qk_norm=True``):
        Normalizes Q and K before attention. Stabilizes deep networks.

    **SwiGLU** (``mlp_type='swiglu'``):
        Gated MLP with SiLU activation. Better than GeLU empirically.

    **LayerScale** (``use_layer_scale=True``):
        Learnable per-channel scaling on residuals. Stabilizes training.
        Initialize near zero (e.g., 1e-5) for identity-like initialization.

    :param dim: Hidden dimension
    :param num_heads: Number of attention heads
    :param mlp_ratio: MLP hidden dim = dim * mlp_ratio
    :param self_attn: Enable self-attention
    :param cross_attn: Enable cross-attention
    :param use_adaln: Enable AdaLN-Zero conditioning (for diffusion/flow)
    :param use_rope: Enable 2D Rotary Position Embedding in attention
    :param use_qk_norm: Enable Query-Key normalization in attention
    :param mlp_type: MLP activation type: 'gelu' or 'swiglu'
    :param use_layer_scale: Enable LayerScale on residual connections
    :param layer_scale_init: Initial value for LayerScale (default: 1e-5)
    :param drop_path: Stochastic depth rate
    :param attn_drop: Attention dropout rate
    :param proj_drop: Projection dropout rate
    :param max_grid_size: Maximum grid size for RoPE cache

    Example::

        # Standard ViT block
        block = TransformerBlock(dim=768, num_heads=12)

        # Modern ViT (DINOv2-style)
        block = TransformerBlock(
            dim=768,
            num_heads=12,
            use_rope=True,
            use_qk_norm=True,
            mlp_type="swiglu",
            use_layer_scale=True,
        )
        out = block(x, grid_size=(14, 14))

        # NEPA block (modern + causal)
        causal_mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        out = block(x, grid_size=(14, 14), attn_mask=causal_mask)

        # DiT block (with conditioning)
        block = TransformerBlock(dim=768, num_heads=12, use_adaln=True)
        out = block(x, cond=time_emb)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        self_attn: bool = True,
        cross_attn: bool = False,
        use_adaln: bool = False,
        use_rope: bool = False,
        use_qk_norm: bool = False,
        mlp_type: Literal["gelu", "swiglu"] = "gelu",
        use_layer_scale: bool = False,
        layer_scale_init: float = 1e-5,
        drop_path: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        max_grid_size: int = 32,
        act_layer: type = nn.GELU,
    ):
        super().__init__()
        self.use_self_attn = self_attn
        self.use_cross_attn = cross_attn
        self.use_adaln = use_adaln
        self.use_rope = use_rope
        self.use_layer_scale = use_layer_scale

        if not self_attn and not cross_attn:
            raise ValueError("At least one of self_attn or cross_attn must be True")

        # Self-attention
        if self_attn:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=not use_adaln)
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                use_rope=use_rope,
                use_qk_norm=use_qk_norm,
                max_grid_size=max_grid_size,
            )
            self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            if use_layer_scale:
                self.ls1 = nn.Parameter(layer_scale_init * torch.ones(dim))

        # Cross-attention
        if cross_attn:
            self.norm2_q = nn.LayerNorm(dim, elementwise_affine=not use_adaln)
            self.norm2_kv = nn.LayerNorm(dim, elementwise_affine=not use_adaln)
            self.cross_attn = CrossAttention(
                dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            if use_layer_scale:
                self.ls2 = nn.Parameter(layer_scale_init * torch.ones(dim))

        # MLP
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=not use_adaln)
        mlp_hidden = int(dim * mlp_ratio)
        if mlp_type == "swiglu":
            self.mlp = SwiGLU(dim, mlp_hidden, dim, drop=proj_drop)
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden,
                act_layer=act_layer,
                drop=proj_drop,
            )
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if use_layer_scale:
            self.ls3 = nn.Parameter(layer_scale_init * torch.ones(dim))

        # AdaLN modulation
        if use_adaln:
            num_ops = int(self_attn) + int(cross_attn) + 1
            self.num_mods = num_ops * 3
            self.adaLN_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, self.num_mods * dim),
            )
            nn.init.zeros_(self.adaLN_mlp[1].weight)
            nn.init.zeros_(self.adaLN_mlp[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Forward pass.

        :param x: Input tensor [B, N, D]
        :param context: Context for cross-attention [B, M, D]
        :param cond: Conditioning tensor [B, D] (required if use_adaln=True)
        :param attn_mask: Self-attention mask. True = blocked.
        :param cross_attn_mask: Cross-attention mask. True = blocked.
        :param grid_size: (grid_h, grid_w) for RoPE. Required if use_rope=True.
        :return: Output tensor [B, N, D]
        """
        if self.use_cross_attn and context is None:
            raise ValueError("context required when cross_attn=True")
        if self.use_adaln and cond is None:
            raise ValueError("cond required when use_adaln=True")
        if self.use_rope and grid_size is None:
            raise ValueError("grid_size required when use_rope=True")

        if self.use_adaln:
            mods = self.adaLN_mlp(cond).chunk(self.num_mods, dim=-1)
            mods = [m.unsqueeze(1) for m in mods]
            i = 0

            if self.use_self_attn:
                shift, scale, gate = mods[i], mods[i + 1], mods[i + 2]
                i += 3
                x = x + gate * self.drop_path1(
                    self.attn(
                        modulate(self.norm1(x), shift, scale),
                        attn_mask=attn_mask,
                        grid_size=grid_size,
                    )
                )

            if self.use_cross_attn:
                shift, scale, gate = mods[i], mods[i + 1], mods[i + 2]
                i += 3
                x = x + gate * self.drop_path2(
                    self.cross_attn(
                        modulate(self.norm2_q(x), shift, scale),
                        self.norm2_kv(context),
                        attn_mask=cross_attn_mask,
                    )
                )

            shift, scale, gate = mods[i], mods[i + 1], mods[i + 2]
            x = x + gate * self.drop_path3(
                self.mlp(modulate(self.norm3(x), shift, scale))
            )
        else:
            # Standard forward (with optional LayerScale)
            if self.use_self_attn:
                attn_out = self.attn(
                    self.norm1(x),
                    attn_mask=attn_mask,
                    grid_size=grid_size,
                )
                if self.use_layer_scale:
                    attn_out = self.ls1 * attn_out
                x = x + self.drop_path1(attn_out)

            if self.use_cross_attn:
                cross_out = self.cross_attn(
                    self.norm2_q(x),
                    self.norm2_kv(context),
                    attn_mask=cross_attn_mask,
                )
                if self.use_layer_scale:
                    cross_out = self.ls2 * cross_out
                x = x + self.drop_path2(cross_out)

            mlp_out = self.mlp(self.norm3(x))
            if self.use_layer_scale:
                mlp_out = self.ls3 * mlp_out
            x = x + self.drop_path3(mlp_out)

        return x


class FlexibleTransformer(nn.Module):
    """Flexible transformer supporting multiple architectures.

    Unified backbone for:
    - **MAE decoder**: `self_attn=True, cross_attn=False, use_adaln=False`
    - **IJEPA predictor**: `self_attn=True, cross_attn=True, use_adaln=False`
    - **DiT / Flow**: `self_attn=True, cross_attn=True/False, use_adaln=True`
    - **MaskGIT**: `self_attn=True, cross_attn=False, use_adaln=True, add_mask_token=True`
    - **Lightweight predictor**: `self_attn=True, cross_attn=False, use_adaln=False, num_registers>0`
    - **Leave-one-out prediction**: `self_attn=True, cross_attn=False` with diagonal `attn_mask`

    :param input_dim: Input embedding dimension (from encoder)
    :param hidden_dim: Internal transformer dimension
    :param output_dim: Output dimension
    :param num_patches: Total number of patches (for positional embeddings)
    :param depth: Number of transformer blocks
    :param num_heads: Number of attention heads
    :param mlp_ratio: MLP hidden dim multiplier
    :param self_attn: Enable self-attention in blocks
    :param cross_attn: Enable cross-attention in blocks
    :param use_adaln: Enable AdaLN-Zero conditioning
    :param pos_embed_type: 'sincos_1d', 'sincos_2d', or 'learned'
    :param grid_size: Grid size for 2D positional embeddings
    :param drop_path_rate: Stochastic depth rate (linearly increases through layers)
    :param attn_drop: Attention dropout rate
    :param proj_drop: Projection dropout rate
    :param zero_init_output: Zero-initialize output projection
    :param num_prefix_tokens: Number of prefix tokens (e.g., CLS token) expected in input.
        These are tokens whose content comes from the encoder but need special
        positional embeddings.
    :param num_registers: Number of learnable register tokens to prepend internally.
        Unlike prefix tokens, registers are fully learnable (both content and position)
        and are prepended automatically—callers don't include them in input.
    :param add_mask_token: Enable learnable [MASK] token for masked prediction.
        When enabled, use `context_mask` and/or `query_mask` in forward() to
        replace tokens at specified positions with the [MASK] token.

    Example::

        # MAE decoder
        decoder = FlexibleTransformer(
            768,
            512,
            768,
            196,
            depth=8,
            self_attn=True,
            cross_attn=False,
            use_adaln=False,
        )
        out = decoder(context, queries, context_idx, query_idx)

        # IJEPA predictor
        predictor = FlexibleTransformer(
            768,
            384,
            768,
            196,
            depth=6,
            self_attn=True,
            cross_attn=False,
            add_mask_token=True,
            use_adaln=False,
        )
        out = predictor(context, queries, context_idx, query_idx)

        # DiT-style flow matching
        flow = FlexibleTransformer(
            768,
            384,
            768,
            196,
            depth=12,
            self_attn=True,
            cross_attn=False,
            use_adaln=True,
        )
        out = flow(context, queries, context_idx, query_idx, t=timesteps)

        # MaskGIT-style: variable number of masks per sample
        maskgit = FlexibleTransformer(
            768,
            512,
            768,
            196,
            depth=8,
            self_attn=True,
            cross_attn=False,
            use_adaln=True,
            add_mask_token=True,
        )
        context_mask = torch.rand(B, num_patches) < mask_ratio
        out = maskgit(
            context=all_patches,
            queries=all_patches[:, :0],
            context_idx=torch.arange(196).expand(B, -1),
            query_idx=torch.empty(B, 0, dtype=torch.long),
            context_mask=context_mask,
            t=timesteps,
            return_all=True,
        )

        # Leave-one-out prediction: each token predicted from all others
        predictor = FlexibleTransformer(
            768,
            384,
            768,
            196,
            depth=4,
            self_attn=True,
            cross_attn=False,
            use_adaln=False,
        )
        # Diagonal mask: each token cannot attend to itself
        T = x.shape[1]
        attn_mask = torch.eye(T, dtype=torch.bool, device=x.device)
        out = predictor(
            context=x,
            queries=x[:, :0],  # empty queries
            context_idx=torch.arange(T).expand(B, -1),
            query_idx=torch.empty(B, 0, dtype=torch.long),
            attn_mask=attn_mask,  # [T, T] bool, True = blocked
            return_all=True,
        )  # out[:, t] is predicted from x[:, ≠t]

        # Lightweight predictor with register tokens
        predictor = FlexibleTransformer(
            768,
            384,
            768,
            196,
            depth=4,
            num_heads=6,
            self_attn=True,
            cross_attn=False,
            use_adaln=False,
            num_registers=4,
            num_prefix_tokens=0,
        )
        out, registers = predictor(
            context=encoder_output,
            queries=encoder_output[:, :0],
            context_idx=ids_keep,
            query_idx=torch.empty(B, 0, dtype=torch.long),
            return_all=True,
            return_registers=True,
        )
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 384,
        output_dim: int = 768,
        num_patches: int = 196,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        self_attn: bool = True,
        cross_attn: bool = True,
        use_adaln: bool = True,
        pos_embed_type: Literal["sincos_1d", "sincos_2d", "learned"] = "sincos_2d",
        grid_size: Optional[int | tuple[int, int]] = None,
        drop_path_rate: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        zero_init_output: bool = True,
        num_prefix_tokens: int = 1,
        num_registers: int = 0,
        add_mask_token: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.hidden_dim = hidden_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.num_registers = num_registers
        self.use_cross_attn = cross_attn
        self.use_adaln = use_adaln
        self.add_mask_token = add_mask_token

        # Input/output projections
        self.context_proj = nn.Linear(input_dim, hidden_dim)
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        if zero_init_output:
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

        # Positional embeddings
        if pos_embed_type == "sincos_2d":
            if grid_size is None:
                grid_size = int(num_patches**0.5)
                if grid_size**2 != num_patches:
                    raise ValueError(
                        f"num_patches ({num_patches}) must be a perfect square for sincos_2d"
                    )
            pe = get_sincos_pos_embed(
                hidden_dim, num_patches, mode="2d", grid_size=grid_size
            )
            self.register_buffer("pos_embed", pe.unsqueeze(0))
        elif pos_embed_type == "sincos_1d":
            pe = get_sincos_pos_embed(hidden_dim, num_patches, mode="1d")
            self.register_buffer("pos_embed", pe.unsqueeze(0))
        else:  # learned
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
            trunc_normal_(self.pos_embed, std=0.02)

        # Prefix token positional embeddings (content comes from input)
        if num_prefix_tokens > 0:
            self.prefix_pos_embed = nn.Parameter(
                torch.zeros(1, num_prefix_tokens, hidden_dim)
            )
            nn.init.normal_(self.prefix_pos_embed, std=0.02)

        # Learnable register tokens (both content and position are learned)
        if num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, num_registers, hidden_dim)
            )
            self.register_pos_embed = nn.Parameter(
                torch.zeros(1, num_registers, hidden_dim)
            )
            nn.init.normal_(self.register_tokens, std=0.02)
            nn.init.normal_(self.register_pos_embed, std=0.02)

        # Learnable mask token (shared for context and query masking)
        if add_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.mask_token, std=0.02)

        # Time embedding MLP (only needed for AdaLN)
        if use_adaln:
            self.time_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )

        # Transformer blocks with linearly increasing drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    self_attn=self_attn,
                    cross_attn=cross_attn,
                    use_adaln=use_adaln,
                    drop_path=dpr[i],
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for i in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def _gather_pos(self, idx: torch.Tensor, num_prefix: int = 0) -> torch.Tensor:
        """Gather positional embeddings for given indices.

        :param idx: Token indices [B, N] where values index into pos_embed
        :param num_prefix: Number of prefix tokens at the start of idx
        :return: Positional embeddings [B, N, hidden_dim]
        """
        B = idx.shape[0]

        if num_prefix > 0:
            prefix_pos = self.prefix_pos_embed.expand(B, -1, -1)
            patch_idx = idx[:, num_prefix:]
            patch_pos = torch.gather(
                self.pos_embed.expand(B, -1, -1),
                dim=1,
                index=patch_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
            )
            return torch.cat([prefix_pos, patch_pos], dim=1)
        else:
            idx = idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            return torch.gather(self.pos_embed.expand(B, -1, -1), 1, idx)

    def _get_registers(self, batch_size: int) -> torch.Tensor:
        """Get register tokens with positional embeddings.

        :param batch_size: Batch size B
        :return: Register tokens [B, num_registers, hidden_dim]
        """
        return (self.register_tokens + self.register_pos_embed).expand(
            batch_size, -1, -1
        )

    def _expand_attn_mask(
        self,
        attn_mask: torch.Tensor,
        n_registers: int,
        n_context: int,
        n_queries: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand attention mask to account for registers and queries.

        The input attn_mask is for the original context tokens. This function
        expands it to cover [registers, context, queries] for joint attention.

        :param attn_mask: Original mask [T, T] or [B, T, T] for context tokens.
            True = blocked, False = allowed.
        :param n_registers: Number of register tokens
        :param n_context: Number of context tokens (after projection, before registers)
        :param n_queries: Number of query tokens
        :param device: Target device
        :return: Expanded mask [T_total, T_total] or [B, T_total, T_total]
        """
        has_batch = attn_mask.dim() == 3
        T_total = n_registers + n_context + n_queries

        if has_batch:
            B = attn_mask.shape[0]
            expanded = torch.zeros(B, T_total, T_total, dtype=torch.bool, device=device)
        else:
            expanded = torch.zeros(T_total, T_total, dtype=torch.bool, device=device)

        # Place original mask in the context region
        ctx_start = n_registers
        ctx_end = n_registers + n_context

        if has_batch:
            expanded[:, ctx_start:ctx_end, ctx_start:ctx_end] = attn_mask
        else:
            expanded[ctx_start:ctx_end, ctx_start:ctx_end] = attn_mask

        # Registers and queries can attend to everything (no additional masking)
        # If you want queries to also have leave-one-out, extend the mask accordingly

        return expanded

    def forward(
        self,
        context: torch.Tensor,
        queries: torch.Tensor = None,
        context_idx: torch.Tensor = None,
        query_idx: torch.Tensor = None,
        t: Optional[torch.Tensor] = None,
        num_prefix: Optional[int] = None,
        return_all: bool = False,
        return_registers: bool = False,
        context_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        :param context: Context token embeddings [B, N_ctx, input_dim]
        :param queries: Query token embeddings [B, N_qry, input_dim]
        :param context_idx: Patch indices for context tokens [B, N_ctx]
        :param query_idx: Patch indices for query tokens [B, N_qry]
        :param t: Timesteps for conditioning [B] (required if use_adaln=True)
        :param num_prefix: Override for number of prefix tokens in context
        :param return_all: If True and using joint attention (cross_attn=False),
            return all tokens unshuffled to original position order.
            Output shape: [B, N_ctx + N_qry, output_dim].
            Ignored for cross-attention modes.
        :param return_registers: If True and num_registers > 0, also return
            register token outputs as a second tensor. Returns tuple of
            (main_output, register_output) where register_output is
            [B, num_registers, output_dim].
        :param context_mask: Boolean mask indicating which context tokens to replace
            with [MASK] token [B, N_ctx]. True = replace with mask. Each sample can
            have a different number of True values. Requires add_mask_token=True.
        :param query_mask: Boolean mask indicating which query tokens to replace
            with [MASK] token [B, N_qry]. True = replace with mask. Each sample can
            have a different number of True values. Requires add_mask_token=True.
        :param attn_mask: Attention mask for self-attention [T, T] or [B, T, T].
            True = blocked (cannot attend), False = allowed.
            For leave-one-out prediction, use `torch.eye(T, dtype=torch.bool)`.
            Only applies to joint attention mode (cross_attn=False).
            The mask is automatically expanded to account for registers.
        :return: Output embeddings. Shape depends on mode:
            - cross_attn=True: [B, N_qry, output_dim]
            - cross_attn=False, return_all=False: [B, N_qry, output_dim]
            - cross_attn=False, return_all=True: [B, N_ctx + N_qry, output_dim]
            If return_registers=True, returns tuple (output, registers) where
            registers is [B, num_registers, output_dim].
        """
        B, N_ctx, _ = context.shape
        device = context.device

        # Default: empty queries
        if queries is None:
            queries = context.new_empty(B, 0, context.shape[-1])

        N_qry = queries.shape[1]

        # Default: sequential indices
        if context_idx is None:
            context_idx = torch.arange(N_ctx, device=device).expand(B, -1)
        if query_idx is None:
            query_idx = torch.arange(N_qry, device=device).expand(B, -1)

        # Validate mask token usage
        if context_mask is not None or query_mask is not None:
            if not self.add_mask_token:
                raise ValueError(
                    "context_mask or query_mask provided but "
                    "add_mask_token=False at initialization"
                )

        if num_prefix is None:
            num_prefix = self.num_prefix_tokens

        B = context.shape[0]
        n_registers = self.num_registers

        # Project context and optionally replace masked positions with [MASK] token
        context = self.context_proj(context)
        if context_mask is not None:
            mask_tokens = self.mask_token.expand_as(context)
            context = torch.where(context_mask.unsqueeze(-1), mask_tokens, context)
        context = context + self._gather_pos(context_idx, num_prefix)

        # Project queries and optionally replace masked positions with [MASK] token
        queries = self.query_proj(queries)
        if query_mask is not None:
            mask_tokens = self.mask_token.expand_as(queries)
            queries = torch.where(query_mask.unsqueeze(-1), mask_tokens, queries)
        queries = queries + self._gather_pos(query_idx)

        n_context_orig = context.shape[1]  # before adding registers
        n_queries = queries.shape[1]

        # Prepend learnable register tokens to context
        if n_registers > 0:
            registers = self._get_registers(B)
            context = torch.cat([registers, context], dim=1)

        # Time conditioning (only for AdaLN mode)
        cond = None
        if self.use_adaln:
            if t is None:
                raise ValueError("Timestep t required when use_adaln=True")
            cond = self.time_mlp(get_timestep_embed(t, self.hidden_dim))

        n_context = context.shape[1]  # includes registers

        if self.use_cross_attn:
            # Cross-attention mode: queries attend to context (including registers)
            # attn_mask not typically used here, but could be passed for cross-attn masking
            for block in self.blocks:
                queries = block(
                    queries, context=context, cond=cond, attn_mask=attn_mask
                )
            out = self.output_proj(self.final_norm(queries))

            if return_registers and n_registers > 0:
                reg_out = self.output_proj(self.final_norm(registers))
                return out, reg_out

            return out

        # Joint attention mode
        x = torch.cat([context, queries], dim=1)

        # Expand attention mask to cover [registers, context, queries]
        expanded_attn_mask = None
        if attn_mask is not None:
            expanded_attn_mask = self._expand_attn_mask(
                attn_mask, n_registers, n_context_orig, n_queries, x.device
            )

        for block in self.blocks:
            x = block(x, cond=cond, attn_mask=expanded_attn_mask)
        x = self.final_norm(x)

        # Extract register outputs if needed
        reg_out = None
        if return_registers and n_registers > 0:
            reg_out = self.output_proj(x[:, :n_registers])

        if return_all:
            # Unshuffle to original positions (excluding registers)
            T = n_context_orig + n_queries
            out = torch.empty(B, T, self.hidden_dim, device=x.device, dtype=x.dtype)

            # Context part (skip registers)
            out.scatter_(
                dim=1,
                index=context_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
                src=x[:, n_registers:n_context],
            )
            # Query part
            if n_queries > 0:
                out.scatter_(
                    dim=1,
                    index=query_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
                    src=x[:, n_context:],
                )
            out = self.output_proj(out)

            if return_registers and n_registers > 0:
                return out, reg_out
            return out

        # Return only query outputs
        if n_queries == 0:
            out = torch.empty(
                B, 0, self.output_proj.out_features, device=x.device, dtype=x.dtype
            )
        else:
            out = self.output_proj(x[:, -n_queries:])

        if return_registers and n_registers > 0:
            return out, reg_out
        return out


class MAEDecoder(nn.Module):
    """MAE-style Vision Transformer Decoder using FlexibleTransformer.

    Implements the decoder component of Masked Autoencoders (MAE) [1]_ for
    self-supervised visual representation learning. The decoder reconstructs
    masked patches from visible patch embeddings using joint self-attention,
    where visible tokens and learnable mask tokens attend to each other.
    The decoder is intentionally lightweight compared to the encoder, as MAE
    demonstrates that a shallow decoder is sufficient for pixel reconstruction
    while keeping the encoder focused on learning semantic representations.
    Architecture Overview
    ---------------------
    1. **Input projection**: Maps encoder embeddings (embed_dim) to decoder
       dimension (decoder_embed_dim)
    2. **Mask token expansion**: Learnable mask tokens are placed at masked
       positions
    3. **Positional encoding**: Adds position information to all tokens
    4. **Transformer blocks**: Joint self-attention over visible + mask tokens
    5. **Output projection**: Maps to output_dim (typically patch_size² × channels)
    Parameters
    ----------
    embed_dim : int, default=768
        Embedding dimension from the encoder. This is the input dimension
        of visible tokens passed to the decoder.
    decoder_embed_dim : int, default=512
        Internal hidden dimension of the decoder transformer blocks.
        Typically smaller than embed_dim for efficiency.
    output_dim : int, default=768
        Output dimension per token. For pixel reconstruction, this should be
        ``patch_size ** 2 * in_channels`` (e.g., 16×16×3 = 768 for RGB).
    num_patches : int, default=196
        Total number of patches T in the image (e.g., 14×14 = 196 for
        224×224 images with patch_size=16).
    depth : int, default=4
        Number of transformer blocks in the decoder. MAE typically uses
        fewer blocks than the encoder (e.g., 4-8 vs 12-24).
    num_heads : int, default=16
        Number of attention heads in multi-head self-attention.
    mlp_ratio : float, default=4.0
        Expansion ratio for the MLP hidden dimension relative to
        decoder_embed_dim.
    pos_embed_type : {'sincos_1d', 'sincos_2d', 'learned'}, default='sincos_2d'
        Type of positional embedding:
        - 'sincos_2d': Fixed 2D sinusoidal (recommended for images)
        - 'sincos_1d': Fixed 1D sinusoidal
        - 'learned': Learnable positional embeddings
    grid_size : int, optional
        Spatial grid size for 2D positional embeddings. If None, inferred
        as ``int(sqrt(num_patches))``. Required for non-square grids.
    drop_path_rate : float, default=0.0
        Stochastic depth rate for regularization during training.

    Attributes:
    ----------
    mask_token : nn.Parameter
        Learnable token of shape (1, 1, embed_dim) used to represent
        masked positions. Initialized with truncated normal (std=0.02).
    transformer : FlexibleTransformer
        Core transformer module handling attention and projections.

    Notes:
    -----
    - The mask convention follows MAE: **0 = visible/kept, 1 = masked**
    - The decoder receives visible tokens and reconstructs masked positions
    - For efficiency, only masked positions are predicted by default

    References:
    ----------
    .. [1] He, K., et al. "Masked Autoencoders Are Scalable Vision Learners."
           CVPR 2022. https://arxiv.org/abs/2111.06377

    Examples:
    --------
    **Basic Usage with MAE Encoder**
    >>> import torch
    >>> import torch.nn as nn
    >>>
    >>> # Configuration matching ViT-Base
    >>> B, T = 4, 196  # batch size, num_patches (14x14)
    >>> embed_dim = 768  # encoder dimension
    >>> mask_ratio = 0.75  # MAE default: mask 75% of patches
    >>>
    >>> # Initialize decoder
    >>> decoder = MAEDecoder(
    ...     embed_dim=embed_dim,
    ...     decoder_embed_dim=512,
    ...     output_dim=16 * 16 * 3,  # patch_size² × channels = 768
    ...     num_patches=T,
    ...     depth=4,
    ...     num_heads=16,
    ... )
    >>>
    >>> # Simulate encoder output (visible tokens only)
    >>> N_vis = int(T * (1 - mask_ratio))  # 49 visible patches
    >>> visible_tokens = torch.randn(B, N_vis, embed_dim)
    >>>
    >>> # Create random mask (0=visible, 1=masked)
    >>> mask = torch.zeros(B, T)
    >>> for i in range(B):
    ...     masked_indices = torch.randperm(T)[: T - N_vis]
    ...     mask[i, masked_indices] = 1
    >>>
    >>> # Decode - predict masked patches only
    >>> pred_masked = decoder(visible_tokens, mask, output_masked_only=True)
    >>> print(pred_masked.shape)  # [B, N_mask, output_dim]
    torch.Size([4, 147, 768])
    **Full Sequence Reconstruction**
    >>> # Get predictions for ALL positions (for visualization)
    >>> pred_full = decoder(visible_tokens, mask, output_masked_only=False)
    >>> print(pred_full.shape)  # [B, T, output_dim]
    torch.Size([4, 196, 768])
    **Using Full Sequence Input**
    If you have the full sequence with mask tokens already inserted:
    >>> full_sequence = torch.randn(B, T, embed_dim)  # [B, 196, 768]
    >>> pred = decoder(full_sequence, mask, output_masked_only=True)
    >>> print(pred.shape)
    torch.Size([4, 147, 768])
    **Integration with MAE Training Loop**
    >>> # Typical MAE training step (pseudocode)
    >>> def mae_forward(encoder, decoder, images, mask_ratio=0.75):
    ...     # Patchify and mask
    ...     patches = patchify(images)  # [B, T, patch_dim]
    ...     mask = random_mask(B, T, mask_ratio)  # [B, T], 0=keep, 1=mask
    ...
    ...     # Encode visible patches only
    ...     visible_patches = patches[~mask.bool()].reshape(B, -1, patch_dim)
    ...     latent = encoder(visible_patches)  # [B, N_vis, embed_dim]
    ...
    ...     # Decode to predict masked patches
    ...     pred = decoder(
    ...         latent, mask, output_masked_only=True
    ...     )  # [B, N_mask, output_dim]
    ...
    ...     # Reconstruction loss on masked patches only
    ...     target = patches[mask.bool()].reshape(B, -1, patch_dim)
    ...     loss = F.mse_loss(pred, target)
    ...     return loss
    **Custom Configuration for ViT-Large**
    >>> decoder_large = MAEDecoder(
    ...     embed_dim=1024,  # ViT-L encoder dim
    ...     decoder_embed_dim=512,  # Keep decoder lightweight
    ...     output_dim=768,  # 16×16×3 pixels
    ...     num_patches=256,  # 16×16 patches for 256×256 images
    ...     depth=8,  # Slightly deeper
    ...     num_heads=16,
    ...     pos_embed_type="sincos_2d",
    ...     drop_path_rate=0.1,  # Regularization
    ... )

    See Also:
    --------
    FlexibleTransformer : Core transformer implementation used internally.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        output_dim: int = 768,
        num_patches: int = 196,
        depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        pos_embed_type: Literal["sincos_1d", "sincos_2d", "learned"] = "sincos_2d",
        grid_size: Optional[int] = None,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.num_patches = num_patches
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.mask_token, std=0.02)
        # Core transformer
        self.transformer = FlexibleTransformer(
            input_dim=embed_dim,
            hidden_dim=decoder_embed_dim,
            output_dim=output_dim,
            num_patches=num_patches,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            self_attn=True,
            cross_attn=False,
            use_adaln=False,
            pos_embed_type=pos_embed_type,
            grid_size=grid_size,
            drop_path_rate=drop_path_rate,
            zero_init_output=False,
            num_prefix_tokens=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        ids_keep: torch.Tensor | None = None,
        output_masked_only: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        :param x: Visible tokens [B, N_vis, D] or full sequence [B, T, D]
        :param mask: Binary mask [B, T], 0=kept, 1=masked
        :param ids_keep: Indices of kept (visible) patches (B, N_keep)
        :param output_masked_only: If True, return [B, N_mask, D].
                                If False, return [B, T, D].
        :return: Predictions
        """
        B, T = mask.shape
        mask_bool = mask.bool()  # Convert once, use everywhere

        n_vis_per = (~mask_bool).sum(dim=1)
        n_mask_per = mask_bool.sum(dim=1)

        assert torch.all(n_vis_per == n_vis_per[0]), (
            "Number of visible patches must be the same for all samples"
        )

        N_vis = int(n_vis_per[0].item())
        N_mask = int(n_mask_per[0].item())

        if N_mask == 0:
            # visible idx is all patches in order
            visible_idx = torch.arange(T, device=mask.device).unsqueeze(0).expand(B, -1)
            visible_tokens = x if x.shape[1] == T else x
            out = self.transformer(
                context=visible_tokens,
                queries=visible_tokens.new_empty(B, 0, visible_tokens.shape[-1]),
                context_idx=visible_idx,
                query_idx=visible_idx[:, :0],
                return_all=True,
            )

            if output_masked_only:
                return out[:, :0]
            return out

        # Get indices (sort False/0 before True/1, so visible indices come first)
        visible_idx = torch.argsort(mask_bool.int(), dim=1, stable=True)[:, :N_vis]
        masked_idx = torch.argsort(mask_bool.int(), dim=1, stable=True)[:, N_vis:]
        # Get visible tokens
        if x.shape[1] == T:
            visible_tokens = torch.gather(
                x, dim=1, index=visible_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            )
        else:
            if ids_keep is not None:
                visible_idx = ids_keep
            visible_tokens = x

        order = torch.argsort(mask_bool.int(), dim=1, stable=True)
        masked_idx = order[:, N_vis:]

        # Mask tokens for masked positions
        mask_tokens = self.mask_token.expand(B, N_mask, -1)

        return self.transformer(
            context=visible_tokens,
            queries=mask_tokens,
            context_idx=visible_idx,
            query_idx=masked_idx,
            return_all=not output_masked_only,
        )


class PositionalEncoding2D(nn.Module):
    """Flexible 2D positional encoding for vision transformers."""

    def __init__(
        self,
        embed_dim: int,
        grid_size: Tuple[int, int],
        pos_type: Literal["learnable", "sinusoidal", "rope", "none"] = "learnable",
        num_prefix_tokens: int = 1,
        learnable: Optional[
            bool
        ] = None,  # Override: force learnable even for sinusoidal
    ):
        """Positional encoding for 2d input.

        :param embed_dim: Embedding dimension
        :param grid_size: (H, W) grid size in patches
        :param pos_type: Type of positional encoding
        :param num_prefix_tokens: Number of prefix tokens (CLS + registers)
        :param learnable: If True, make sinusoidal learnable; if None, use default

        """
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_h, self.grid_w = grid_size
        self.num_patches = self.grid_h * self.grid_w
        self.pos_type = pos_type
        self.num_prefix_tokens = num_prefix_tokens

        # Override learnable if specified
        if learnable is not None:
            self.is_learnable = learnable
        else:
            self.is_learnable = pos_type == "learnable"

        if pos_type == "none":
            # No positional encoding
            self.pos_embed = None

        elif pos_type == "learnable":
            # Learnable absolute positional embeddings
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_prefix_tokens + self.num_patches, embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        elif pos_type == "sinusoidal":
            # 2D sinusoidal positional embeddings
            pos_embed = self._build_sinusoidal_2d(embed_dim, self.grid_h, self.grid_w)

            # Add prefix token positions (zeros or learned separately)
            prefix_pos = torch.zeros(1, num_prefix_tokens, embed_dim)
            pos_embed = torch.cat([prefix_pos, pos_embed], dim=1)

            if self.is_learnable:
                self.pos_embed = nn.Parameter(pos_embed)
            else:
                self.register_buffer("pos_embed", pos_embed)

        elif pos_type == "rope":
            # RoPE doesn't use additive embeddings
            self.pos_embed = None
            # Precompute RoPE frequencies
            self.register_buffer(
                "freqs_h", self._build_rope_freqs(embed_dim // 4, self.grid_h)
            )
            self.register_buffer(
                "freqs_w", self._build_rope_freqs(embed_dim // 4, self.grid_w)
            )
        else:
            raise ValueError(f"Unknown pos_type: {pos_type}")

    def _build_sinusoidal_2d(
        self, embed_dim: int, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        """Build 2D sinusoidal positional embeddings."""
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sinusoidal"

        dim_h = embed_dim // 2
        dim_w = embed_dim // 2

        # Height positions
        pos_h = torch.arange(grid_h).unsqueeze(1)  # [H, 1]
        dim_t_h = torch.arange(0, dim_h, 2).float()  # [dim_h/2]
        omega_h = 1.0 / (10000 ** (dim_t_h / dim_h))

        pos_embed_h = torch.zeros(grid_h, dim_h)
        pos_embed_h[:, 0::2] = torch.sin(pos_h * omega_h)
        pos_embed_h[:, 1::2] = torch.cos(pos_h * omega_h)

        # Width positions
        pos_w = torch.arange(grid_w).unsqueeze(1)  # [W, 1]
        dim_t_w = torch.arange(0, dim_w, 2).float()
        omega_w = 1.0 / (10000 ** (dim_t_w / dim_w))

        pos_embed_w = torch.zeros(grid_w, dim_w)
        pos_embed_w[:, 0::2] = torch.sin(pos_w * omega_w)
        pos_embed_w[:, 1::2] = torch.cos(pos_w * omega_w)

        # Combine: [H, W, D]
        pos_embed_h = pos_embed_h.unsqueeze(1).expand(-1, grid_w, -1)  # [H, W, dim_h]
        pos_embed_w = pos_embed_w.unsqueeze(0).expand(grid_h, -1, -1)  # [H, W, dim_w]

        pos_embed = torch.cat([pos_embed_h, pos_embed_w], dim=-1)  # [H, W, D]
        pos_embed = pos_embed.reshape(1, grid_h * grid_w, embed_dim)  # [1, H*W, D]

        return pos_embed

    def _build_rope_freqs(
        self, dim: int, max_seq_len: int, base: float = 10000.0
    ) -> torch.Tensor:
        """Build RoPE frequency tensor."""
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(max_seq_len)
        freqs = torch.einsum("i,j->ij", pos, inv_freq)  # [seq_len, dim/2]
        freqs = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        return freqs

    def _apply_rope_2d(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """Apply 2D RoPE to patch tokens."""
        B, N, D = x.shape

        # Separate prefix and patch tokens
        prefix = x[:, : self.num_prefix_tokens, :]
        patches = x[:, self.num_prefix_tokens :, :]  # [B, H*W, D]

        # Reshape to 2D grid
        patches = patches.reshape(B, grid_h, grid_w, D)

        # Split embedding into 4 parts for 2D RoPE
        d_quarter = D // 4
        x1, x2, x3, x4 = patches.split(d_quarter, dim=-1)

        # Get frequencies (interpolate if needed)
        freqs_h = self.freqs_h[:grid_h, :d_quarter]  # [H, d_quarter]
        freqs_w = self.freqs_w[:grid_w, :d_quarter]  # [W, d_quarter]

        # Apply rotation to height dimension (x1, x2)
        cos_h = torch.cos(freqs_h).unsqueeze(1)  # [H, 1, d_quarter]
        sin_h = torch.sin(freqs_h).unsqueeze(1)  # [H, 1, d_quarter]
        x1_rot = x1 * cos_h - x2 * sin_h
        x2_rot = x1 * sin_h + x2 * cos_h

        # Apply rotation to width dimension (x3, x4)
        cos_w = torch.cos(freqs_w).unsqueeze(0)  # [1, W, d_quarter]
        sin_w = torch.sin(freqs_w).unsqueeze(0)  # [1, W, d_quarter]
        x3_rot = x3 * cos_w - x4 * sin_w
        x4_rot = x3 * sin_w + x4 * cos_w

        # Combine
        patches = torch.cat([x1_rot, x2_rot, x3_rot, x4_rot], dim=-1)
        patches = patches.reshape(B, grid_h * grid_w, D)

        # Recombine with prefix (prefix tokens don't get RoPE)
        return torch.cat([prefix, patches], dim=1)

    def forward(
        self, x: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Apply positional encoding.

        :param x: [B, num_prefix + num_patches, D]
        :param grid_size: (H, W) if different from default (for dynamic size)
        :return: x with positional encoding applied
        """
        if self.pos_type == "none":
            return x

        grid_h = grid_size[0] if grid_size else self.grid_h
        grid_w = grid_size[1] if grid_size else self.grid_w

        if self.pos_type == "rope":
            return self._apply_rope_2d(x, grid_h, grid_w)

        # Additive positional embeddings (learnable or sinusoidal)
        pos_embed = self.pos_embed

        # Interpolate if dynamic size
        if grid_h != self.grid_h or grid_w != self.grid_w:
            pos_embed = self._interpolate(pos_embed, grid_h, grid_w)

        return x + pos_embed

    def _interpolate(
        self, pos_embed: torch.Tensor, target_h: int, target_w: int
    ) -> torch.Tensor:
        """Interpolate positional embeddings to new grid size."""
        prefix_pos = pos_embed[:, : self.num_prefix_tokens, :]
        patch_pos = pos_embed[:, self.num_prefix_tokens :, :]

        D = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(1, self.grid_h, self.grid_w, D).permute(
            0, 3, 1, 2
        )
        patch_pos = F.interpolate(
            patch_pos, size=(target_h, target_w), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, target_h * target_w, D)

        return torch.cat([prefix_pos, patch_pos], dim=1)
