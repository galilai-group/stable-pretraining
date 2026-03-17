"""NEPA: Next-Embedding Predictive Autoregression."""

from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, PatchEmbed

from stable_pretraining import Module
from stable_pretraining.backbone import TransformerBlock


@dataclass
class NEPAOutput:
    """Docstring for NEPAOutput."""

    loss: torch.Tensor
    embeddings: torch.Tensor
    grid_size: Tuple[int, int]


class NEPA(Module):
    """NEPA: Next-Embedding Predictive Autoregression.

    Uses standard TransformerBlock with modern options enabled:
        - ``use_rope=True``: 2D Rotary Position Embedding
        - ``use_qk_norm=True``: Query-Key normalization
        - ``mlp_type='swiglu'``: Gated MLP activation
        - ``use_layer_scale=True``: Residual scaling

    Causal masking is applied via ``attn_mask`` during training.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_rope: bool = True,
        use_qk_norm: bool = True,
        use_swiglu: bool = True,
        layer_scale_init: float = 1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_rope = use_rope

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        gs = self.patch_embed.grid_size
        self.grid_h = gs[0] if isinstance(gs, tuple) else gs
        self.grid_w = gs[1] if isinstance(gs, tuple) else gs

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Additive pos_embed only when RoPE disabled
        if not use_rope:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_buffer("pos_embed", None)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Standard TransformerBlock with modern options
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    self_attn=True,
                    cross_attn=False,
                    use_adaln=False,
                    use_rope=use_rope,
                    use_qk_norm=use_qk_norm,
                    mlp_type="swiglu" if use_swiglu else "gelu",
                    use_layer_scale=True,
                    layer_scale_init=layer_scale_init,
                    drop_path=dpr[i],
                    attn_drop=attn_drop_rate,
                    proj_drop=drop_rate,
                    max_grid_size=max(self.grid_h, self.grid_w) * 2,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)

    def _get_grid_size(self, images: torch.Tensor) -> Tuple[int, int]:
        H, W = images.shape[-2:]
        return H // self.patch_size, W // self.patch_size

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )

    def forward_features(
        self, images: torch.Tensor, causal: bool = False
    ) -> torch.Tensor:
        grid_size = self._get_grid_size(images)
        x = self.patch_embed(images)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        attn_mask = self._get_causal_mask(x.shape[1], x.device) if causal else None
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask, grid_size=grid_size)

        return self.norm(x)

    def forward(self, images: torch.Tensor) -> NEPAOutput:
        grid_size = self._get_grid_size(images)

        input_embed = self.patch_embed(images)
        if self.pos_embed is not None:
            input_embed = input_embed + self.pos_embed
        x = self.pos_drop(input_embed)

        attn_mask = (
            self._get_causal_mask(x.shape[1], x.device) if self.training else None
        )
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask, grid_size=grid_size)

        pred_embed = self.norm(x)

        if self.training:
            target = input_embed.detach()
            pred = F.normalize(pred_embed[:, :-1], dim=-1)
            target = F.normalize(target[:, 1:], dim=-1)
            loss = -(pred * target).sum(dim=-1).mean()
        else:
            loss = torch.tensor(0.0, device=images.device)

        return NEPAOutput(loss=loss, embeddings=pred_embed, grid_size=grid_size)

    def get_classifier_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_features(images, causal=False)[:, -1]

    def get_dense_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_features(images, causal=False)

    def freeze_patch_embed(self):
        for p in self.patch_embed.parameters():
            p.requires_grad = False


def nepa_base_patch14(**kwargs) -> NEPA:
    return NEPA(patch_size=14, embed_dim=768, depth=12, num_heads=12, **kwargs)


def nepa_large_patch14(**kwargs) -> NEPA:
    return NEPA(patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
