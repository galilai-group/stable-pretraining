"""iGPT: Autoregressive image modelling.

This implementation follows the modern *AIM* lineage (Apple, 2024) — an
autoregressive ViT that predicts the next *patch* (not the next pixel) by
regressing pixel values with MSE. It keeps the iGPT spirit (left-to-right
transformer over an image sequence) while sidestepping the pixel-cluster
tokenization that the original 2020 iGPT paper depended on.

If you want the classical pixel-clustered iGPT, supply a custom tokenizer
that maps images to discrete pixel codes (analogous to BEiT's tokenizer).

References:
    Chen, Radford, et al. "Generative Pretraining from Pixels." ICML 2020.
        https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf
    El-Nouby et al. "Scalable Pre-training of Large Autoregressive Image
        Models." arXiv 2024. https://arxiv.org/abs/2401.08541
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import patchify


@dataclass
class iGPTOutput(ModelOutput):
    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    predictions: Optional[torch.Tensor] = None


def _causal_mask(N: int, device) -> torch.Tensor:
    """Boolean upper-triangular mask of shape [N, N] (True = block)."""
    return torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)


class iGPT(Module):
    """Autoregressive image GPT (AIM-style next-patch regression).

    A standard timm ViT encoder is used in causal mode: every patch can
    only attend to itself and earlier patches (raster order). At every
    position the model predicts the *next* patch's pixel values via a
    linear head and minimises MSE.

    :param encoder_name: timm ViT model name (default ``"vit_small_patch16_224"``).
    :param patch_size: Patch side length (default 16, must match encoder).
    :param image_size: Input size (default 224).
    :param in_channels: Image channels (default 3).
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        patch_size: int = 16,
        image_size: int = 224,
        in_channels: int = 3,
        pretrained: bool = False,
    ):
        super().__init__()

        if isinstance(encoder_name, str):
            import timm

            self.encoder = timm.create_model(
                encoder_name, num_classes=0, pretrained=pretrained
            )
        else:
            self.encoder = encoder_name

        with torch.no_grad():
            seq = self.encoder.forward_features(
                torch.zeros(1, in_channels, image_size, image_size)
            )
        embed_dim = seq.shape[-1]
        self._has_cls = (
            hasattr(self.encoder, "cls_token")
            and self.encoder.cls_token is not None
            and seq.shape[1] > 1
        )
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.in_channels = in_channels

        # Linear head: predict the next patch's pixel values.
        self.head = nn.Linear(embed_dim, in_channels * patch_size * patch_size)

        # Pre-build attention mask hooks: timm's attention is fused, so we set
        # the global ``attn_mask`` attribute on each block at forward time.
        self._mask_cache: dict = {}

    def _causal_forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Run the encoder with a causal mask over patch tokens.

        We bypass ``forward_features`` so we can pass an attention mask
        through. Compatible with the standard timm ``Attention`` module
        which accepts ``attn_mask`` via PyTorch's ``F.scaled_dot_product_attention``
        (timm 1.x).
        """
        vit = self.encoder
        B = images.shape[0]
        x = vit.patch_embed(images)
        if self._has_cls:
            cls = vit.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = x + vit.pos_embed
        x = vit.pos_drop(x)

        N = x.shape[1]
        if N not in self._mask_cache or self._mask_cache[N].device != x.device:
            self._mask_cache[N] = _causal_mask(N, x.device)
        mask = self._mask_cache[N]

        # timm Block.forward accepts ``attn_mask`` keyword in recent versions;
        # fall back to manually patching each block's Attention if not supported.
        for block in vit.blocks:
            try:
                x = block(x, attn_mask=mask)
            except TypeError:
                # Older timm: monkey-patch attention call.
                x = self._block_with_mask(block, x, mask)
        x = vit.norm(x)
        return x

    @staticmethod
    def _block_with_mask(block: nn.Module, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply a transformer block with a causal mask (manual MHA)."""
        attn = block.attn
        B, N, C = x.shape
        h = attn.num_heads
        qkv = attn.qkv(block.norm1(x))
        qkv = qkv.reshape(B, N, 3, h, C // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=~mask)
        a = a.transpose(1, 2).reshape(B, N, C)
        a = attn.proj_drop(attn.proj(a))
        x = x + a
        x = x + block.mlp(block.norm2(x))
        return x

    def forward(self, images: torch.Tensor) -> iGPTOutput:
        """Forward pass.

        :param images: ``[B, C, H, W]``.
        """
        B, C, H, W = images.shape
        if not self.training:
            features = self.encoder.forward_features(images)
            cls = features[:, 0] if self._has_cls else features.mean(dim=1)
            return iGPTOutput(
                loss=torch.zeros((), device=images.device, dtype=images.dtype),
                embedding=cls,
            )

        encoded = self._causal_forward_features(images)  # [B, N(+1), D]
        # Drop CLS column for prediction alignment.
        if self._has_cls:
            encoded_patches = encoded[:, 1:]
        else:
            encoded_patches = encoded
        # Predict next-patch pixels: position i predicts patch i+1.
        pred_pixels = self.head(encoded_patches[:, :-1])  # [B, N-1, P]

        # Targets are patch i+1's flat pixel values.
        targets = patchify(images, (self.in_channels, self.patch_size, self.patch_size))
        # Per-patch normalisation (helps training stability, like MAE).
        mean = targets.mean(dim=-1, keepdim=True)
        var = targets.var(dim=-1, keepdim=True)
        targets = (targets - mean) / (var + 1e-6).sqrt()
        targets = targets[:, 1:]  # shift so target at position i is patch i+1

        loss = F.mse_loss(pred_pixels, targets)

        # Embedding for probes: mean over patch tokens.
        embedding = encoded_patches.mean(dim=1).detach()

        return iGPTOutput(loss=loss, embedding=embedding, predictions=pred_pixels)
