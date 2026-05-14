"""BEiT: BERT Pre-Training of Image Transformers.

Masked image modeling where the encoder predicts a *discrete* visual-token
ID at every masked patch. The original paper uses a frozen DALL-E
discrete VAE as the tokenizer; BEiT v2 distils a pretrained CLIP via a
VQ-VAE; BEiT v3 uses a multi-modal MoE.

This implementation is tokenizer-agnostic: pass any callable that maps
``[B, C, H, W]`` images to ``[B, N]`` ``int64`` token IDs in
``[0, vocab_size)``. No tokenizer is bundled to keep the dependency
surface small. A simple in-repo placeholder is provided as
:func:`patch_kmeans_tokenizer` for testing.

References:
    Bao et al. "BEiT: BERT Pre-Training of Image Transformers."
    ICLR 2022. https://arxiv.org/abs/2106.08254
"""

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import patchify


@dataclass
class BEiTOutput(ModelOutput):
    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    logits: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None


def patch_kmeans_tokenizer(vocab_size: int, patch_size: int, in_channels: int = 3):
    """Return a dummy tokenizer that hashes flattened patches into ``vocab_size``
    buckets via a fixed random projection. Useful for end-to-end smoke tests
    only — replace with a DALL-E or VQ-VAE tokenizer for real training."""
    proj = None

    def _tok(images: torch.Tensor) -> torch.Tensor:
        nonlocal proj
        patches = patchify(images, (in_channels, patch_size, patch_size))  # [B, N, P]
        if proj is None or proj.shape[0] != patches.shape[-1]:
            g = torch.Generator(device="cpu").manual_seed(0)
            proj = torch.randn(patches.shape[-1], vocab_size, generator=g).to(patches.device)
        scores = patches.float() @ proj.to(patches.device)
        return scores.argmax(dim=-1)  # [B, N]

    return _tok


class BEiT(Module):
    """BEiT masked image modeling with a discrete visual tokenizer.

    :param encoder_name: timm ViT name (default ``"vit_small_patch16_224"``).
    :param tokenizer: Callable ``images -> [B, N] int64`` returning visual
        token IDs. If ``None``, defaults to :func:`patch_kmeans_tokenizer`
        (placeholder; not SOTA).
    :param vocab_size: Number of visual tokens (default 8192, matches DALL-E).
    :param patch_size: Patch size of the encoder (default 16).
    :param mask_ratio: Fraction of patches masked (default 0.4, BEiT used 0.4).
    :param image_size: Input size (default 224).
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        tokenizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        vocab_size: int = 8192,
        patch_size: int = 16,
        mask_ratio: float = 0.4,
        image_size: int = 224,
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
            seq = self.encoder.forward_features(torch.zeros(1, 3, image_size, image_size))
        self._has_cls = (
            hasattr(self.encoder, "cls_token")
            and self.encoder.cls_token is not None
            and seq.shape[1] > 1
        )
        embed_dim = seq.shape[-1]
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.image_size = image_size

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Predict visual-token IDs at each (masked) patch position.
        self.head = nn.Linear(embed_dim, vocab_size)

        self.tokenizer = tokenizer or patch_kmeans_tokenizer(
            vocab_size=vocab_size, patch_size=patch_size
        )

    def _encode_with_mask(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        vit = self.encoder
        x = vit.patch_embed(images)
        m = mask.unsqueeze(-1)
        x = x * (1 - m) + self.mask_token.expand_as(x) * m
        if self._has_cls:
            cls = vit.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = x + vit.pos_embed
        x = vit.pos_drop(x)
        x = vit.blocks(x)
        x = vit.norm(x)
        if self._has_cls:
            x = x[:, 1:]
        return x

    def _random_mask(self, B, N, device):
        n_mask = int(round(N * self.mask_ratio))
        noise = torch.rand(B, N, device=device)
        order = noise.argsort(dim=1)
        mask = torch.zeros(B, N, device=device)
        mask.scatter_(1, order[:, :n_mask], 1.0)
        return mask

    def forward(self, images: torch.Tensor) -> BEiTOutput:
        B = images.shape[0]
        if not self.training:
            features = self.encoder.forward_features(images)
            cls = features[:, 0] if self._has_cls else features.mean(dim=1)
            return BEiTOutput(
                loss=torch.zeros((), device=images.device, dtype=images.dtype),
                embedding=cls,
            )

        # Tokenize the full image (no augmentation needed for token targets).
        with torch.no_grad():
            targets = self.tokenizer(images)  # [B, N]
        N = targets.shape[1]
        mask = self._random_mask(B, N, device=images.device)

        encoded = self._encode_with_mask(images, mask)  # [B, N, D]
        logits = self.head(encoded)  # [B, N, V]

        # Cross-entropy over masked positions only.
        flat_mask = mask.bool().view(-1)
        flat_logits = logits.reshape(-1, self.vocab_size)[flat_mask]
        flat_targets = targets.reshape(-1)[flat_mask]
        loss = F.cross_entropy(flat_logits, flat_targets)

        # Embedding for online probes: mean of unmasked encoded tokens.
        unmasked = (1 - mask).unsqueeze(-1)
        embedding = (encoded * unmasked).sum(dim=1) / unmasked.sum(dim=1).clamp(min=1.0)

        return BEiTOutput(
            loss=loss,
            embedding=embedding.detach(),
            logits=logits,
            targets=targets,
            mask=mask,
        )
