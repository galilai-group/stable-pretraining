"""data2vec: Predicting contextualised representations.

Self-supervised learning by having a student predict the EMA-teacher's
contextualised representation (top-K block average of patch tokens) at
masked positions. No augmentations, no negatives, modality-agnostic.

References:
    Baevski et al. "data2vec: A General Framework for Self-supervised
    Learning in Speech, Vision and Language." ICML 2022.
    https://arxiv.org/abs/2202.03555
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import TeacherStudentWrapper


@dataclass
class Data2VecOutput(ModelOutput):
    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    predictions: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None


class _BlockHook:
    """Capture the output of every transformer block via forward hooks."""

    def __init__(self, blocks: nn.ModuleList):
        self.outputs: list = []
        self._handles = [b.register_forward_hook(self._hook) for b in blocks]

    def _hook(self, module, inputs, output):
        self.outputs.append(output if isinstance(output, torch.Tensor) else output[0])

    def reset(self):
        self.outputs = []

    def remove(self):
        for h in self._handles:
            h.remove()


class Data2Vec(Module):
    """data2vec for vision: predict EMA-teacher block-averaged features.

    :param encoder_name: timm ViT name (default ``"vit_small_patch16_224"``).
    :param top_k_blocks: Number of top transformer blocks averaged on the
        teacher side to form the prediction target (default 6).
    :param mask_ratio: Fraction of patch tokens masked on the student input
        (default 0.6). Masked tokens are replaced by a learnable token before
        the encoder.
    :param ema_decay_start: Initial teacher EMA (default 0.999).
    :param ema_decay_end: Final teacher EMA (default 0.9999).
    :param image_size: Input size (default 224).
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        top_k_blocks: int = 6,
        mask_ratio: float = 0.6,
        ema_decay_start: float = 0.999,
        ema_decay_end: float = 0.9999,
        image_size: int = 224,
        pretrained: bool = False,
    ):
        super().__init__()

        if isinstance(encoder_name, str):
            import timm

            base = timm.create_model(encoder_name, num_classes=0, pretrained=pretrained)
        else:
            base = encoder_name

        with torch.no_grad():
            embed_dim = base(torch.zeros(1, 3, image_size, image_size)).shape[-1]
        self.embed_dim = embed_dim
        self.top_k_blocks = top_k_blocks
        self.mask_ratio = mask_ratio
        self.image_size = image_size

        self.encoder = TeacherStudentWrapper(
            base,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Linear regression head that maps encoded features to the target dim
        self.regressor = nn.Linear(embed_dim, embed_dim)

    def _patches_for_vit(self, vit: nn.Module, images: torch.Tensor) -> torch.Tensor:
        return vit.patch_embed(images)

    def _encode_with_mask(
        self, vit: nn.Module, images: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = self._patches_for_vit(vit, images)
        if mask is not None:
            m = mask.unsqueeze(-1)
            x = x * (1 - m) + self.mask_token.expand_as(x) * m
        if hasattr(vit, "cls_token") and vit.cls_token is not None:
            cls = vit.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = x + vit.pos_embed
        x = vit.pos_drop(x)
        x = vit.blocks(x)
        x = vit.norm(x)
        if hasattr(vit, "cls_token") and vit.cls_token is not None:
            x = x[:, 1:]
        return x

    def _teacher_target(
        self, images: torch.Tensor
    ) -> torch.Tensor:
        """Run the unmasked image through the EMA teacher and return the
        average of the last K block outputs at patch positions.

        We install + remove the hook inside this method so we never capture
        the student's forward (which would corrupt the target).
        """
        vit = self.encoder.teacher
        with torch.no_grad():
            hook = _BlockHook(vit.blocks)
            try:
                _ = self._encode_with_mask(vit, images, mask=None)
                blocks = list(hook.outputs[-self.top_k_blocks :])
            finally:
                hook.remove()
        # Drop CLS column for each block output if present.
        cls_offset = 1 if hasattr(vit, "cls_token") and vit.cls_token is not None else 0
        cleaned = [b[:, cls_offset:] for b in blocks]
        target = torch.stack(cleaned, dim=0).mean(dim=0)
        # Per-token layer norm (paper: stability of the regression target).
        target = F.layer_norm(target, [target.shape[-1]])
        return target

    def _random_mask(self, B: int, N: int, device) -> torch.Tensor:
        n_mask = int(round(N * self.mask_ratio))
        noise = torch.rand(B, N, device=device)
        order = noise.argsort(dim=1)
        mask = torch.zeros(B, N, device=device)
        mask.scatter_(1, order[:, :n_mask], 1.0)
        return mask

    def forward(self, images: torch.Tensor) -> Data2VecOutput:
        """Forward pass.

        :param images: ``[B, C, H, W]``.
        """
        B, _, H, W = images.shape
        if not self.training:
            with torch.no_grad():
                feats = self.encoder.forward_teacher(images)
            cls = feats[:, 0] if feats.ndim == 3 else feats
            return Data2VecOutput(
                loss=torch.zeros((), device=images.device, dtype=images.dtype),
                embedding=cls.detach(),
            )

        # Build random mask in the patch grid
        vit = self.encoder.student
        with torch.no_grad():
            n_patches = self._patches_for_vit(vit, images).shape[1]
        mask = self._random_mask(B, n_patches, device=images.device)

        # Student encodes the masked image
        student_tokens = self._encode_with_mask(vit, images, mask)
        predictions = self.regressor(student_tokens)

        # Teacher provides the target representation (no mask)
        target = self._teacher_target(images)

        # Smooth-L1 loss on masked positions only
        diff = F.smooth_l1_loss(predictions, target, beta=2.0, reduction="none").mean(dim=-1)
        loss = (diff * mask).sum() / mask.sum().clamp(min=1.0)

        # Embedding for online probes: mean of all student patch tokens.
        embedding = student_tokens.mean(dim=1)

        return Data2VecOutput(
            loss=loss,
            embedding=embedding.detach(),
            predictions=predictions,
            target=target.detach(),
            mask=mask,
        )
