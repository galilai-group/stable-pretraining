"""iBOT: Image BERT pre-training with Online Tokenizer.

Combines DINO's CLS-token self-distillation with a per-patch masked
self-distillation: the teacher sees the unmasked image and produces patch
prototype logits; the student sees the masked image, and on masked
positions has to match the teacher's prototype distribution.

References:
    Zhou et al. "iBOT: Image BERT Pre-Training with Online Tokenizer."
    ICLR 2022. https://arxiv.org/abs/2111.07832
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import L2Norm, TeacherStudentWrapper
from stable_pretraining.losses import DINOv1Loss, iBOTPatchLoss


@dataclass
class iBOTOutput(ModelOutput):
    """Structured output of the :class:`iBOT` SSL method."""

    loss: torch.Tensor = None
    loss_cls: torch.Tensor = None
    loss_patch: torch.Tensor = None
    embedding: torch.Tensor = None


def _ibot_head(
    in_dim: int, hidden_dim: int, bottleneck_dim: int, n_prototypes: int
) -> nn.Module:
    """Shared head for both CLS and patch prototypes."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, bottleneck_dim),
        L2Norm(),
        nn.Linear(bottleneck_dim, n_prototypes, bias=False),
    )


def _split_cls_patches(features: torch.Tensor, has_cls: bool):
    # Eval path: timm ViTs return the pooled [B, D] tensor — already CLS-like.
    if features.ndim == 2:
        return features, None
    if features.ndim != 3:
        raise ValueError(f"Expected sequence output, got {tuple(features.shape)}")
    if has_cls:
        return features[:, 0], features[:, 1:]
    # No CLS; use mean as a stand-in CLS for downstream consumers
    return features.mean(dim=1), features


class iBOT(Module):
    """iBOT: DINO on CLS + masked patch self-distillation.

    Architecture:
        - Backbone wrapped with EMA teacher (timm ViT with ``forward_features``).
        - Two prototype heads: CLS head and patch head, both wrapped with EMA.
        - Loss: DINOv1 on CLS + iBOT patch loss on masked patch positions.

    :param encoder_name: timm ViT name (default ``"vit_small_patch16_224"``).
    :param projector_hidden_dim: Hidden dim for both heads (default 2048).
    :param projector_bottleneck_dim: Bottleneck dim before prototypes (default 256).
    :param n_cls_prototypes: Number of CLS prototypes (default 65536).
    :param n_patch_prototypes: Number of patch prototypes (default 8192).
    :param mask_ratio: Patch masking ratio for the student (default 0.3).
    :param patch_loss_weight: Weight on the patch loss term (default 1.0).
    :param temperature_student: Student softmax temperature (default 0.1).
    :param temperature_teacher_warmup: Teacher temperature at start (default 0.04).
    :param temperature_teacher: Teacher temperature after warmup (default 0.07).
    :param warmup_epochs_temperature_teacher: Warmup length (default 30).
    :param ema_decay_start: Initial backbone/head EMA (default 0.996).
    :param ema_decay_end: Final EMA (default 1.0).
    :param image_size: Input size (default 224).
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_hidden_dim: int = 2048,
        projector_bottleneck_dim: int = 256,
        n_cls_prototypes: int = 65536,
        n_patch_prototypes: int = 8192,
        mask_ratio: float = 0.3,
        patch_loss_weight: float = 1.0,
        temperature_student: float = 0.1,
        temperature_teacher_warmup: float = 0.04,
        temperature_teacher: float = 0.07,
        warmup_epochs_temperature_teacher: int = 30,
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 1.0,
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
            dummy = torch.zeros(1, 3, image_size, image_size)
            seq = base.forward_features(dummy)
        self._has_cls = (
            hasattr(base, "cls_token")
            and base.cls_token is not None
            and seq.shape[1] > 1
        )
        embed_dim = seq.shape[-1]
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.patch_loss_weight = patch_loss_weight
        self.image_size = image_size

        self.backbone = TeacherStudentWrapper(
            base,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.cls_head = TeacherStudentWrapper(
            _ibot_head(
                embed_dim,
                projector_hidden_dim,
                projector_bottleneck_dim,
                n_cls_prototypes,
            ),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.patch_head = TeacherStudentWrapper(
            _ibot_head(
                embed_dim,
                projector_hidden_dim,
                projector_bottleneck_dim,
                n_patch_prototypes,
            ),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.dino_loss = DINOv1Loss(temperature_student=temperature_student)
        self.ibot_patch_loss = iBOTPatchLoss(student_temp=temperature_student)
        self.temperature_teacher_warmup = temperature_teacher_warmup
        self.temperature_teacher = temperature_teacher
        self.warmup_epochs_temperature_teacher = warmup_epochs_temperature_teacher

    def _teacher_temperature(self) -> float:
        epoch = int(self.current_epoch) if hasattr(self, "current_epoch") else 0
        warmup = self.warmup_epochs_temperature_teacher
        if epoch >= warmup:
            return self.temperature_teacher
        progress = epoch / max(warmup, 1)
        return self.temperature_teacher_warmup + progress * (
            self.temperature_teacher - self.temperature_teacher_warmup
        )

    def _encode(
        self, vit: nn.Module, images: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Patch-embed → optionally substitute mask tokens → blocks → norm."""
        x = vit.patch_embed(images)
        if mask is not None:
            m = mask.unsqueeze(-1)
            x = x * (1 - m) + self.mask_token.expand_as(x) * m
        if self._has_cls:
            cls = vit.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = x + vit.pos_embed
        x = vit.pos_drop(x)
        x = vit.blocks(x)
        return vit.norm(x)

    def _random_mask(self, B: int, N: int, device) -> torch.Tensor:
        n_mask = int(round(N * self.mask_ratio))
        noise = torch.rand(B, N, device=device)
        order = noise.argsort(dim=1)
        mask = torch.zeros(B, N, device=device)
        mask.scatter_(1, order[:, :n_mask], 1.0)
        return mask

    def forward(
        self,
        global_views: Optional[Sequence[torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
    ) -> iBOTOutput:
        """Forward pass.

        :param global_views: List of ``n_global`` tensors ``[B, C, H, W]``.
        :param images: Single batch for evaluation.
        """
        if images is not None:
            with torch.no_grad():
                feats = self.backbone.forward_teacher(images)
                cls, _ = _split_cls_patches(feats, self._has_cls)
            return iBOTOutput(
                loss=torch.zeros((), device=cls.device, dtype=cls.dtype),
                embedding=cls.detach(),
            )

        if not global_views:
            raise ValueError("iBOT.forward needs global_views or images")

        global_views = list(global_views)
        n_global = len(global_views)
        global_imgs = torch.cat(global_views, dim=0)
        B = global_views[0].shape[0]

        # Random patch mask for the student forward only
        with torch.no_grad():
            n_patches = self.backbone.student.patch_embed(global_imgs[:1]).shape[1]
        mask = self._random_mask(
            global_imgs.shape[0], n_patches, device=global_imgs.device
        )

        # Teacher: unmasked
        with torch.no_grad():
            t_feats = self._encode(self.backbone.teacher, global_imgs, mask=None)
            t_cls, t_patches = _split_cls_patches(t_feats, self._has_cls)
            t_cls_logits = self.cls_head.forward_teacher(t_cls).view(n_global, B, -1)
            t_patch_logits = self.patch_head.forward_teacher(t_patches.flatten(0, 1))
            t_patch_logits = t_patch_logits.view(
                t_patches.shape[0], t_patches.shape[1], -1
            )

        # Student: masked
        s_feats = self._encode(self.backbone.student, global_imgs, mask=mask)
        s_cls, s_patches = _split_cls_patches(s_feats, self._has_cls)
        s_cls_logits = self.cls_head.forward_student(s_cls).view(n_global, B, -1)
        s_patch_logits = self.patch_head.forward_student(s_patches.flatten(0, 1))
        s_patch_logits = s_patch_logits.view(s_patches.shape[0], s_patches.shape[1], -1)

        teacher_temp = self._teacher_temperature()
        teacher_probs = self.dino_loss.softmax_center_teacher(
            t_cls_logits, teacher_temp=teacher_temp
        )
        loss_cls = self.dino_loss(s_cls_logits, teacher_probs)
        self.dino_loss.update_center(t_cls_logits)

        # iBOT patch loss: select masked positions on a per-image basis,
        # then run Sinkhorn-Knopp on the teacher targets.
        mask_flat = mask.bool().view(-1)  # [n_imgs * N]
        s_patch_flat = s_patch_logits.reshape(-1, s_patch_logits.shape[-1])[mask_flat]
        t_patch_flat = t_patch_logits.reshape(-1, t_patch_logits.shape[-1])[mask_flat]
        teacher_patch_probs = self.ibot_patch_loss.sinkhorn_knopp_teacher(
            t_patch_flat,
            teacher_temp=teacher_temp,
            num_samples=t_patch_flat.shape[0],
        )
        loss_patch = self.ibot_patch_loss(s_patch_flat, teacher_patch_probs)
        loss = loss_cls + self.patch_loss_weight * loss_patch

        return iBOTOutput(
            loss=loss,
            loss_cls=loss_cls.detach(),
            loss_patch=loss_patch.detach(),
            embedding=t_cls.detach(),
        )
