"""DINOv2: scaled DINO + iBOT with Sinkhorn-Knopp normalisation.

DINOv2 builds on DINO and iBOT by:
- Replacing classical centering with Sinkhorn-Knopp on both CLS and patch
  prototype distributions.
- Using KoLeo regularisation (optional, omitted here for simplicity) to
  spread out features.
- Larger schedules and registers (also omitted; can be set via
  ``encoder_kwargs={"global_pool": "token", "reg_tokens": 4}`` on
  recent timm versions).

This implementation reuses :class:`iBOT` for the architecture and swaps the
loss to use Sinkhorn-Knopp via :class:`DINOv2Loss`.

References:
    Oquab et al. "DINOv2: Learning Robust Visual Features without
    Supervision." TMLR 2024. https://arxiv.org/abs/2304.07193
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import L2Norm, TeacherStudentWrapper
from stable_pretraining.losses import DINOv2Loss


@dataclass
class DINOv2Output(ModelOutput):
    loss: torch.Tensor = None
    loss_cls: torch.Tensor = None
    loss_patch: torch.Tensor = None
    embedding: torch.Tensor = None


def _ibot_head(in_dim: int, hidden_dim: int, bottleneck_dim: int, n_prototypes: int) -> nn.Module:
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
    if features.ndim == 2:
        # Eval path: timm ViT pooled output is already CLS-like.
        return features, None
    if has_cls:
        return features[:, 0], features[:, 1:]
    return features.mean(dim=1), features


class DINOv2(Module):
    """DINOv2: DINO + iBOT with Sinkhorn-Knopp on CLS and patch prototypes.

    :param encoder_name: timm ViT name (default ``"vit_small_patch16_224"``).
    :param projector_hidden_dim: Hidden dim for both heads (default 2048).
    :param projector_bottleneck_dim: Bottleneck dim (default 256).
    :param n_cls_prototypes: CLS prototypes (default 65536).
    :param n_patch_prototypes: Patch prototypes (default 8192).
    :param mask_ratio: Patch mask ratio for the student (default 0.3).
    :param patch_loss_weight: Weight on the patch loss (default 1.0).
    :param temperature_student: Student softmax temperature (default 0.1).
    :param temperature_teacher: Teacher temperature (default 0.07).
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
        temperature_teacher: float = 0.07,
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 1.0,
        image_size: int = 224,
        pretrained: bool = False,
    ):
        super().__init__()

        if isinstance(encoder_name, str):
            import timm

            base = timm.create_model(
                encoder_name,
                num_classes=0,
                pretrained=pretrained,
                dynamic_img_size=True,  # support multi-crop (224 + 96 etc.)
            )
        else:
            base = encoder_name

        with torch.no_grad():
            seq = base.forward_features(torch.zeros(1, 3, image_size, image_size))
        self._has_cls = (
            hasattr(base, "cls_token") and base.cls_token is not None and seq.shape[1] > 1
        )
        embed_dim = seq.shape[-1]
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.patch_loss_weight = patch_loss_weight
        self.temperature_teacher = temperature_teacher
        self.image_size = image_size

        self.backbone = TeacherStudentWrapper(
            base,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.cls_head = TeacherStudentWrapper(
            _ibot_head(embed_dim, projector_hidden_dim, projector_bottleneck_dim, n_cls_prototypes),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.patch_head = TeacherStudentWrapper(
            _ibot_head(embed_dim, projector_hidden_dim, projector_bottleneck_dim, n_patch_prototypes),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.dinov2_loss = DINOv2Loss(student_temp=temperature_student)

    def _encode(self, vit, images, mask=None):
        x = vit.patch_embed(images)
        # With ``dynamic_img_size=True`` patch_embed returns 4D [B, H', W', D]
        # and ``_pos_embed`` *requires* 4D. For the mask substitution we
        # reshape to 3D, apply the mask, then reshape back.
        is_4d = x.ndim == 4
        if mask is not None:
            if is_4d:
                B_, H_, W_, D_ = x.shape
                x = x.reshape(B_, H_ * W_, D_)
            m = mask.unsqueeze(-1)
            x = x * (1 - m) + self.mask_token.expand_as(x) * m
            if is_4d:
                x = x.reshape(B_, H_, W_, D_)
        x = vit._pos_embed(x)
        if hasattr(vit, "patch_drop"):
            x = vit.patch_drop(x)
        if hasattr(vit, "norm_pre"):
            x = vit.norm_pre(x)
        x = vit.blocks(x)
        return vit.norm(x)

    def _random_mask(self, B, N, device):
        n_mask = int(round(N * self.mask_ratio))
        noise = torch.rand(B, N, device=device)
        order = noise.argsort(dim=1)
        mask = torch.zeros(B, N, device=device)
        mask.scatter_(1, order[:, :n_mask], 1.0)
        return mask

    def forward(
        self,
        global_views: Optional[Sequence[torch.Tensor]] = None,
        local_views: Optional[Sequence[torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
    ) -> DINOv2Output:
        if images is not None:
            with torch.no_grad():
                feats = self.backbone.forward_teacher(images)
                cls, _ = _split_cls_patches(feats, self._has_cls)
            return DINOv2Output(
                loss=torch.zeros((), device=cls.device, dtype=cls.dtype),
                embedding=cls.detach(),
            )

        if not global_views:
            raise ValueError("DINOv2.forward needs global_views or images")

        global_views = list(global_views)
        local_views = list(local_views or [])
        n_global = len(global_views)
        n_local = len(local_views)
        global_imgs = torch.cat(global_views, dim=0)
        B = global_views[0].shape[0]

        with torch.no_grad():
            pe = self.backbone.student.patch_embed(global_imgs[:1])
            # ``dynamic_img_size=True`` returns [B, H', W', D]; flat returns [B, N, D].
            n_patches = pe.shape[1] * pe.shape[2] if pe.ndim == 4 else pe.shape[1]
        mask = self._random_mask(global_imgs.shape[0], n_patches, device=global_imgs.device)

        # Teacher: globals only, unmasked. Provides Sinkhorn targets for
        # both CLS and (masked) patches.
        with torch.no_grad():
            t_feats = self._encode(self.backbone.teacher, global_imgs, mask=None)
            t_cls, t_patches = _split_cls_patches(t_feats, self._has_cls)
            t_cls_logits = self.cls_head.forward_teacher(t_cls).view(n_global, B, -1)
            t_patch_logits = self.patch_head.forward_teacher(t_patches.flatten(0, 1))
            t_patch_logits = t_patch_logits.view(t_patches.shape[0], t_patches.shape[1], -1)

        # Student: globals (with patch mask) → CLS + patch logits.
        s_feats_g = self._encode(self.backbone.student, global_imgs, mask=mask)
        s_cls_g, s_patches_g = _split_cls_patches(s_feats_g, self._has_cls)
        s_cls_logits_g = self.cls_head.forward_student(s_cls_g).view(n_global, B, -1)
        s_patch_logits = self.patch_head.forward_student(s_patches_g.flatten(0, 1))
        s_patch_logits = s_patch_logits.view(s_patches_g.shape[0], s_patches_g.shape[1], -1)

        # Student: locals (no mask). Locals contribute *only* to the CLS
        # loss — they have a smaller spatial extent so iBOT-style patch
        # supervision doesn't apply across resolutions (matches paper).
        if n_local > 0:
            local_imgs = torch.cat(local_views, dim=0)
            s_feats_l = self._encode(self.backbone.student, local_imgs, mask=None)
            s_cls_l, _ = _split_cls_patches(s_feats_l, self._has_cls)
            s_cls_logits_l = self.cls_head.forward_student(s_cls_l).view(n_local, B, -1)
            s_cls_logits = torch.cat([s_cls_logits_g, s_cls_logits_l], dim=0)
        else:
            s_cls_logits = s_cls_logits_g

        # Sinkhorn-Knopp on CLS targets
        n_cls = t_cls_logits.numel() // t_cls_logits.shape[-1]
        teacher_cls_probs = self.dinov2_loss.dino_loss.sinkhorn_knopp_teacher(
            t_cls_logits, teacher_temp=self.temperature_teacher, num_samples=n_cls
        )
        loss_cls = self.dinov2_loss.dino_loss(s_cls_logits, teacher_cls_probs)

        mask_flat = mask.bool().view(-1)
        s_patch_flat = s_patch_logits.reshape(-1, s_patch_logits.shape[-1])[mask_flat]
        t_patch_flat = t_patch_logits.reshape(-1, t_patch_logits.shape[-1])[mask_flat]
        teacher_patch_probs = self.dinov2_loss.ibot_loss.sinkhorn_knopp_teacher(
            t_patch_flat,
            teacher_temp=self.temperature_teacher,
            num_samples=t_patch_flat.shape[0],
        )
        loss_patch = self.dinov2_loss.ibot_loss(s_patch_flat, teacher_patch_probs)

        loss = loss_cls + self.patch_loss_weight * loss_patch
        return DINOv2Output(
            loss=loss,
            loss_cls=loss_cls.detach(),
            loss_patch=loss_patch.detach(),
            embedding=t_cls.detach(),
        )
