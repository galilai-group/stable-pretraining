"""DINOv3: DINOv2 + register tokens + KoLeo regularisation.

Inherits the DINOv2 multi-task self-distillation (CLS + iBOT patch loss
with Sinkhorn-Knopp normalisation) and adds:
- **Register tokens**: extra learnable tokens prepended to the sequence,
  introduced by Darcet et al. 2024 to absorb high-norm artefacts in ViTs.
- **KoLeo regularisation**: encourages student CLS embeddings to be
  uniformly spread over the unit sphere by penalising the distance to
  their nearest neighbour in the batch.

References:
    Siméoni et al. "DINOv3." arXiv 2025.
    Darcet et al. "Vision Transformers Need Registers." ICLR 2024.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import L2Norm, TeacherStudentWrapper
from stable_pretraining.losses import DINOv2Loss


@dataclass
class DINOv3Output(ModelOutput):
    loss: torch.Tensor = None
    loss_cls: torch.Tensor = None
    loss_patch: torch.Tensor = None
    loss_koleo: torch.Tensor = None
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


def _split_cls_patches(features: torch.Tensor, n_special: int):
    """Split features into (CLS, patches), discarding any extra register tokens.

    ``n_special`` = 1 (CLS) + n_registers. The first column is CLS; the next
    ``n_registers`` columns are registers (dropped); the rest are patches.
    """
    if features.ndim == 2:
        return features, None
    cls = features[:, 0]
    patches = features[:, n_special:]
    return cls, patches


def _koleo_loss(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KoLeo: -E[log(min_{j!=i} ||z_i - z_j||)]. Encourages spread."""
    z = F.normalize(z, dim=-1)
    sim = z @ z.T
    sim.fill_diagonal_(-1.0)  # ignore self
    nn_sim = sim.max(dim=-1).values  # closest neighbour cosine similarity
    nn_dist = (2.0 - 2.0 * nn_sim).clamp(min=eps).sqrt()
    return -nn_dist.clamp(min=eps).log().mean()


class DINOv3(Module):
    """DINOv3: DINOv2 with register tokens + KoLeo.

    :param encoder_name: timm ViT name. Register tokens are added on top of
        the timm model via a ``Parameter``.
    :param n_register_tokens: Number of register tokens (default 4).
    :param koleo_weight: Weight on the KoLeo penalty (default 0.1).
    :param projector_hidden_dim: Hidden dim for both heads (default 2048).
    :param projector_bottleneck_dim: Bottleneck dim (default 256).
    :param n_cls_prototypes: CLS prototypes (default 65536).
    :param n_patch_prototypes: Patch prototypes (default 8192).
    :param mask_ratio: Patch mask ratio for the student (default 0.3).
    :param patch_loss_weight: Weight on the patch loss term (default 1.0).
    :param temperature_student: Student softmax temperature (default 0.1).
    :param temperature_teacher: Teacher softmax temperature (default 0.07).
    :param ema_decay_start: Initial EMA (default 0.996).
    :param ema_decay_end: Final EMA (default 1.0).
    :param image_size: Input size (default 224).
    :param pretrained: Load pretrained timm weights.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        n_register_tokens: int = 4,
        koleo_weight: float = 0.1,
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
        if not self._has_cls:
            raise ValueError("DINOv3 requires a CLS-token ViT")
        embed_dim = seq.shape[-1]
        self.embed_dim = embed_dim
        self.n_register_tokens = n_register_tokens
        self.koleo_weight = koleo_weight
        self.mask_ratio = mask_ratio
        self.patch_loss_weight = patch_loss_weight
        self.temperature_teacher = temperature_teacher
        self.image_size = image_size

        # Register tokens (shared between teacher and student via deepcopy).
        register_tokens = nn.Parameter(torch.zeros(1, n_register_tokens, embed_dim))
        nn.init.trunc_normal_(register_tokens, std=0.02)
        base.register_tokens = register_tokens  # attach to the timm module so EMA copies it

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
        # See DINOv2._encode for explanation; _pos_embed needs 4D when
        # dynamic_img_size=True so we reshape only for the mask step.
        is_4d = x.ndim == 4
        if mask is not None:
            if is_4d:
                B_, H_, W_, D_ = x.shape
                x = x.reshape(B_, H_ * W_, D_)
            m = mask.unsqueeze(-1)
            x = x * (1 - m) + self.mask_token.expand_as(x) * m
            if is_4d:
                x = x.reshape(B_, H_, W_, D_)
        x = vit._pos_embed(x)  # [B, 1 + N_patches, D]
        # Insert learnable register tokens between CLS and patches.
        regs = vit.register_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat([x[:, :1], regs, x[:, 1:]], dim=1)
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
    ) -> DINOv3Output:
        n_special = 1 + self.n_register_tokens
        if images is not None:
            with torch.no_grad():
                feats = self._encode(self.backbone.teacher, images, mask=None)
                cls, _ = _split_cls_patches(feats, n_special)
            return DINOv3Output(
                loss=torch.zeros((), device=cls.device, dtype=cls.dtype),
                embedding=cls.detach(),
            )

        if not global_views:
            raise ValueError("DINOv3.forward needs global_views or images")
        global_views = list(global_views)
        local_views = list(local_views or [])
        n_global = len(global_views)
        n_local = len(local_views)
        global_imgs = torch.cat(global_views, dim=0)
        B = global_views[0].shape[0]

        with torch.no_grad():
            pe = self.backbone.student.patch_embed(global_imgs[:1])
            n_patches = pe.shape[1] * pe.shape[2] if pe.ndim == 4 else pe.shape[1]
        mask = self._random_mask(global_imgs.shape[0], n_patches, device=global_imgs.device)

        with torch.no_grad():
            t_feats = self._encode(self.backbone.teacher, global_imgs, mask=None)
            t_cls, t_patches = _split_cls_patches(t_feats, n_special)
            t_cls_logits = self.cls_head.forward_teacher(t_cls).view(n_global, B, -1)
            t_patch_logits = self.patch_head.forward_teacher(t_patches.flatten(0, 1))
            t_patch_logits = t_patch_logits.view(t_patches.shape[0], t_patches.shape[1], -1)

        # Student globals (with patch mask) → CLS + patch logits.
        s_feats_g = self._encode(self.backbone.student, global_imgs, mask=mask)
        s_cls_g, s_patches_g = _split_cls_patches(s_feats_g, n_special)
        s_cls_logits_g = self.cls_head.forward_student(s_cls_g).view(n_global, B, -1)
        s_patch_logits = self.patch_head.forward_student(s_patches_g.flatten(0, 1))
        s_patch_logits = s_patch_logits.view(s_patches_g.shape[0], s_patches_g.shape[1], -1)

        # Student locals contribute only to the CLS-level Sinkhorn loss.
        if n_local > 0:
            local_imgs = torch.cat(local_views, dim=0)
            s_feats_l = self._encode(self.backbone.student, local_imgs, mask=None)
            s_cls_l, _ = _split_cls_patches(s_feats_l, n_special)
            s_cls_logits_l = self.cls_head.forward_student(s_cls_l).view(n_local, B, -1)
            s_cls_logits = torch.cat([s_cls_logits_g, s_cls_logits_l], dim=0)
        else:
            s_cls_logits = s_cls_logits_g

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

        # KoLeo on student CLS features (single set, B samples).
        # KoLeo on the *first global* student CLS embedding only — locals
        # are smaller crops and don't have the right semantics for it.
        s_cls_first_global = s_cls_g.view(n_global, B, -1)[0]
        loss_koleo = _koleo_loss(s_cls_first_global)

        loss = loss_cls + self.patch_loss_weight * loss_patch + self.koleo_weight * loss_koleo
        return DINOv3Output(
            loss=loss,
            loss_cls=loss_cls.detach(),
            loss_patch=loss_patch.detach(),
            loss_koleo=loss_koleo.detach(),
            embedding=t_cls.detach(),
        )
