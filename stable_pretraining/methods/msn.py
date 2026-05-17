"""MSN: Masked Siamese Networks.

DINO-style prototype matching where the student sees a *randomly-masked*
view of the image and the teacher sees the unmasked view. Uses
Sinkhorn-Knopp on the teacher's soft assignments to enforce a balanced
prototype distribution; also adds a mean-entropy regulariser to prevent
trivial solutions.

References:
    Assran et al. "Masked Siamese Networks for Label-Efficient Learning."
    ECCV 2022. https://arxiv.org/abs/2204.07141
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import L2Norm, TeacherStudentWrapper
from stable_pretraining.losses.utils import sinkhorn_knopp


@dataclass
class MSNOutput(ModelOutput):
    """Structured output of the :class:`MSN` SSL method."""

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    student_logits: Optional[torch.Tensor] = None
    teacher_logits: Optional[torch.Tensor] = None


def _msn_head(
    in_dim: int, hidden_dim: int, bottleneck_dim: int, n_prototypes: int
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, bottleneck_dim),
        L2Norm(),
        nn.Linear(bottleneck_dim, n_prototypes, bias=False),
    )


def _to_cls(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 2:
        return features
    if features.ndim == 3:
        return features[:, 0]
    raise ValueError(f"Unexpected backbone output shape {tuple(features.shape)}")


class MSN(Module):
    """MSN: masked siamese DINO-style SSL.

    :param encoder_name: timm ViT name (default ``"vit_small_patch16_224"``).
    :param projector_hidden_dim: Hidden dim (default 2048).
    :param projector_bottleneck_dim: Bottleneck dim (default 256).
    :param n_prototypes: Prototype count (default 1024).
    :param mask_ratio: Patch mask ratio for the *student* (default 0.6).
    :param temperature_student: Student softmax temperature (default 0.1).
    :param temperature_teacher: Teacher softmax temperature (default 0.025).
    :param me_max_weight: Mean-entropy maximisation weight (default 1.0).
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
        n_prototypes: int = 1024,
        mask_ratio: float = 0.6,
        temperature_student: float = 0.1,
        temperature_teacher: float = 0.025,
        me_max_weight: float = 1.0,
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
            seq = base.forward_features(torch.zeros(1, 3, image_size, image_size))
        self._has_cls = (
            hasattr(base, "cls_token")
            and base.cls_token is not None
            and seq.shape[1] > 1
        )
        embed_dim = seq.shape[-1]
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.temperature_student = temperature_student
        self.temperature_teacher = temperature_teacher
        self.me_max_weight = me_max_weight
        self.image_size = image_size

        self.backbone = TeacherStudentWrapper(
            base,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.projector = TeacherStudentWrapper(
            _msn_head(
                embed_dim, projector_hidden_dim, projector_bottleneck_dim, n_prototypes
            ),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _encode(self, vit, images, mask=None):
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

    def _random_mask(self, B, N, device):
        n_mask = int(round(N * self.mask_ratio))
        noise = torch.rand(B, N, device=device)
        order = noise.argsort(dim=1)
        mask = torch.zeros(B, N, device=device)
        mask.scatter_(1, order[:, :n_mask], 1.0)
        return mask

    def forward(
        self,
        view1: Optional[torch.Tensor] = None,
        view2: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> MSNOutput:
        # Eval / single-image
        if images is not None or (view1 is not None and view2 is None):
            single = images if images is not None else view1
            with torch.no_grad():
                feats = self.backbone.forward_teacher(single)
                cls = _to_cls(feats)
            return MSNOutput(
                loss=torch.zeros((), device=cls.device, dtype=cls.dtype),
                embedding=cls.detach(),
            )

        if view1 is None or view2 is None:
            raise ValueError("MSN.forward needs view1 and view2 (or images for eval)")

        # Teacher: unmasked view2
        with torch.no_grad():
            t_feats = self._encode(self.backbone.teacher, view2, mask=None)
            t_cls = _to_cls(t_feats)
            t_logits = self.projector.forward_teacher(t_cls)

        # Student: masked view1
        B = view1.shape[0]
        with torch.no_grad():
            n_patches = self.backbone.student.patch_embed(view1[:1]).shape[1]
        mask = self._random_mask(B, n_patches, device=view1.device)
        s_feats = self._encode(self.backbone.student, view1, mask=mask)
        s_cls = _to_cls(s_feats)
        s_logits = self.projector.forward_student(s_cls)

        # Sinkhorn-Knopp on the teacher targets to enforce a balanced
        # prototype distribution; cross-entropy student vs teacher.
        teacher_probs = sinkhorn_knopp(
            teacher_output=t_logits,
            teacher_temp=self.temperature_teacher,
            num_samples=t_logits.shape[0],
        )
        student_log_probs = F.log_softmax(s_logits / self.temperature_student, dim=-1)
        ce = -(teacher_probs * student_log_probs).sum(dim=-1).mean()

        # Mean-entropy maximisation: prevent collapse to a single prototype
        # by maximising the entropy of the *average* student distribution.
        mean_p = F.softmax(s_logits / self.temperature_student, dim=-1).mean(dim=0)
        me_max = (mean_p * mean_p.clamp(min=1e-8).log()).sum()

        loss = ce + self.me_max_weight * me_max

        return MSNOutput(
            loss=loss,
            embedding=t_cls.detach(),
            student_logits=s_logits,
            teacher_logits=t_logits.detach(),
        )
