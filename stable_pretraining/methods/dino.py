"""DINO: Self-distillation with no labels.

Self-supervised learning by distilling a teacher (EMA of student) into a
student that processes a richer view of the data (multi-crop). The student
matches the teacher's softmaxed prototype assignments via cross-entropy.

References:
    Caron et al. "Emerging Properties in Self-Supervised Vision Transformers."
    ICCV 2021. https://arxiv.org/abs/2104.14294
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import L2Norm, TeacherStudentWrapper
from stable_pretraining.losses import DINOv1Loss


@dataclass
class DINOOutput(ModelOutput):
    """Output from DINO forward pass.

    :ivar loss: DINO cross-entropy loss (0 in eval mode)
    :ivar embedding: Teacher CLS features [B, D] (eval) or [n_global * B, D] (train)
    :ivar teacher_logits: Teacher prototype logits [n_global, B, K] (None in eval)
    :ivar student_logits: Student prototype logits [n_views, B, K] (None in eval)
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    teacher_logits: Optional[torch.Tensor] = None
    student_logits: Optional[torch.Tensor] = None


def _build_dino_projector(
    in_dim: int, hidden_dim: int, bottleneck_dim: int, n_prototypes: int
) -> nn.Module:
    """Standard DINO projector: 3-layer MLP + L2 norm + linear prototypes.

    The prototypes layer is bias-free; weight-norm is applied via the L2Norm
    on the bottleneck (the original DINO uses ``nn.utils.weight_norm`` on the
    prototypes Linear instead — equivalent up to a learnable scale that is
    typically frozen).
    """
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
    """Reduce backbone output to a [B, D] CLS-like representation.

    timm ViTs with ``num_classes=0`` already pool to ``[B, D]`` (default
    is class-token pooling). For backbones that return a sequence
    ``[B, N, D]``, take the first token.
    """
    if features.ndim == 2:
        return features
    if features.ndim == 3:
        return features[:, 0]
    raise ValueError(f"Unexpected backbone output shape {tuple(features.shape)}")


class DINO(Module):
    """DINO self-distillation with multi-crop and an EMA teacher.

    Architecture:
        - **Backbone** (student) wrapped in :class:`TeacherStudentWrapper`
          (teacher is an EMA copy).
        - **Projector** (student) wrapped in :class:`TeacherStudentWrapper`:
          3-layer MLP -> L2-norm -> linear prototypes (default 65k).
        - **Loss**: :class:`DINOv1Loss` with classical centering.

    The teacher only sees global crops; the student sees both global and
    local crops. Loss is the average pairwise cross-entropy between every
    student view and every teacher view (excluding same-view pairs handled
    inside :class:`DINOv1Loss`).

    :param encoder_name: timm model name (default ``"vit_small_patch16_224"``)
        or pre-built ``nn.Module``. For multi-crop, the backbone must accept
        variable input sizes; pass ``dynamic_img_size=True`` via ``encoder_kwargs``
        for timm ViTs.
    :param projector_hidden_dim: Hidden dim of the 3-layer MLP (default 2048).
    :param projector_bottleneck_dim: Bottleneck dim before prototypes (default 256).
    :param n_prototypes: Number of prototypes / output dim (default 65536).
    :param temperature_student: Student softmax temperature (default 0.1).
    :param temperature_teacher_warmup: Teacher temp at start (default 0.04).
    :param temperature_teacher: Teacher temp after warmup (default 0.07).
    :param warmup_epochs_temperature_teacher: Linear warmup epochs (default 30).
    :param center_momentum: EMA momentum for the teacher centering (default 0.9).
    :param ema_decay_start: Initial backbone/projector EMA (default 0.996).
    :param ema_decay_end: Final EMA (default 1.0).
    :param encoder_kwargs: Extra kwargs forwarded to ``timm.create_model``.
    :param pretrained: Load pretrained timm weights for the encoder.

    Example::

        model = DINO("vit_small_patch16_224", encoder_kwargs={"dynamic_img_size": True})

        global_views = [torch.randn(8, 3, 224, 224), torch.randn(8, 3, 224, 224)]
        local_views = [torch.randn(8, 3, 96, 96) for _ in range(6)]

        out = model(global_views=global_views, local_views=local_views)
        out.loss.backward()
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_hidden_dim: int = 2048,
        projector_bottleneck_dim: int = 256,
        n_prototypes: int = 65536,
        temperature_student: float = 0.1,
        temperature_teacher_warmup: float = 0.04,
        temperature_teacher: float = 0.07,
        warmup_epochs_temperature_teacher: int = 30,
        center_momentum: float = 0.9,
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 1.0,
        encoder_kwargs: Optional[dict] = None,
        pretrained: bool = False,
    ):
        super().__init__()

        if isinstance(encoder_name, str):
            import timm

            kw = dict(num_classes=0, pretrained=pretrained)
            kw.update(encoder_kwargs or {})
            base_backbone = timm.create_model(encoder_name, **kw)
        else:
            base_backbone = encoder_name

        with torch.no_grad():
            embed_dim = _to_cls(base_backbone(torch.zeros(1, 3, 224, 224))).shape[-1]
        self.embed_dim = embed_dim
        self.n_prototypes = n_prototypes

        self.backbone = TeacherStudentWrapper(
            base_backbone,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.projector = TeacherStudentWrapper(
            _build_dino_projector(
                embed_dim, projector_hidden_dim, projector_bottleneck_dim, n_prototypes
            ),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )

        self.dino_loss = DINOv1Loss(
            temperature_student=temperature_student,
            center_momentum=center_momentum,
        )
        self.temperature_teacher_warmup = temperature_teacher_warmup
        self.temperature_teacher = temperature_teacher
        self.warmup_epochs_temperature_teacher = warmup_epochs_temperature_teacher

    def _teacher_temperature(self) -> float:
        """Linearly warm up teacher temperature over the configured epochs."""
        epoch = int(self.current_epoch) if hasattr(self, "current_epoch") else 0
        warmup = self.warmup_epochs_temperature_teacher
        if epoch >= warmup:
            return self.temperature_teacher
        progress = epoch / max(warmup, 1)
        return self.temperature_teacher_warmup + progress * (
            self.temperature_teacher - self.temperature_teacher_warmup
        )

    def forward(
        self,
        global_views: Optional[Sequence[torch.Tensor]] = None,
        local_views: Optional[Sequence[torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
    ) -> DINOOutput:
        """Forward pass.

        :param global_views: List of ``n_global`` tensors ``[B, C, H, W]`` (e.g.
            two 224x224 crops). Required in training mode.
        :param local_views: List of ``n_local`` tensors ``[B, C, h, w]`` (e.g.
            six 96x96 crops). Optional.
        :param images: Single batch of images for evaluation. If supplied,
            returns the teacher CLS embedding only.
        :return: :class:`DINOOutput`.
        """
        # Eval / single-image path
        if images is not None:
            with torch.no_grad():
                features = self.backbone.forward_teacher(images)
                cls = _to_cls(features)
            return DINOOutput(
                loss=torch.zeros((), device=cls.device, dtype=cls.dtype),
                embedding=cls.detach(),
            )

        if not global_views:
            raise ValueError("DINO.forward needs global_views or images")

        global_views = list(global_views)
        local_views = list(local_views or [])
        n_global = len(global_views)
        n_local = len(local_views)
        B = global_views[0].shape[0]

        # Teacher: only global views
        global_imgs = torch.cat(global_views, dim=0)
        with torch.no_grad():
            t_features = self.backbone.forward_teacher(global_imgs)
            t_cls = _to_cls(t_features)
            t_logits = self.projector.forward_teacher(t_cls).view(n_global, B, -1)

        # Student: global views (same input as teacher)
        s_features_g = self.backbone.forward_student(global_imgs)
        s_cls_g = _to_cls(s_features_g)
        s_logits_g = self.projector.forward_student(s_cls_g).view(n_global, B, -1)
        student_logits_list: List[torch.Tensor] = [s_logits_g]

        # Student: local views (smaller crops)
        if n_local > 0:
            local_imgs = torch.cat(local_views, dim=0)
            s_features_l = self.backbone.forward_student(local_imgs)
            s_cls_l = _to_cls(s_features_l)
            s_logits_l = self.projector.forward_student(s_cls_l).view(n_local, B, -1)
            student_logits_list.append(s_logits_l)

        student_logits = torch.cat(student_logits_list, dim=0)

        teacher_temp = self._teacher_temperature()
        teacher_probs = self.dino_loss.softmax_center_teacher(
            t_logits, teacher_temp=teacher_temp
        )
        loss = self.dino_loss(student_logits, teacher_probs)
        # Queue async center update for next iteration.
        self.dino_loss.update_center(t_logits)

        return DINOOutput(
            loss=loss,
            embedding=t_cls.detach(),
            teacher_logits=t_logits.detach(),
            student_logits=student_logits,
        )
