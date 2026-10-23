"""ReSA: Representation Self-Assignment.

Contrastive learning whose targets come from the encoder's own similarity
structure. Sinkhorn-Knopp turns the cosine similarities between the student's
and the EMA teacher's encoder features of the first (weakly augmented) view
into soft labels, and each student projection learns to match them against the
teacher projection of the other view.

References:
    Weng et al. "Clustering Properties of Self-Supervised Learning." ICML 2025.
    https://arxiv.org/abs/2501.18452

Example::

    from stable_pretraining.methods import ReSA
    import lightning as pl

    model = ReSA(encoder_name="resnet18", low_resolution=True)

    trainer = pl.Trainer(max_epochs=1000)
    trainer.fit(model, dataloader)
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import TeacherStudentWrapper, from_timm
from stable_pretraining.losses import ReSALoss


@dataclass
class ReSAOutput(ModelOutput):
    """Output from ReSA forward pass.

    :ivar loss: ReSA loss (0 in eval mode)
    :ivar embedding: Teacher backbone features [B, D] in eval mode, [2B, D] in
        train mode (always detached)
    :ivar projection: Student projector outputs [2B, P] (None in eval mode)
    :ivar assignment: Soft labels [B, B] built from the first view (None in
        eval mode)
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    projection: Optional[torch.Tensor] = None
    assignment: Optional[torch.Tensor] = None


def _build_resa_projector(in_dim: int, dims: Sequence[int]) -> nn.Module:
    """ReSA projector: Linear -> BN -> ReLU -> ... -> Linear -> BN(affine=False).

    :param in_dim: Backbone output dimension
    :param dims: Hidden + output dimensions, e.g. (2048, 2048, 512)
    """
    if len(dims) < 1:
        raise ValueError(
            "projector_dims must contain at least one entry (the output dim)"
        )
    layers = []
    prev = in_dim
    for i, dim in enumerate(dims):
        layers.append(nn.Linear(prev, dim))
        if i == len(dims) - 1:
            layers.append(nn.BatchNorm1d(dim, affine=False))
        else:
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
        prev = dim
    return nn.Sequential(*layers)


class ReSA(Module):
    """ReSA self-supervised learning with an EMA teacher and Sinkhorn soft labels.

    Architecture:
        - **Backbone**: any feature extractor producing a flat [B, D] embedding,
          wrapped with an EMA teacher
        - **Projector**: MLP ending in a BatchNorm without affine parameters,
          also wrapped with an EMA teacher
        - **Loss**: :class:`~stable_pretraining.losses.ReSALoss`

    :param encoder_name: timm model name (e.g. ``"resnet18"``,
        ``"vit_small_patch16_224"``) or a pre-instantiated ``nn.Module`` whose
        ``forward`` returns a ``[B, D]`` tensor.
    :param projector_dims: Hidden + output dimensions of the projector.
        ``(2048, 2048, 512)`` matches the official CIFAR recipe.
    :param temperature: Temperature of the cross-view predictions (0.4 in the
        paper).
    :param sinkhorn_temperature: Temperature of the similarities that build the
        soft labels (0.05 in the paper).
    :param sinkhorn_iterations: Sinkhorn-Knopp normalization rounds (3 in the
        paper).
    :param ema_decay_start: Initial teacher EMA decay (0.996 in the paper).
    :param ema_decay_end: Final teacher EMA decay (1.0).
    :param low_resolution: Adapt the first conv for 32x32 inputs (CIFAR-style).
    :param pretrained: Load pretrained timm weights for the encoder.

    Note:
        Pass the weakly augmented view (crop and flip only) as ``view1``: the
        soft labels are built from it. Use
        :class:`~stable_pretraining.callbacks.TeacherStudentCallback` to drive
        the teacher EMA updates during training. It updates the teacher every
        step but recomputes the momentum once per epoch, so the momentum lags
        the official code's per-iteration schedule by at most one epoch.

    Example::

        model = ReSA(encoder_name="resnet18", low_resolution=True)

        weak = torch.randn(256, 3, 32, 32)
        strong = torch.randn(256, 3, 32, 32)
        out = model(weak, strong)
        out.loss.backward()

        # eval: single view, no loss
        model.eval()
        out = model(weak)
        features = out.embedding  # [256, embed_dim], from the teacher
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "resnet18",
        projector_dims: Sequence[int] = (2048, 2048, 512),
        temperature: float = 0.4,
        sinkhorn_temperature: float = 0.05,
        sinkhorn_iterations: int = 3,
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 1.0,
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()

        if isinstance(encoder_name, str):
            base_backbone = from_timm(
                encoder_name,
                num_classes=0,
                low_resolution=low_resolution,
                pretrained=pretrained,
            )
        else:
            base_backbone = encoder_name

        with torch.no_grad():
            embed_dim = base_backbone(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.embed_dim = embed_dim

        self.backbone = TeacherStudentWrapper(
            base_backbone,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.projector = TeacherStudentWrapper(
            _build_resa_projector(embed_dim, list(projector_dims)),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.resa_loss = ReSALoss(
            temperature=temperature,
            sinkhorn_temperature=sinkhorn_temperature,
            sinkhorn_iterations=sinkhorn_iterations,
        )

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> ReSAOutput:
        """Forward pass.

        :param view1: Weakly augmented view [B, C, H, W] (or single view at eval).
        :param view2: Strongly augmented view [B, C, H, W]. If ``None``, returns
            only the teacher backbone embedding (eval mode).
        :return: :class:`ReSAOutput`.
        """
        if view2 is None:
            with torch.no_grad():
                embedding = self.backbone.forward_teacher(view1)
            return ReSAOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding.detach(),
                projection=None,
                assignment=None,
            )

        # One backbone call per view keeps BatchNorm statistics per view.
        views = (view1, view2)
        student_features = [self.backbone.forward_student(v) for v in views]
        student_proj = [self.projector.forward_student(h) for h in student_features]
        with torch.no_grad():
            teacher_features = [self.backbone.forward_teacher(v) for v in views]
            teacher_proj = [self.projector.forward_teacher(h) for h in teacher_features]

        assignment = self.resa_loss.assignment(student_features[0], teacher_features[0])
        loss = self.resa_loss(
            student_features[0],
            teacher_features[0],
            student_proj,
            teacher_proj,
            assignment=assignment,
        )
        return ReSAOutput(
            loss=loss,
            embedding=torch.cat(teacher_features, dim=0).detach(),
            projection=torch.cat(student_proj, dim=0),
            assignment=assignment,
        )
