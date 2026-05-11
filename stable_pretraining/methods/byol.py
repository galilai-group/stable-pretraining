"""BYOL: Bootstrap Your Own Latent.

Self-supervised learning without negative pairs. The student network has a
backbone, a projector, and a predictor; the teacher (EMA of student) has
only a backbone and projector. The student predicts the teacher's projection
of the other view; the loss is symmetric MSE between L2-normalised vectors.

References:
    Grill et al. "Bootstrap Your Own Latent: A New Approach to
    Self-Supervised Learning." NeurIPS 2020.
    https://arxiv.org/abs/2006.07733
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.backbone import TeacherStudentWrapper, from_timm
from stable_pretraining.losses import BYOLLoss


@dataclass
class BYOLOutput(ModelOutput):
    """Output from BYOL forward pass.

    :ivar loss: Symmetric BYOL loss (0 in eval mode)
    :ivar embedding: Teacher backbone features (always detached)
    :ivar prediction: Student predictor output [2B, P] (None in eval)
    :ivar target: Teacher projector output [2B, P] (None in eval, detached)
    """

    loss: torch.Tensor = None
    embedding: torch.Tensor = None
    prediction: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None


def _byol_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    """2-layer Linear-BN-ReLU-Linear MLP used for projector and predictor."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class BYOL(Module):
    """BYOL self-supervised learning with EMA target network.

    :param encoder_name: timm model name or pre-built ``nn.Module``.
    :param projector_dims: ``(hidden, output)`` for the 2-layer projector
        (default ``(4096, 256)``).
    :param predictor_dims: ``(hidden, output)`` for the predictor
        (default ``(4096, 256)``).
    :param ema_decay_start: Initial EMA decay (default 0.99).
    :param ema_decay_end: Final EMA decay (default 1.0).
    :param low_resolution: Adapt first conv for low-res input.
    :param pretrained: Load pretrained timm weights.

    Note:
        Use :class:`~stable_pretraining.callbacks.TeacherStudentCallback`
        to drive teacher EMA updates during training.
    """

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (4096, 256),
        predictor_dims: Sequence[int] = (4096, 256),
        ema_decay_start: float = 0.99,
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

        if len(projector_dims) != 2 or len(predictor_dims) != 2:
            raise ValueError(
                "projector_dims and predictor_dims must be (hidden, output) tuples"
            )
        proj_hidden, proj_out = projector_dims
        pred_hidden, pred_out = predictor_dims
        if pred_out != proj_out:
            raise ValueError(
                f"predictor output dim ({pred_out}) must match projector output dim ({proj_out})"
            )

        self.backbone = TeacherStudentWrapper(
            base_backbone,
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.projector = TeacherStudentWrapper(
            _byol_mlp(embed_dim, proj_hidden, proj_out),
            warm_init=True,
            base_ema_coefficient=ema_decay_start,
            final_ema_coefficient=ema_decay_end,
        )
        self.predictor = _byol_mlp(proj_out, pred_hidden, pred_out)
        self.byol_loss = BYOLLoss()

    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
    ) -> BYOLOutput:
        if view2 is None:
            with torch.no_grad():
                embedding = self.backbone.forward_teacher(view1)
            return BYOLOutput(
                loss=torch.zeros((), device=embedding.device, dtype=embedding.dtype),
                embedding=embedding.detach(),
                prediction=None,
                target=None,
            )

        # Student path: backbone -> projector -> predictor
        s1 = self.backbone.forward_student(view1)
        s2 = self.backbone.forward_student(view2)
        zs1 = self.projector.forward_student(s1)
        zs2 = self.projector.forward_student(s2)
        p1 = self.predictor(zs1)
        p2 = self.predictor(zs2)

        # Teacher path: detached, EMA target
        with torch.no_grad():
            t1 = self.backbone.forward_teacher(view1)
            t2 = self.backbone.forward_teacher(view2)
            zt1 = self.projector.forward_teacher(t1)
            zt2 = self.projector.forward_teacher(t2)

        loss = (self.byol_loss(p1, zt2) + self.byol_loss(p2, zt1)) / 2
        return BYOLOutput(
            loss=loss,
            embedding=torch.cat([t1, t2], dim=0).detach(),
            prediction=torch.cat([p1, p2], dim=0),
            target=torch.cat([zt1, zt2], dim=0).detach(),
        )
