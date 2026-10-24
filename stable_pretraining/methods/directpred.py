"""DirectPred: BYOL with a directly set linear predictor.

Non-contrastive self-supervised learning in the BYOL setting (online network,
EMA target network, symmetric normalized MSE), where the linear predictor is
not trained by gradient descent but set in closed form from the statistics of
its inputs:

    F <- rho * F + (1 - rho) * E[f f^T]          (Eq. 19, uncentered)
    F  = U diag(s) U^T                           (eigendecomposition)
    W_p = U diag(p) U^T,  p_j = sqrt(s_j / max_k s_k) + eps    (Eq. 18)

References:
    Tian, Chen, Ganguli. "Understanding self-supervised learning dynamics
    without contrastive pairs." ICML 2021.
    https://arxiv.org/abs/2102.06810
"""

from typing import Sequence, Union

import torch
import torch.nn as nn

from stable_pretraining import Module
from stable_pretraining.backbone import (
    DirectPredictor,
    TeacherStudentWrapper,
    from_timm,
)
from stable_pretraining.losses import BYOLLoss
from stable_pretraining.methods.byol import BYOLOutput, _byol_mlp


class DirectPred(Module):
    """DirectPred: BYOL whose linear predictor is set in closed form.

    The paper's CIFAR-10 recipe uses a ResNet-18 with
    ``encoder_name="resnet18", low_resolution=True``, ``projector_dims=(512,
    128)`` and a constant EMA decay of 0.996.

    :param encoder_name: timm model name or pre-built ``nn.Module``.
    :param projector_dims: ``(hidden, output)`` for the 2-layer projector
        (default ``(4096, 256)``, as in BYOL).
    :param ema_decay_start: Initial EMA decay (default 0.99).
    :param ema_decay_end: Final EMA decay (default 1.0).
    :param rho: Weight of the previous correlation estimate (default 0.3).
    :param eps: Boost of the small eigenvalues, Eq. 18 (default 0.1).
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
        ema_decay_start: float = 0.99,
        ema_decay_end: float = 1.0,
        rho: float = 0.3,
        eps: float = 0.1,
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

        if len(projector_dims) != 2:
            raise ValueError("projector_dims must be a (hidden, output) tuple")
        proj_hidden, proj_out = projector_dims

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
        self.predictor = DirectPredictor(proj_out, rho=rho, eps=eps)
        self.byol_loss = BYOLLoss()

    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor = None,
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

        # Online path: backbone -> projector -> directly set predictor
        zs1 = self.projector.forward_student(self.backbone.forward_student(view1))
        zs2 = self.projector.forward_student(self.backbone.forward_student(view2))
        p1 = self.predictor(zs1)
        p2 = self.predictor(zs2)

        # Target path: detached, EMA target
        with torch.no_grad():
            t1 = self.backbone.forward_teacher(view1)
            t2 = self.backbone.forward_teacher(view2)
            zt1 = self.projector.forward_teacher(t1)
            zt2 = self.projector.forward_teacher(t2)

        loss = (self.byol_loss(p1, zt2) + self.byol_loss(p2, zt1)) / 2

        # Refresh the predictor from this step's projections (used next step)
        if self.training:
            self.predictor.update(torch.cat([zs1, zs2], dim=0))

        return BYOLOutput(
            loss=loss,
            embedding=torch.cat([t1, t2], dim=0).detach(),
            prediction=torch.cat([p1, p2], dim=0),
            target=torch.cat([zt1, zt2], dim=0).detach(),
        )
