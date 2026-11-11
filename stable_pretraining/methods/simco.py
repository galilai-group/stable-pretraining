"""SimCo: symmetric contrastive learning with dual temperatures (CVPR 2022)."""

from typing import Any

from torch import nn

from stable_pretraining import Module
from stable_pretraining.forward import simco
from stable_pretraining.losses import SimCoLoss


class SimCo(Module):
    """Compose an encoder and projector with the two-view SimCo objective.

    Args:
        backbone: Encoder returning one feature vector per image.
        projector: Projection head; the loss normalizes its outputs per row.
        intra_temperature: Positive temperature for within-anchor hardness.
        inter_temperature: Positive temperature for between-anchor hardness.
        optim: Standard ``spt.Module`` optimizer/scheduler configuration.
            If omitted, use Module's default AdamW configuration.
        hparams: Additional serializable experiment metadata.

    Note:
        SimCo uses a shared encoder with gradients through both views. It does
        not require a predictor, momentum encoder, or queue. The paper's CIFAR
        experiments use a two-layer projector with ReLU between the layers.
        Labels are used only by evaluation callbacks.
    """

    def __init__(
        self,
        backbone: nn.Module,
        projector: nn.Module,
        intra_temperature: float = 0.1,
        inter_temperature: float = 1.0,
        optim: dict[str, Any] | None = None,
        hparams: dict[str, Any] | None = None,
    ) -> None:
        metadata = {
            **(hparams or {}),
            "method": "simco",
            "intra_temperature": intra_temperature,
            "inter_temperature": inter_temperature,
        }
        optimizer_options = {} if optim is None else {"optim": optim}
        super().__init__(
            forward=simco,
            backbone=backbone,
            projector=projector,
            simco_loss=SimCoLoss(intra_temperature, inter_temperature),
            hparams=metadata,
            **optimizer_options,
        )
