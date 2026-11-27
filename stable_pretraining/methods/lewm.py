"""LeWorldModel (LeWM)."""

from torch import nn

from stable_pretraining.module import Module
from stable_pretraining.forward import lewm_forward
from stable_pretraining.losses.world_model import LeWMLoss
from stable_pretraining.methods.lewm_components.architecture import SIGReg


class LeWM(Module):
    """Stable Pretraining implementation of LeWorldModel."""

    def __init__(
        self,
        model: nn.Module,
        history_size: int,
        num_preds: int,
        sigreg_weight: float = 0.1,
        sigreg: nn.Module | None = None,
        optim: dict | None = None,
    ):
        super().__init__(
            forward=lewm_forward,
            model=model,
            loss_fn=LeWMLoss(
                sigreg=sigreg if sigreg is not None else SIGReg(),
                sigreg_weight=sigreg_weight,
            ),
            history_size=history_size,
            num_preds=num_preds,
            optim=optim,
        )
