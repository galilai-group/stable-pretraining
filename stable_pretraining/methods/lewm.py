"""LeWM: Latent Embedding World Model."""

from torch import nn

from stable_pretraining.module import Module
from stable_pretraining.forward import lewm_forward
from stable_pretraining.methods.lewm_components.architecture import SIGReg


class LeWM(Module):
    """Stable Pretraining implementation of LeWM."""

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
            sigreg=sigreg if sigreg is not None else SIGReg(),
            history_size=history_size,
            num_preds=num_preds,
            sigreg_weight=sigreg_weight,
            optim=optim,
        )
