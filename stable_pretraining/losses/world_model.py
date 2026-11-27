"""LeWorldModel training objective."""

import torch
from torch import nn


class LeWMLoss(nn.Module):
    """Combine prediction MSE and SIGReg regularization."""

    def __init__(
        self,
        sigreg: nn.Module,
        sigreg_weight: float = 0.1,
    ):
        super().__init__()
        self.sigreg = sigreg
        self.sigreg_weight = sigreg_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred_loss = (pred - target).pow(2).mean()
        sigreg_loss = self.sigreg(embeddings.transpose(0, 1))

        return {
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "loss": pred_loss + self.sigreg_weight * sigreg_loss,
        }
