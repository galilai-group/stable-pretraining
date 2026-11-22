"""RINCE: Robust InfoNCE for contrastive learning."""

from typing import Sequence, Union

import torch.nn as nn

from stable_pretraining.losses import RINCELoss
from stable_pretraining.methods.simclr import SimCLR


class RINCE(SimCLR):
    """SimCLR architecture with the Robust InfoNCE objective."""

    def __init__(
        self,
        encoder_name: Union[str, nn.Module] = "vit_small_patch16_224",
        projector_dims: Sequence[int] = (2048, 2048, 256),
        temperature: float = 0.5,
        q: float = 1.0,
        lam: float = 0.5,
        low_resolution: bool = False,
        pretrained: bool = False,
    ):
        super().__init__(
            encoder_name=encoder_name,
            projector_dims=projector_dims,
            temperature=temperature,
            low_resolution=low_resolution,
            pretrained=pretrained,
        )

        self.simclr_loss = RINCELoss(
            temperature=temperature,
            q=q,
            lam=lam,
        )
