"""Convenience module for FroSSL.

Example::

    import torch
    import stable_pretraining as spt

    model = spt.FroSSL(
        backbone=spt.backbone.from_torchvision("resnet18", low_resolution=True),
        projector=torch.nn.Linear(512, 1024),
        optim={"optimizer": {"type": "SGD", "lr": 0.03}},
        invariance_weight=1.4,
    )
"""

from typing import Any

import torch

from stable_pretraining import Module
from stable_pretraining.forward import frossl
from stable_pretraining.losses import FroSSLLoss


class FroSSL(Module):
    """Wire FroSSL components into the dictionary-based training interface.

    Args:
        backbone: Encoder producing one embedding per input image.
        projector: Head mapping embeddings to FroSSL projections.
        optim: Optimizer configuration accepted by :class:`Module`.
        invariance_weight: Weight of the view-invariance term.

    Note:
        The projector and optimizer are explicit so experiments can match
        the authors' settings without hidden architecture defaults.
        Introduced by Skean et al. :cite:`skean2024frossl`.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        projector: torch.nn.Module,
        optim: dict[str, Any],
        invariance_weight: float = 1.0,
    ) -> None:
        super().__init__(
            forward=frossl,
            hparams={"invariance_weight": invariance_weight},
            backbone=backbone,
            projector=projector,
            frossl_loss=FroSSLLoss(invariance_weight=invariance_weight),
            optim=optim,
        )
