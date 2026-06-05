"""Loss functions for the JAX backend (jnp ports of :mod:`stable_pretraining.losses`)."""

from .joint_embedding import (
    BarlowTwinsLoss,
    BYOLLoss,
    InfoNCELoss,
    NegativeCosineSimilarity,
    NTXEntLoss,
    VICRegLoss,
    l2_normalize,
)
from .swav import SwAVLoss, sinkhorn
from .utils import off_diagonal, vcreg_loss

__all__ = [
    "NTXEntLoss",
    "InfoNCELoss",
    "NegativeCosineSimilarity",
    "BYOLLoss",
    "VICRegLoss",
    "BarlowTwinsLoss",
    "SwAVLoss",
    "sinkhorn",
    "l2_normalize",
    "off_diagonal",
    "vcreg_loss",
]
