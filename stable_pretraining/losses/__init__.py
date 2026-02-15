"""SSL Losses.

This module provides various self-supervised learning loss functions organized by category:
- DINO losses: Self-distillation methods (DINOLoss, iBOTPatchLoss)
- Joint embedding losses: Contrastive and non-contrastive methods (BYOL, VICReg, Barlow Twins, SimCLR)
- Reconstruction losses: Masked prediction methods (MAE)
- Utilities: Helper functions (sinkhorn_knopp, off_diagonal, NegativeCosineSimilarity)
"""

# DINO self-distillation losses
from .dino import DINOv1Loss, DINOv2Loss, iBOTPatchLoss

# Joint embedding losses
from .joint_embedding import (
    BYOLLoss,
    VICRegLoss,
    BarlowTwinsLoss,
    NTXEntLoss,
)

# Multimodal losses
from .multimodal import CLIPLoss

# Reconstruction losses
from .reconstruction import mae

# Utilities
from .utils import (
    sinkhorn_knopp,
    off_diagonal,
    NegativeCosineSimilarity,
    VCRegLoss,
)

__all__ = [
    # DINO
    "DINOv1Loss",
    "DINOv2Loss",
    "iBOTPatchLoss",
    # Joint embedding
    "BYOLLoss",
    "VICRegLoss",
    "BarlowTwinsLoss",
    "NTXEntLoss",
    "CLIPLoss",
    # Reconstruction
    "mae",
    # Utils
    "sinkhorn_knopp",
    "off_diagonal",
    "NegativeCosineSimilarity",
    "VCRegLoss",
]
