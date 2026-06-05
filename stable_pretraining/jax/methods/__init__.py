"""SSL method classes for the JAX backend (mirror of :mod:`stable_pretraining.methods`)."""

from .byol import BYOL
from .joint_embedding import BarlowTwins, SimSiam, VICReg
from .simclr import SimCLR

__all__ = ["SimCLR", "VICReg", "BarlowTwins", "SimSiam", "BYOL"]
