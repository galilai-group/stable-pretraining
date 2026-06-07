"""JAX / Flax-NNX backend for stable-pretraining.

This subpackage is a parallel backend to the default torch/Lightning one. It
mirrors the same concepts — a stateless ``forward(self, batch, stage)`` bound to
a :class:`Module`, callbacks that read the forward-output dict, a
Lightning-style hook lifecycle — but the engine is JAX-native (functional
``nnx.value_and_grad`` + ``optax`` instead of ``loss.backward()`` +
``torch.optim``).

It is **opt-in and isolated**: ``import stable_pretraining`` never imports
jax/flax. Access it explicitly::

    import stable_pretraining.jax as spj

    backbone = spj.backbone.MLP(64, [128], rngs=nnx.Rngs(0))
    model = spj.SimCLR(backbone=backbone, embed_dim=128, rngs=nnx.Rngs(0))
    trainer = spj.Trainer(max_epochs=10, callbacks=[probe, rankme])
    trainer.fit(model, train_loader, val_loader)

Install the backend with ``pip install -e ".[jax]"`` (CPU) or a CUDA jaxlib
wheel for HPC.
"""

from . import augment, backbone, callbacks, checkpoint, forward, losses, optim, utils
from .backbone import TeacherStudentWrapper
from .callbacks import (
    EarlyStopping,
    LiDAR,
    OnlineKNN,
    OnlineProbe,
    OnlineQueue,
    OnlineWriter,
    RankMe,
    TeacherStudentCallback,
)
from .losses import (
    BarlowTwinsLoss,
    BYOLLoss,
    InfoNCELoss,
    NegativeCosineSimilarity,
    NTXEntLoss,
    SwAVLoss,
    VICRegLoss,
)
from .manager import Manager
from .methods import BYOL, BarlowTwins, SimCLR, SimSiam, VICReg
from .module import Module
from .optim import create_optimizer
from .trainer import Callback, Trainer

__all__ = [
    # sub-packages
    "augment",
    "backbone",
    "callbacks",
    "checkpoint",
    "forward",
    "losses",
    "optim",
    "utils",
    # core
    "Module",
    "Trainer",
    "Manager",
    "Callback",
    "create_optimizer",
    "TeacherStudentWrapper",
    # losses
    "NTXEntLoss",
    "InfoNCELoss",
    "NegativeCosineSimilarity",
    "BYOLLoss",
    "VICRegLoss",
    "BarlowTwinsLoss",
    "SwAVLoss",
    # methods
    "SimCLR",
    "VICReg",
    "BarlowTwins",
    "SimSiam",
    "BYOL",
    # callbacks
    "OnlineProbe",
    "OnlineKNN",
    "RankMe",
    "LiDAR",
    "OnlineQueue",
    "OnlineWriter",
    "EarlyStopping",
    "TeacherStudentCallback",
]
