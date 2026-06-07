"""SimCLR method class for the JAX backend.

Batteries-included counterpart to :func:`stable_pretraining.jax.forward.simclr`,
mirroring :class:`stable_pretraining.methods.SimCLR`. It wires a backbone, an
MLP projector, and the NT-Xent loss into a :class:`stable_pretraining.jax.Module`.
"""

from typing import Sequence

from flax import nnx

from ..backbone import MLP
from ..forward import simclr
from ..losses import NTXEntLoss
from ..module import Module


class SimCLR(Module):
    """SimCLR: contrastive joint-embedding SSL on the JAX backend.

    Args:
        backbone: An ``nnx.Module`` producing flat ``[B, D]`` embeddings.
        embed_dim: Backbone output dimension ``D`` (projector input).
        rngs: NNX RNG collection used to initialize the projector.
        projector_dims: Hidden + output dims of the MLP projector.
            Default ``(2048, 2048, 256)`` (original SimCLR ResNet50 recipe).
        temperature: NT-Xent temperature. Default ``0.5``.
        optim: Optimizer spec (see
            :func:`stable_pretraining.jax.optim.create_optimizer`).

    Example:
        >>> import jax  # doctest: +SKIP
        >>> from flax import nnx  # doctest: +SKIP
        >>> import stable_pretraining.jax as spj  # doctest: +SKIP
        >>> rngs = nnx.Rngs(0)  # doctest: +SKIP
        >>> backbone = spj.backbone.MLP(64, [128], rngs=rngs)  # doctest: +SKIP
        >>> model = spj.SimCLR(  # doctest: +SKIP
        ...     backbone=backbone, embed_dim=128, rngs=rngs, temperature=0.5
        ... )
    """

    def __init__(
        self,
        *,
        backbone: nnx.Module,
        embed_dim: int,
        rngs: nnx.Rngs,
        projector_dims: Sequence[int] = (2048, 2048, 256),
        temperature: float = 0.5,
        optim="adamw",
        transform=None,
        aug_seed: int = 0,
        dtype=None,
    ):
        projector = MLP(embed_dim, list(projector_dims), rngs=rngs, dtype=dtype)
        super().__init__(
            forward=simclr,
            optim=optim,
            transform=transform,
            aug_seed=aug_seed,
            backbone=backbone,
            projector=projector,
            simclr_loss=NTXEntLoss(temperature=temperature),
        )
