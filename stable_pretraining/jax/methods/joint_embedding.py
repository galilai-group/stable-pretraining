"""VICReg and Barlow Twins method classes for the JAX backend.

Batteries-included counterparts to the forward functions in
:mod:`stable_pretraining.jax.forward`, mirroring the torch method classes.
Both wire a backbone + MLP projector + their loss into a
:class:`stable_pretraining.jax.Module`.
"""

from typing import Sequence

from flax import nnx

from ..backbone import MLP
from ..forward import barlow_twins, simsiam, vicreg
from ..losses import BarlowTwinsLoss, NegativeCosineSimilarity, VICRegLoss
from ..module import Module


class VICReg(Module):
    """VICReg: variance-invariance-covariance SSL on the JAX backend.

    Args:
        backbone: ``nnx.Module`` producing ``[B, embed_dim]`` features.
        embed_dim: Backbone output dimension.
        rngs: NNX RNG collection for the projector.
        projector_dims: Hidden + output dims of the MLP projector.
        sim_coeff/std_coeff/cov_coeff: VICReg loss weights.
        optim: Optimizer spec.
    """

    def __init__(
        self,
        *,
        backbone: nnx.Module,
        embed_dim: int,
        rngs: nnx.Rngs,
        projector_dims: Sequence[int] = (2048, 2048, 2048),
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        optim="adamw",
    ):
        super().__init__(
            forward=vicreg,
            optim=optim,
            backbone=backbone,
            projector=MLP(embed_dim, list(projector_dims), rngs=rngs),
            vicreg_loss=VICRegLoss(
                sim_coeff=sim_coeff, std_coeff=std_coeff, cov_coeff=cov_coeff
            ),
        )


class BarlowTwins(Module):
    """Barlow Twins: cross-correlation SSL on the JAX backend.

    Args:
        backbone: ``nnx.Module`` producing ``[B, embed_dim]`` features.
        embed_dim: Backbone output dimension.
        rngs: NNX RNG collection for the projector.
        projector_dims: Hidden + output dims of the MLP projector.
        lambd: Off-diagonal redundancy weight.
        optim: Optimizer spec.
    """

    def __init__(
        self,
        *,
        backbone: nnx.Module,
        embed_dim: int,
        rngs: nnx.Rngs,
        projector_dims: Sequence[int] = (2048, 2048, 2048),
        lambd: float = 5e-3,
        optim="adamw",
    ):
        super().__init__(
            forward=barlow_twins,
            optim=optim,
            backbone=backbone,
            projector=MLP(embed_dim, list(projector_dims), rngs=rngs),
            barlow_loss=BarlowTwinsLoss(lambd=lambd),
        )


class SimSiam(Module):
    """SimSiam: stop-gradient predictor SSL on the JAX backend :cite:`chen2021exploring`.

    Args:
        backbone: ``nnx.Module`` producing ``[B, embed_dim]`` features.
        embed_dim: Backbone output dimension.
        rngs: NNX RNG collection for the projector/predictor.
        projector_dims: Hidden + output dims of the MLP projector.
        predictor_dim: Bottleneck hidden width of the predictor MLP.
        optim: Optimizer spec.
    """

    def __init__(
        self,
        *,
        backbone: nnx.Module,
        embed_dim: int,
        rngs: nnx.Rngs,
        projector_dims: Sequence[int] = (2048, 2048, 2048),
        predictor_dim: int = 512,
        optim="adamw",
    ):
        proj_out = projector_dims[-1]
        super().__init__(
            forward=simsiam,
            optim=optim,
            backbone=backbone,
            projector=MLP(embed_dim, list(projector_dims), rngs=rngs),
            predictor=MLP(proj_out, [predictor_dim, proj_out], rngs=rngs),
            simsiam_loss=NegativeCosineSimilarity(),
        )
