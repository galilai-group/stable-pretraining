"""BYOL method class for the JAX backend :cite:`grill2020bootstrap`.

Wires an EMA teacher-student backbone + projector, an online predictor, and the
BYOL loss into a :class:`stable_pretraining.jax.Module`. Pair it with
:class:`stable_pretraining.jax.callbacks.TeacherStudentCallback` so the teacher
is EMA-updated each step.
"""

from typing import Sequence

from flax import nnx

from ..backbone import MLP
from ..backbone.teacher_student import TeacherStudentWrapper
from ..forward import byol
from ..losses import BYOLLoss
from ..module import Module


class BYOL(Module):
    """BYOL: bootstrap-your-own-latent SSL with an EMA teacher.

    Args:
        backbone: ``nnx.Module`` producing ``[B, embed_dim]`` features (wrapped
            in a teacher-student wrapper internally).
        embed_dim: Backbone output dimension.
        rngs: NNX RNG collection for projector/predictor.
        projector_dims: Hidden + output dims of the MLP projector.
        predictor_hidden: Hidden width of the online predictor MLP.
        base_ema/final_ema: Cosine EMA schedule endpoints for the teacher.
        optim: Optimizer spec.

    Note:
        Requires ``TeacherStudentCallback`` in the trainer to perform the EMA
        update; without it the teacher stays at initialization.
    """

    def __init__(
        self,
        *,
        backbone: nnx.Module,
        embed_dim: int,
        rngs: nnx.Rngs,
        projector_dims: Sequence[int] = (4096, 256),
        predictor_hidden: int = 4096,
        base_ema: float = 0.996,
        final_ema: float = 1.0,
        optim="adamw",
    ):
        proj_out = projector_dims[-1]
        super().__init__(
            forward=byol,
            optim=optim,
            backbone=TeacherStudentWrapper(backbone, base_ema, final_ema),
            projector=TeacherStudentWrapper(
                MLP(embed_dim, list(projector_dims), rngs=rngs), base_ema, final_ema
            ),
            predictor=MLP(proj_out, [predictor_hidden, proj_out], rngs=rngs),
            byol_loss=BYOLLoss(),
        )
