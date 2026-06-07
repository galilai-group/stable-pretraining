"""EMA teacher-student wrapper for the JAX backend.

JAX/NNX port of :class:`stable_pretraining.backbone.utils.TeacherStudentWrapper`.
It holds a trainable ``student`` and an EMA ``teacher`` copy and exposes
``forward_student`` / ``forward_teacher`` so BYOL/DINO/MoCo-style forwards can
address each network.

The teacher's parameters are stored as :class:`EMAParam` (a non-``Param``
Variable, like ``BatchStat``). This is the crux of the NNX design: because the
optimizer is built with ``wrt=nnx.Param`` and ``nnx.value_and_grad`` defaults to
differentiating ``Param``, the teacher is automatically excluded from **both**
gradient and the optimizer step — it changes *only* through :meth:`update_teacher`.
"""

import copy
import math

import jax
from flax import nnx


class EMAParam(nnx.Variable):
    """Teacher parameter — a non-``Param`` Variable, excluded from grad/optimizer."""


def _iter_modules(module):
    """Yield ``module`` and all nested NNX submodules (incl. inside nnx.List)."""
    yield module
    for value in vars(module).values():
        if isinstance(value, nnx.Module):
            yield from _iter_modules(value)
        elif isinstance(value, (list, tuple, nnx.List)):
            for item in value:
                if isinstance(item, nnx.Module):
                    yield from _iter_modules(item)


def _freeze_params_to_ema(module: nnx.Module) -> None:
    """Convert every ``nnx.Param`` in ``module`` to an :class:`EMAParam` in place."""
    for m in _iter_modules(module):
        for name, value in list(vars(m).items()):
            if type(value) is nnx.Param:
                setattr(m, name, EMAParam(value[...]))


class TeacherStudentWrapper(nnx.Module):
    """Wrap a network with an EMA-updated teacher copy.

    Args:
        student: The trainable network (its params stay ``nnx.Param``).
        base_ema_coefficient: EMA decay at the start of training (cosine-scheduled
            up to ``final_ema_coefficient``). ``0`` ⇒ teacher tracks student
            exactly; ``1`` ⇒ teacher frozen. Default ``0.996``.
        final_ema_coefficient: EMA decay at the end of training. Default ``1.0``.

    Attributes:
        ema: The current EMA coefficient (updated by ``TeacherStudentCallback``).
    """

    def __init__(
        self,
        student: nnx.Module,
        base_ema_coefficient: float = 0.996,
        final_ema_coefficient: float = 1.0,
    ):
        self.student = student
        # Teacher starts as an exact copy, then its params become EMAParam so
        # the optimizer/grad ignore them.
        self.teacher = copy.deepcopy(student)
        _freeze_params_to_ema(self.teacher)
        self.base_ema_coefficient = base_ema_coefficient
        self.final_ema_coefficient = final_ema_coefficient
        self.ema = base_ema_coefficient

    def forward_student(self, *args, **kwargs):
        """Forward through the student (gradients flow)."""
        return self.student(*args, **kwargs)

    def forward_teacher(self, *args, **kwargs):
        """Forward through the teacher (gradient-stopped)."""
        return jax.lax.stop_gradient(self.teacher(*args, **kwargs))

    def __call__(self, *args, **kwargs):
        """Default forward uses the teacher (common for evaluation)."""
        return self.forward_teacher(*args, **kwargs)

    def update_teacher(self, ema) -> None:
        """EMA step: ``teacher = ema * teacher + (1 - ema) * student`` (params + BN stats).

        Mutates the teacher's :class:`EMAParam` and ``BatchStat`` variables in
        place (matching torch, which EMAs buffers too). RNG/other state is left
        untouched. ``ema`` may be a Python float or a JAX scalar (pass a scalar
        to avoid recompiles when the coefficient is cosine-scheduled).
        """
        teacher_state = dict(nnx.state(self.teacher).flat_state())
        student_state = dict(nnx.state(self.student).flat_state())
        for path, tvar in teacher_state.items():
            if not isinstance(tvar, (EMAParam, nnx.BatchStat)):
                continue
            svar = student_state.get(path)
            if svar is not None:
                tvar[...] = ema * tvar[...] + (1.0 - ema) * svar[...]

    def update_ema_coefficient(self, epoch: int, total_epochs: int) -> None:
        """Set ``self.ema`` from a cosine schedule (base → final over training)."""
        # Python float (not jnp) so ``self.ema`` stays a static attribute.
        t = epoch / max(total_epochs, 1)
        self.ema = self.final_ema_coefficient - 0.5 * (
            self.final_ema_coefficient - self.base_ema_coefficient
        ) * (1 + math.cos(math.pi * t))


@nnx.jit
def ema_update(wrapper: TeacherStudentWrapper, ema) -> None:
    """jit-compiled in-place EMA update of ``wrapper``'s teacher."""
    wrapper.update_teacher(ema)


__all__ = ["EMAParam", "TeacherStudentWrapper", "ema_update"]
