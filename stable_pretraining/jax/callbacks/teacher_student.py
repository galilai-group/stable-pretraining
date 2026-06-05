"""EMA teacher-update callback for the JAX backend.

JAX port of :class:`stable_pretraining.callbacks.TeacherStudentCallback`. After
each training batch it EMA-updates every
:class:`~stable_pretraining.jax.backbone.TeacherStudentWrapper` found on the
module; each epoch it advances the cosine EMA schedule. Runs in
``on_train_batch_end`` (after the optimizer step), matching the torch ordering.
"""

import jax.numpy as jnp

from ..backbone.teacher_student import TeacherStudentWrapper, ema_update
from ..trainer import Callback


def _find_wrappers(module):
    """Collect all TeacherStudentWrapper submodules (depth-first)."""
    found = []

    def rec(m):
        from flax import nnx

        for value in vars(m).values():
            if isinstance(value, TeacherStudentWrapper):
                found.append(value)
            elif isinstance(value, nnx.Module):
                rec(value)
            elif isinstance(value, (list, tuple, nnx.List)):
                for item in value:
                    if isinstance(item, nnx.Module):
                        rec(item)

    rec(module)
    return found


class TeacherStudentCallback(Callback):
    """EMA-update teacher networks after each step; cosine-schedule the coefficient.

    Args:
        update_schedule: If True, advance each wrapper's EMA coefficient on a
            cosine schedule from ``base`` to ``final`` over ``trainer.max_epochs``.
            Default True.
    """

    def __init__(self, update_schedule: bool = True):
        self.update_schedule = update_schedule

    def on_train_epoch_start(self, trainer, module):
        if not self.update_schedule:
            return
        for wrapper in _find_wrappers(module):
            wrapper.update_ema_coefficient(trainer.current_epoch, trainer.max_epochs)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        for wrapper in _find_wrappers(module):
            # Pass a JAX scalar so a cosine-scheduled coefficient doesn't recompile.
            ema_update(wrapper, jnp.asarray(wrapper.ema, dtype=jnp.float32))
