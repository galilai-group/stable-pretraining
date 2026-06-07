"""Checkpointing for the JAX backend.

Serializes the state of any NNX objects (the :class:`Module`, its optimizer,
and eval-callback sub-modules like an ``OnlineProbe``'s probe) to a single
msgpack file via ``flax.serialization`` — no extra dependency beyond flax.
Saving the optimizer state alongside the model enables **exact** resume.

Example:
    >>> from stable_pretraining.jax import checkpoint  # doctest: +SKIP
    >>> checkpoint.save(
    ...     "run.msgpack", step=1000, module=model, optimizer=opt
    ... )  # doctest: +SKIP
    >>> step = checkpoint.load(
    ...     "run.msgpack", module=model, optimizer=opt
    ... )  # doctest: +SKIP
"""

import os
import tempfile
from pathlib import Path

import numpy as np
from flax import nnx, serialization

from .trainer import Callback


def _pure(obj: nnx.Module) -> dict:
    return nnx.state(obj).to_pure_dict()


def _atomic_write(path: Path, data: bytes) -> None:
    """Crash-safe write: temp file in the same dir + fsync + atomic rename.

    Mirrors :mod:`stable_pretraining.utils.atomic_checkpoint` — a SIGTERM
    between steps can never corrupt the previous checkpoint (the rename is
    atomic; a partial temp file is just discarded).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on POSIX (same filesystem)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def save(path, *, step: int = 0, **objects: nnx.Module) -> None:
    """Save the state of one or more NNX objects to ``path`` atomically.

    Args:
        path: Destination file.
        step: Training step/epoch counter stored alongside (returned by :func:`load`).
        **objects: Named NNX objects to checkpoint (e.g. ``module=...``,
            ``optimizer=...``, ``probe=...``).
    """
    payload = {name: _pure(obj) for name, obj in objects.items()}
    payload["_step"] = np.asarray(step, dtype=np.int64)
    _atomic_write(Path(path), serialization.to_bytes(payload))


def load(path, *, strict: bool = True, **objects: nnx.Module) -> int:
    """Restore NNX objects from ``path`` in place and return the saved step.

    Args:
        path: Checkpoint file written by :func:`save`.
        strict: Unused placeholder for API symmetry; restoration always matches
            by the live objects' structure.
        **objects: The same named NNX objects passed to :func:`save`, to be
            updated in place.

    Returns:
        int: The ``step`` value stored at save time.
    """
    template = {name: _pure(obj) for name, obj in objects.items()}
    template["_step"] = np.asarray(0, dtype=np.int64)
    payload = serialization.from_bytes(template, Path(path).read_bytes())
    for name, obj in objects.items():
        state = nnx.state(obj)
        state.replace_by_pure_dict(payload[name])
        nnx.update(obj, state)
    return int(payload["_step"])


class Checkpoint(Callback):
    """Save the module (and optimizer) during training — JAX analogue of ModelCheckpoint.

    Args:
        path: Destination file (overwritten each save).
        every_n_epochs: If set, also save at the end of every ``n``-th epoch;
            otherwise only at ``on_fit_end``.
        save_optimizer: Include optimizer state for exact resume. Default True.
    """

    def __init__(self, path, every_n_epochs=None, save_optimizer: bool = True):
        self.path = path
        self.every_n_epochs = every_n_epochs
        self.save_optimizer = save_optimizer

    def _save(self, trainer, module):
        objects = {"module": module}
        if self.save_optimizer and trainer.optimizer is not None:
            objects["optimizer"] = trainer.optimizer
        save(self.path, step=trainer.global_step, **objects)
        # Record the path in any logger that tracks it (e.g. RegistryLogger).
        if hasattr(trainer, "log_checkpoint_path"):
            trainer.log_checkpoint_path(self.path)

    def on_train_epoch_end(self, trainer, module):
        if (
            self.every_n_epochs
            and (trainer.current_epoch + 1) % self.every_n_epochs == 0
        ):
            self._save(trainer, module)

    def on_fit_end(self, trainer, module):
        self._save(trainer, module)


__all__ = ["save", "load", "Checkpoint"]
