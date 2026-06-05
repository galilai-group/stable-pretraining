"""Online linear-probe callback for the JAX backend.

JAX port of :class:`stable_pretraining.callbacks.OnlineProbe`. It trains a
small probe (typically a single ``nnx.Linear``) on top of *detached* backbone
embeddings during the main training run, then reports top-1 accuracy on the
validation set. The probe owns its own parameters and optax optimizer, so it
never perturbs the backbone — the JAX analogue of the torch version's
callback-parameter exclusion, here for free via state isolation.
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from ..optim import create_optimizer
from ..trainer import Callback, Trainer


@nnx.jit
def _probe_train_step(probe, optimizer, x, y):
    """One probe optimization step on detached features ``x`` with labels ``y``."""

    def loss_fn(p):
        logits = p(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(probe)
    optimizer.update(probe, grads)
    return loss


@nnx.jit
def _probe_predict(probe, x):
    """Probe logits for features ``x`` (no update)."""
    return probe(x)


class OnlineProbe(Callback):
    """Train a probe on detached embeddings and report validation accuracy.

    Args:
        name: Metric name stem (logged as ``train/<name>_loss`` and
            ``val/<name>_acc``).
        probe: An ``nnx.Module`` mapping embeddings ``[N, D]`` to logits
            ``[N, num_classes]`` (e.g. ``nnx.Linear(D, num_classes, rngs=...)``).
        input: Key in the forward-output dict holding the embeddings.
            Default ``"embedding"``.
        target: Key holding integer labels. Default ``"label"``.
        optim: Optimizer spec for the probe (see
            :func:`stable_pretraining.jax.optim.create_optimizer`).

    Attributes:
        accuracy: Top-1 accuracy from the most recent validation epoch.
    """

    def __init__(
        self,
        name: str,
        probe: nnx.Module,
        input: str = "embedding",
        target: str = "label",
        optim="adamw",
    ):
        self.name = name
        self.probe = probe
        self.input = input
        self.target = target
        self.optimizer = nnx.Optimizer(probe, create_optimizer(optim), wrt=nnx.Param)
        self.accuracy: float = float("nan")
        self._correct = 0
        self._total = 0

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.input not in outputs or self.target not in outputs:
            return
        # Detach: the probe must not push gradients into the backbone.
        x = jax.lax.stop_gradient(outputs[self.input])
        y = outputs[self.target]
        loss = _probe_train_step(self.probe, self.optimizer, x, y)
        trainer.log(f"train/{self.name}_loss", float(loss))

    def on_validation_epoch_start(self, trainer, module):
        self._correct = 0
        self._total = 0

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.input not in outputs or self.target not in outputs:
            return
        logits = _probe_predict(self.probe, outputs[self.input])
        preds = jnp.argmax(logits, axis=-1)
        self._correct += int(jnp.sum(preds == outputs[self.target]))
        self._total += int(outputs[self.target].shape[0])

    def on_validation_epoch_end(self, trainer: Trainer, module):
        if self._total > 0:
            self.accuracy = self._correct / self._total
            # ``eval/`` prefix matches the torch OnlineProbe metric key.
            trainer.log(f"eval/{self.name}_acc", self.accuracy)
