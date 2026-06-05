"""RankMe effective-rank callback for the JAX backend.

JAX port of :class:`stable_pretraining.callbacks.RankMe`. RankMe
:cite:`garrido2023rankme` estimates the effective rank of the embedding matrix
as the entropy of its normalized singular-value spectrum — a label-free proxy
for representation quality.
"""

import jax.numpy as jnp
import numpy as np

from ..trainer import Callback, Trainer

_EPS = 1e-7


def _rankme_components(embeddings: jnp.ndarray, eps: float = _EPS):
    """Return ``(rankme, entropy, top_singular_value)`` for an embedding matrix."""
    s = jnp.linalg.svd(embeddings, compute_uv=False)
    p = s / (jnp.sum(s) + eps) + eps
    entropy = -jnp.sum(p * jnp.log(p))
    return float(jnp.exp(entropy)), float(entropy), float(s[0])


def rankme(embeddings: jnp.ndarray, eps: float = _EPS) -> float:
    """Compute the RankMe effective rank of an embedding matrix.

    Args:
        embeddings: Array of shape ``[N, D]``.
        eps: Numerical-stability epsilon added to the normalized spectrum.

    Returns:
        float: ``exp(-sum_k p_k log p_k)`` where ``p_k`` is the L1-normalized
        singular-value spectrum of ``embeddings``.
    """
    return _rankme_components(embeddings, eps)[0]


class RankMe(Callback):
    """Accumulate validation embeddings and report their RankMe effective rank.

    Logs under the bare metric name ``<name>`` (matching the torch RankMe key);
    with ``verbose=True`` also logs ``<name>/entropy`` and
    ``<name>/top_singular_value``.

    Args:
        name: Metric name (logged as ``<name>``). Default ``"rankme"``.
        input: Forward-output key holding embeddings. Default ``"embedding"``.
        verbose: Also log entropy + top singular value. Default ``False``.

    Attributes:
        value: RankMe value from the most recent validation epoch.
    """

    def __init__(self, name: str = "rankme", input: str = "embedding", verbose=False):
        self.name = name
        self.input = input
        self.verbose = verbose
        self.value: float = float("nan")
        self._feats: list = []

    def on_validation_epoch_start(self, trainer, module):
        self._feats = []

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.input in outputs:
            self._feats.append(np.asarray(outputs[self.input]))

    def on_validation_epoch_end(self, trainer: Trainer, module):
        if not self._feats:
            return
        z = jnp.asarray(np.concatenate(self._feats, axis=0))
        self.value, entropy, top_sv = _rankme_components(z)
        trainer.log(self.name, self.value)  # bare key, matches torch
        if self.verbose:
            trainer.log(f"{self.name}/entropy", entropy)
            trainer.log(f"{self.name}/top_singular_value", top_sv)
