"""SwAV loss and Sinkhorn-Knopp normalization for the JAX backend.

JAX ports of the SwAV pieces in :mod:`stable_pretraining.losses.joint_embedding`
:cite:`caron2020unsupervised`. The Sinkhorn-Knopp iteration is numerically
faithful to the torch version and is regression-tested for parity; ``SwAVLoss``
operates on prototype *scores* (the method class owns the prototype layer and
its normalization), keeping the loss a pure function.
"""

import jax
import jax.numpy as jnp


def sinkhorn(scores: jnp.ndarray, epsilon: float = 0.05, n_iterations: int = 3):
    """Sinkhorn-Knopp optimal-transport normalization (matches ``SwAVLoss.sinkhorn``).

    Args:
        scores: Prototype scores ``[B, K]`` (B samples, K prototypes).
        epsilon: Entropic regularization temperature.
        n_iterations: Number of Sinkhorn iterations.

    Returns:
        jnp.ndarray: Soft assignment ``[B, K]`` (rows sum to 1).
    """
    q = jnp.exp(scores / epsilon).T  # [K, B]
    q = q / jnp.sum(q)
    k, b = q.shape
    r = jnp.ones(k) / k
    c = jnp.ones(b) / b
    for _ in range(n_iterations):
        u = jnp.sum(q, axis=1)
        q = q * (r / u)[:, None]
        q = q * (c / jnp.sum(q, axis=0))[None, :]
    return (q / jnp.sum(q, axis=0, keepdims=True)).T


class SwAVLoss:
    """SwAV swapped-prediction loss over prototype scores :cite:`caron2020unsupervised`.

    Args:
        temperature: Softmax temperature for the prediction task. Default ``0.1``.
        sinkhorn_iterations: Sinkhorn iterations for target assignment. Default ``3``.
        epsilon: Sinkhorn entropic regularization. Default ``0.05``.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        epsilon: float = 0.05,
    ):
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon

    def _swapped_prediction(self, scores, q):
        return -jnp.mean(
            jnp.sum(q * jax.nn.log_softmax(scores / self.temperature, axis=1), axis=1)
        )

    def __call__(self, scores1: jnp.ndarray, scores2: jnp.ndarray) -> jnp.ndarray:
        """Compute the swapped-prediction loss between two views' prototype scores.

        Args:
            scores1: Prototype scores of view 1 ``[B, K]``.
            scores2: Prototype scores of view 2 ``[B, K]``.

        Returns:
            jnp.ndarray: Scalar SwAV loss.
        """
        q1 = jax.lax.stop_gradient(
            sinkhorn(scores1, self.epsilon, self.sinkhorn_iterations)
        )
        q2 = jax.lax.stop_gradient(
            sinkhorn(scores2, self.epsilon, self.sinkhorn_iterations)
        )
        return 0.5 * (
            self._swapped_prediction(scores1, q2)
            + self._swapped_prediction(scores2, q1)
        )


__all__ = ["sinkhorn", "SwAVLoss"]
