"""Shared math helpers for JAX-backend losses (ports of :mod:`stable_pretraining.losses.utils`)."""

import jax.numpy as jnp


def off_diagonal(x: jnp.ndarray) -> jnp.ndarray:
    """Return a flattened view of the off-diagonal elements of a square matrix.

    Mirrors the torch implementation exactly (flatten, drop last, reshape,
    slice) so the Barlow Twins / VICReg covariance terms match bit-for-bit.

    Args:
        x: A square matrix ``[n, n]``.

    Returns:
        jnp.ndarray: The ``n * (n - 1)`` off-diagonal elements, flattened.
    """
    n, m = x.shape
    if n != m:
        raise ValueError("Input matrix must be square.")
    return x.flatten()[:-1].reshape(n - 1, n + 1)[:, 1:].flatten()


def vcreg_loss(
    z: jnp.ndarray,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    epsilon: float = 1e-4,
) -> jnp.ndarray:
    """Variance-Covariance regularization term used by VICReg.

    JAX port of :class:`stable_pretraining.losses.utils.VCRegLoss`. Note the
    variance uses Bessel's correction (``ddof=1``) to match torch's default
    ``Tensor.var`` semantics.

    Args:
        z: Embeddings ``[N, D]``.
        std_coeff: Weight of the variance (std) term.
        cov_coeff: Weight of the covariance term.
        epsilon: Stability epsilon inside the std sqrt.

    Returns:
        jnp.ndarray: Scalar regularization loss.
    """
    z = z - jnp.mean(z, axis=0)
    std = jnp.sqrt(jnp.var(z, axis=0, ddof=1) + epsilon)
    std_loss = jnp.mean(jax_relu(1.0 - std)) / 2.0
    cov = (z.T @ z) / (z.shape[0] - 1)
    cov_loss = jnp.sum(off_diagonal(cov) ** 2) / z.shape[1]
    return std_coeff * std_loss + cov_coeff * cov_loss


def jax_relu(x: jnp.ndarray) -> jnp.ndarray:
    """ReLU without importing jax.nn at call sites that only need this."""
    return jnp.maximum(x, 0.0)


__all__ = ["off_diagonal", "vcreg_loss"]
