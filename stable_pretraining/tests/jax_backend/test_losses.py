"""JAX-specific unit tests for backend losses."""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.jax]

from stable_pretraining.jax.losses import (  # noqa: E402
    NTXEntLoss,
    SwAVLoss,
    l2_normalize,
    sinkhorn,
)


def test_l2_normalize_unit_norm():
    x = jnp.asarray(np.random.RandomState(0).randn(5, 7).astype("float32"))
    xn = l2_normalize(x, axis=-1)
    norms = np.asarray(jnp.linalg.norm(xn, axis=-1))
    np.testing.assert_allclose(norms, np.ones(5), atol=1e-5)


def test_l2_normalize_zero_vector_is_safe():
    x = jnp.zeros((3, 4))
    out = np.asarray(l2_normalize(x))
    assert np.all(np.isfinite(out))  # eps floor prevents div-by-zero


def test_ntxent_is_finite_and_nonnegative():
    rng = np.random.RandomState(1)
    z_i = jnp.asarray(rng.randn(8, 16).astype("float32"))
    z_j = jnp.asarray(rng.randn(8, 16).astype("float32"))
    loss = float(NTXEntLoss(temperature=0.5)(z_i, z_j))
    assert np.isfinite(loss)
    assert loss >= 0.0


def test_ntxent_aligned_views_beats_random():
    """Loss for matched (aligned) views should be lower than for random pairs."""
    rng = np.random.RandomState(2)
    z = jnp.asarray(rng.randn(16, 8).astype("float32"))
    loss_fn = NTXEntLoss(temperature=0.5)
    aligned = float(loss_fn(z, z))  # perfect positives
    random = float(loss_fn(z, jnp.asarray(rng.randn(16, 8).astype("float32"))))
    assert aligned < random


def test_sinkhorn_rows_sum_to_one():
    rng = np.random.RandomState(0)
    scores = jnp.asarray(rng.randn(16, 10).astype("float32"))
    q = sinkhorn(scores, epsilon=0.05, n_iterations=3)
    np.testing.assert_allclose(np.asarray(jnp.sum(q, axis=1)), np.ones(16), atol=1e-4)


def test_swav_loss_finite_and_nonnegative():
    rng = np.random.RandomState(1)
    s1 = jnp.asarray(rng.randn(16, 10).astype("float32"))
    s2 = jnp.asarray(rng.randn(16, 10).astype("float32"))
    loss = float(SwAVLoss()(s1, s2))
    assert np.isfinite(loss) and loss >= 0.0
