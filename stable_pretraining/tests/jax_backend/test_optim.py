"""Optimizer-factory matrix for the JAX backend."""

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

pytestmark = [pytest.mark.unit, pytest.mark.jax]

from stable_pretraining.jax.optim import create_optimizer, warmup_cosine_schedule  # noqa: E402


@pytest.mark.parametrize("name", ["adamw", "adam", "sgd", "lars", "lamb"])
def test_create_optimizer_by_name_steps(name):
    tx = create_optimizer({"type": name, "learning_rate": 0.1})
    assert isinstance(tx, optax.GradientTransformation)
    lin = nnx.Linear(4, 3, rngs=nnx.Rngs(0))
    opt = nnx.Optimizer(lin, tx, wrt=nnx.Param)
    x = jnp.ones((2, 4))
    before = float((lin(x) ** 2).mean())

    @nnx.jit
    def step(m, o):
        loss, g = nnx.value_and_grad(lambda mm: (mm(x) ** 2).mean())(m)
        o.update(m, g)
        return loss

    for _ in range(10):
        after = float(step(lin, opt))
    assert np.isfinite(after) and after < before


def test_create_optimizer_accepts_transformation_and_string():
    tx = optax.sgd(0.1)
    assert create_optimizer(tx) is tx  # passthrough
    assert isinstance(create_optimizer("sgd"), optax.GradientTransformation)


def test_create_optimizer_rejects_unknown():
    with pytest.raises(ValueError):
        create_optimizer({"type": "definitely_not_an_optimizer"})


@pytest.mark.parametrize("warmup", [0, 5])
def test_warmup_cosine_schedule_shape(warmup):
    sched = warmup_cosine_schedule(base_lr=1.0, total_steps=100, warmup_steps=warmup)
    lrs = [float(sched(s)) for s in (0, warmup, 50, 100)]
    assert all(np.isfinite(lr) for lr in lrs)
    assert max(lrs) <= 1.0 + 1e-6
    if warmup:
        assert lrs[0] < lrs[1]  # warms up
