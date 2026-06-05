r"""Multi-device (SPMD data-parallel) tests for the JAX backend.

The real sharding path only exercises when more than one device is visible.
Run with two simulated CPU devices to cover it::

    XLA_FLAGS="--xla_force_host_platform_device_count=2" \
        python -m pytest stable_pretraining/tests/jax_backend/test_multidevice.py

On a single device the data-parallel switch must gracefully fall back to the
plain loop, which is what the default CI run checks.
"""

import jax
import numpy as np
import pytest
from flax import nnx

pytestmark = [pytest.mark.unit, pytest.mark.jax]

import stable_pretraining.jax as spj  # noqa: E402


def _two_view(n_batches=4, b=32, d=64, c=10, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_batches):
        y = rng.randint(0, c, size=b)
        out.append(
            {
                "views": [
                    {"image": rng.randn(b, d).astype("float32"), "label": y},
                    {"image": rng.randn(b, d).astype("float32"), "label": y},
                ]
            }
        )
    return out


def _model(d=64, seed=0):
    rngs = nnx.Rngs(seed)
    return spj.SimCLR(
        backbone=spj.backbone.MLP(d, [128, d], rngs=rngs),
        embed_dim=d,
        rngs=rngs,
        projector_dims=(64, 32),
    )


def test_data_parallel_falls_back_on_single_device():
    """data_parallel=True must run even with one device (no sharding applied)."""
    trainer = spj.Trainer(max_epochs=2, data_parallel=True)
    trainer.fit(_model(), _two_view())
    assert trainer.global_step == 8
    if jax.device_count() == 1:
        assert trainer._batch_sharding is None


@pytest.mark.skipif(
    jax.device_count() < 2, reason="needs >=2 devices (set XLA_FLAGS device count)"
)
def test_data_parallel_handles_ragged_batches():
    """A validation batch not divisible by device count must not crash (replicated)."""
    n = jax.device_count()
    rng = np.random.RandomState(0)
    train = _two_view(b=8 * n)  # divisible -> sharded
    # ragged: leading dim not divisible by device count
    val = [
        {
            "image": rng.randn(8 * n - 1, 64).astype("float32"),
            "label": rng.randint(0, 10, size=8 * n - 1),
        }
    ]
    trainer = spj.Trainer(max_epochs=1, data_parallel=True)
    trainer.fit(_model(), train, val)  # must complete without IndivisibleError
    assert trainer.global_step == len(train)


@pytest.mark.skipif(
    jax.device_count() < 2, reason="needs >=2 devices (set XLA_FLAGS device count)"
)
def test_data_parallel_matches_single_device_loss():
    """SPMD data-parallel and single-device must agree on the loss curve.

    Same init, same data, same global batch — sharding the batch across devices
    must not change the math (XLA all-reduces the gradients).
    """
    data = _two_view()
    single = spj.Trainer(max_epochs=3, data_parallel=False)
    single.fit(_model(seed=0), data)

    multi = spj.Trainer(max_epochs=3, data_parallel=True)
    multi.fit(_model(seed=0), data)

    assert multi._batch_sharding is not None  # sharding really engaged
    np.testing.assert_allclose(
        multi.callback_metrics["fit/loss"],
        single.callback_metrics["fit/loss"],
        rtol=1e-4,
        atol=1e-4,
    )
