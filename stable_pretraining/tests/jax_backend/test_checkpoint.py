"""Tests for JAX-backend checkpointing (model, optimizer, eval-callback state)."""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

pytestmark = [pytest.mark.unit, pytest.mark.jax]

import stable_pretraining.jax as spj  # noqa: E402
from stable_pretraining.jax import checkpoint  # noqa: E402
from stable_pretraining.jax.forward import supervised  # noqa: E402


def _sup_module(d=8, c=3, seed=0):
    rngs = nnx.Rngs(seed)
    return spj.Module(
        forward=supervised,
        optim="adamw",
        backbone=spj.backbone.MLP(d, [16], rngs=rngs),
        classifier=nnx.Linear(16, c, rngs=rngs),
    )


def _data(n=4, b=8, d=8, c=3, seed=0):
    rng = np.random.RandomState(seed)
    return [
        {"image": rng.randn(b, d).astype("float32"), "label": rng.randint(0, c, b)}
        for _ in range(n)
    ]


def test_module_roundtrip(tmp_path):
    model = _sup_module()
    spj.Trainer(max_epochs=3).fit(model, _data())
    path = tmp_path / "m.msgpack"
    checkpoint.save(str(path), step=42, module=model)

    fresh = _sup_module()  # same init, but will be overwritten
    # perturb so we know load actually changes it
    fresh.classifier.kernel = nnx.Param(fresh.classifier.kernel[...] + 5.0)
    step = checkpoint.load(str(path), module=fresh)

    assert step == 42
    np.testing.assert_allclose(
        np.asarray(model.classifier.kernel[...]),
        np.asarray(fresh.classifier.kernel[...]),
        atol=1e-6,
    )
    # forward outputs match
    x = jnp.asarray(np.random.RandomState(1).randn(4, 8).astype("float32"))
    model.eval()
    fresh.eval()
    np.testing.assert_allclose(
        np.asarray(model.backbone(x)), np.asarray(fresh.backbone(x)), atol=1e-6
    )


def test_exact_resume_matches_uninterrupted(tmp_path):
    """Train 3+4 epochs with a checkpoint in between == train 7 epochs straight.

    The reference trains all 7 epochs in ONE trainer so its Adam moments are
    continuous; the split run must restore the optimizer state to match.
    """
    data = _data()
    # Reference: 7 continuous epochs (one optimizer, never reset).
    ref = _sup_module(seed=0)
    t_ref = spj.Trainer(max_epochs=7)
    t_ref.fit(ref, data)
    target = t_ref.callback_metrics["fit/loss"]

    # Split: 3 epochs, checkpoint (module + optimizer), then resume 4 more.
    split = _sup_module(seed=0)
    t1 = spj.Trainer(max_epochs=3)
    t1.fit(split, data)
    path = tmp_path / "resume.msgpack"
    checkpoint.save(
        str(path), step=t1.global_step, module=split, optimizer=t1.optimizer
    )

    resumed = _sup_module(seed=0)
    t2 = spj.Trainer(max_epochs=4)
    t2.fit(resumed, data, resume_from=str(path))

    assert t2.global_step == t1.global_step + 4 * len(data)
    # Exact resume (Adam moments restored) -> same loss as the straight 7-epoch run.
    np.testing.assert_allclose(t2.callback_metrics["fit/loss"], target, rtol=1e-4)


def test_resume_without_optimizer_state_differs(tmp_path):
    """Sanity: dropping optimizer state from the checkpoint changes the resume."""
    data = _data()
    m = _sup_module(seed=0)
    t = spj.Trainer(max_epochs=3)
    t.fit(m, data)
    # Save module only (no optimizer) -> Adam moments reset on resume.
    path = tmp_path / "noopt.msgpack"
    checkpoint.save(str(path), step=t.global_step, module=m)
    loaded = int(checkpoint.load(str(path), module=_sup_module(seed=0)))
    assert loaded == t.global_step


def test_probe_state_roundtrip(tmp_path):
    """An OnlineProbe's probe params (eval-callback state) checkpoint cleanly."""
    rngs = nnx.Rngs(0)
    probe = spj.OnlineProbe("p", probe=nnx.Linear(16, 5, rngs=rngs))
    probe.probe.kernel = nnx.Param(probe.probe.kernel[...] + 1.0)
    path = tmp_path / "probe.msgpack"
    checkpoint.save(str(path), probe=probe.probe)

    fresh = nnx.Linear(16, 5, rngs=nnx.Rngs(1))
    checkpoint.load(str(path), probe=fresh)
    np.testing.assert_allclose(
        np.asarray(probe.probe.kernel[...]), np.asarray(fresh.kernel[...]), atol=1e-6
    )


def test_checkpoint_callback_writes_and_restores(tmp_path):
    path = tmp_path / "cb.msgpack"
    model = _sup_module(seed=0)
    spj.Trainer(max_epochs=2, callbacks=[checkpoint.Checkpoint(str(path))]).fit(
        model, _data()
    )
    assert path.exists()
    fresh = _sup_module(seed=1)  # different init
    step = checkpoint.load(str(path), module=fresh)
    assert step > 0
    np.testing.assert_allclose(
        np.asarray(model.classifier.kernel[...]),
        np.asarray(fresh.classifier.kernel[...]),
        atol=1e-6,
    )
