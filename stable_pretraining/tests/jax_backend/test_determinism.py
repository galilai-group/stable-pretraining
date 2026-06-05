"""Determinism + a small end-to-end integration run for the JAX backend."""

import numpy as np
import pytest
from flax import nnx

pytestmark = [pytest.mark.unit, pytest.mark.jax]

import stable_pretraining.jax as spj  # noqa: E402


def _two_view(n=4, b=16, d=32, c=4, seed=0):
    rng = np.random.RandomState(seed)
    return [
        {
            "views": [
                {
                    "image": rng.randn(b, d).astype("float32"),
                    "label": rng.randint(0, c, b),
                },
                {
                    "image": rng.randn(b, d).astype("float32"),
                    "label": rng.randint(0, c, b),
                },
            ]
        }
        for _ in range(n)
    ]


def _simclr(d=32, seed=0):
    rngs = nnx.Rngs(seed)
    return spj.SimCLR(
        backbone=spj.backbone.MLP(d, [64, d], rngs=rngs),
        embed_dim=d,
        rngs=rngs,
        projector_dims=(64, 32),
    )


def test_same_seed_gives_identical_loss():
    """Two runs with the same seed + data must produce the bit-identical loss curve."""
    data = _two_view()
    losses = []
    for _ in range(2):
        model = _simclr(seed=0)
        t = spj.Trainer(max_epochs=5)
        t.fit(model, data)
        losses.append(t.callback_metrics["fit/loss"])
    assert losses[0] == losses[1]


def test_different_seed_gives_different_init():
    a, b = _simclr(seed=0), _simclr(seed=1)
    ka = np.asarray(a.backbone.layers[0].kernel[...])
    kb = np.asarray(b.backbone.layers[0].kernel[...])
    assert not np.allclose(ka, kb)


@pytest.mark.integration
def test_simclr_resnet9_integration_probe_learns():
    """End-to-end: SimCLR(ResNet-9) on synthetic separable images; probe beats chance."""
    rng = np.random.RandomState(0)
    n_cls, b = 4, 24
    centers = rng.randn(n_cls, 16, 16, 3).astype("float32") * 3.0

    def make(seed, two_view):
        r = np.random.RandomState(seed)
        y = r.randint(0, n_cls, b)
        img = centers[y] + 0.3 * r.randn(b, 16, 16, 3).astype("float32")
        if two_view:
            img2 = centers[y] + 0.3 * r.randn(b, 16, 16, 3).astype("float32")
            return {"views": [{"image": img, "label": y}, {"image": img2, "label": y}]}
        return {"image": img, "label": y}

    rngs = nnx.Rngs(0)
    model = spj.SimCLR(
        backbone=spj.backbone.ResNet9(rngs=rngs),
        embed_dim=512,
        rngs=rngs,
        projector_dims=(256, 128),
        temperature=0.5,
    )
    probe = spj.OnlineProbe(
        "probe",
        probe=nnx.Linear(512, n_cls, rngs=rngs),
        optim={"type": "adamw", "learning_rate": 0.01},
    )
    train = [make(s, True) for s in range(6)]
    val = [make(100 + s, False) for s in range(2)]
    trainer = spj.Trainer(max_epochs=8, callbacks=[probe, spj.RankMe()])
    trainer.fit(model, train, val)
    assert probe.accuracy > 1.0 / n_cls  # beats chance on separable data
