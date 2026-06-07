"""JAX-specific end-to-end test for the SimCLR method class."""

import numpy as np
import pytest
from flax import nnx

pytestmark = [pytest.mark.unit, pytest.mark.jax]

import stable_pretraining.jax as spj  # noqa: E402


def _two_view_data(n_batches=4, b=16, d=32, c=4, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_batches):
        labels = rng.randint(0, c, size=b)
        out.append(
            {
                "views": [
                    {"image": rng.randn(b, d).astype("float32"), "label": labels},
                    {"image": rng.randn(b, d).astype("float32"), "label": labels},
                ]
            }
        )
    return out


def _val_data(n_batches=2, b=16, d=32, c=4, seed=1):
    rng = np.random.RandomState(seed)
    return [
        {"image": rng.randn(b, d).astype("float32"), "label": rng.randint(0, c, size=b)}
        for _ in range(n_batches)
    ]


def _build(d=32, c=4, seed=0):
    rngs = nnx.Rngs(seed)
    backbone = spj.backbone.MLP(d, [64, d], rngs=rngs)
    model = spj.SimCLR(
        backbone=backbone,
        embed_dim=d,
        rngs=rngs,
        projector_dims=(64, 32),
        temperature=0.5,
        optim="adamw",
    )
    return model, rngs


def test_simclr_trains_end_to_end():
    model, rngs = _build()
    probe = spj.OnlineProbe("probe", probe=nnx.Linear(32, 4, rngs=rngs))
    rankme = spj.RankMe()
    trainer = spj.Trainer(max_epochs=3, callbacks=[probe, rankme])
    trainer.fit(model, _two_view_data(), _val_data())

    assert trainer.global_step == 12  # 3 epochs * 4 batches
    assert np.isfinite(trainer.callback_metrics["fit/loss"])
    assert "eval/probe_acc" in trainer.callback_metrics  # torch-matching key
    assert "rankme" in trainer.callback_metrics  # bare key, matches torch
    assert np.isfinite(rankme.value) and rankme.value > 1.0


def test_simclr_loss_decreases_on_fixed_data():
    model, _ = _build()
    data = _two_view_data()
    trainer = spj.Trainer(max_epochs=1)
    trainer.fit(model, data)
    first = trainer.callback_metrics["fit/loss"]
    trainer.max_epochs = 30
    trainer.fit(model, data)
    assert trainer.callback_metrics["fit/loss"] < first


def test_simclr_requires_two_views():
    model, _ = _build()
    rng = np.random.RandomState(0)
    bad = [{"views": [{"image": rng.randn(8, 32).astype("float32")}]}]  # 1 view
    with pytest.raises(ValueError):
        spj.Trainer(max_epochs=1).fit(model, bad)
