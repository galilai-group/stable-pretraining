"""Tests for JAX-backend evaluation callbacks (OnlineKNN, OnlineProbe, RankMe)."""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

pytestmark = [pytest.mark.unit, pytest.mark.jax]

import stable_pretraining.jax as spj  # noqa: E402
from stable_pretraining.jax.callbacks import knn_predict  # noqa: E402
from stable_pretraining.jax.callbacks.rankme import rankme  # noqa: E402


def test_rankme_discriminates_rank():
    """Full-rank embeddings score high; rank-1 embeddings score ~1."""
    rng = np.random.RandomState(0)
    full = jnp.asarray(rng.randn(64, 16).astype("float32"))
    direction = rng.randn(1, 16).astype("float32")
    rank1 = jnp.asarray((rng.randn(64, 1).astype("float32")) * direction)
    assert rankme(full) > 5.0  # many effective dimensions
    assert rankme(rank1) < 1.5  # collapsed to one direction


def test_online_probe_learns_on_separable_data():
    """The probe (trained via the callback) must classify separable embeddings."""

    # Backbone is the identity-ish: embeddings ARE the inputs, separable by label.
    class _Identity(nnx.Module):
        embed_dim = 8

        def __call__(self, x):
            return x

    rngs = nnx.Rngs(0)
    centers = np.eye(3, 8, dtype="float32") * 8.0  # 3 well-separated class centers

    def batch(seed):
        rng = np.random.RandomState(seed)
        y = rng.randint(0, 3, 32)
        img = centers[y] + 0.1 * rng.randn(32, 8).astype("float32")
        return {
            "views": [{"image": img, "label": y}, {"image": img, "label": y}],
        }

    def val_batch(seed):
        rng = np.random.RandomState(seed)
        y = rng.randint(0, 3, 32)
        return {
            "image": centers[y] + 0.1 * rng.randn(32, 8).astype("float32"),
            "label": y,
        }

    model = spj.SimCLR(
        backbone=_Identity(), embed_dim=8, rngs=rngs, projector_dims=(8, 8)
    )
    probe = spj.OnlineProbe(
        "probe",
        probe=nnx.Linear(8, 3, rngs=rngs),
        optim={"type": "adamw", "learning_rate": 0.05},
    )
    train = [batch(s) for s in range(8)]
    val = [val_batch(100 + s) for s in range(2)]
    spj.Trainer(max_epochs=15, callbacks=[probe]).fit(model, train, val)
    assert probe.accuracy > 0.9  # learned the separable mapping


def test_online_probe_resets_counters_each_epoch():
    rngs = nnx.Rngs(0)
    model = spj.SimCLR(
        backbone=spj.backbone.MLP(8, [8], rngs=rngs),
        embed_dim=8,
        rngs=rngs,
        projector_dims=(8, 8),
    )
    probe = spj.OnlineProbe("probe", probe=nnx.Linear(8, 3, rngs=rngs))
    rng = np.random.RandomState(0)
    train = [
        {
            "views": [
                {
                    "image": rng.randn(16, 8).astype("float32"),
                    "label": rng.randint(0, 3, 16),
                }
            ]
            * 2
        }
        for _ in range(2)
    ]
    val = [
        {"image": rng.randn(16, 8).astype("float32"), "label": rng.randint(0, 3, 16)}
    ]
    trainer = spj.Trainer(max_epochs=3, callbacks=[probe])
    trainer.fit(model, train, val)
    # _total reflects exactly one val epoch (counters reset on epoch start).
    assert probe._total == 16


def test_knn_predict_separable_is_correct():
    """On linearly separable clusters, weighted k-NN must classify perfectly."""
    rng = np.random.RandomState(0)
    centers = np.array([[5.0, 5.0], [-5.0, -5.0], [5.0, -5.0]], dtype="float32")
    bank = np.repeat(centers, 20, axis=0) + 0.1 * rng.randn(60, 2).astype("float32")
    bank_labels = np.repeat(np.arange(3), 20)
    queries = centers + 0.1 * rng.randn(3, 2).astype("float32")
    scores = knn_predict(
        jnp.asarray(queries),
        jnp.asarray(bank),
        jnp.asarray(bank_labels),
        num_classes=3,
        k=5,
        temperature=0.07,
        metric="euclidean",
    )
    preds = np.asarray(jnp.argmax(scores, axis=-1))
    np.testing.assert_array_equal(preds, np.array([0, 1, 2]))


def test_knn_predict_cosine_metric_runs():
    rng = np.random.RandomState(1)
    bank = rng.randn(30, 8).astype("float32")
    labels = rng.randint(0, 4, size=30)
    q = rng.randn(5, 8).astype("float32")
    scores = knn_predict(
        jnp.asarray(q), jnp.asarray(bank), jnp.asarray(labels), 4, 5, 0.07, "cosine"
    )
    assert scores.shape == (5, 4)


def test_online_knn_end_to_end():
    """OnlineKNN must populate a bank during training and report a val accuracy."""
    d = 16
    rng = np.random.RandomState(0)
    rngs = nnx.Rngs(0)
    model = spj.SimCLR(
        backbone=spj.backbone.MLP(d, [32, d], rngs=rngs),
        embed_dim=d,
        rngs=rngs,
        projector_dims=(32, 16),
    )
    train = [
        {
            "views": [
                {
                    "image": rng.randn(16, d).astype("float32"),
                    "label": rng.randint(0, 5, 16),
                },
                {
                    "image": rng.randn(16, d).astype("float32"),
                    "label": rng.randint(0, 5, 16),
                },
            ]
        }
        for _ in range(4)
    ]
    val = [
        {"image": rng.randn(16, d).astype("float32"), "label": rng.randint(0, 5, 16)}
        for _ in range(2)
    ]
    knn = spj.OnlineKNN(num_classes=5, k=5)
    spj.Trainer(max_epochs=2, callbacks=[knn]).fit(model, train, val)
    assert 0.0 <= knn.accuracy <= 1.0


def test_online_knn_rejects_bad_args():
    with pytest.raises(ValueError):
        spj.OnlineKNN(num_classes=5, k=0)
    with pytest.raises(ValueError):
        spj.OnlineKNN(num_classes=5, temperature=0.0)
