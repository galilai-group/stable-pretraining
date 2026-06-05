"""Tests for the additional SSL callbacks (OnlineQueue, OnlineWriter, LiDAR, EarlyStopping)."""

import types

import numpy as np
import pytest
from flax import nnx

pytestmark = [pytest.mark.unit, pytest.mark.jax]

import stable_pretraining.jax as spj  # noqa: E402
from stable_pretraining.jax.callbacks.lidar import lidar  # noqa: E402


def _two_view(n=4, b=16, d=16, c=5, seed=0):
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


def _simclr(d=16, seed=0):
    rngs = nnx.Rngs(seed)
    return spj.SimCLR(
        backbone=spj.backbone.MLP(d, [32, d], rngs=rngs),
        embed_dim=d,
        rngs=rngs,
        projector_dims=(32, 16),
    )


# --------------------------- OnlineQueue --------------------------- #
def test_online_queue_fifo_trims_and_exposes(seed=0):
    q = spj.OnlineQueue(key="embedding", queue_length=40, label_key="label")
    spj.Trainer(max_epochs=3, callbacks=[q]).fit(_simclr(), _two_view())
    assert len(q) <= 40
    feats = q.features
    assert feats is not None and feats.shape[1] == 16
    assert q.labels is not None and q.labels.shape[0] == feats.shape[0]


def test_online_queue_rejects_bad_length():
    with pytest.raises(ValueError):
        spj.OnlineQueue(queue_length=0)


# --------------------------- OnlineWriter --------------------------- #
def test_online_writer_saves_npz(tmp_path):
    writer = spj.OnlineWriter(names="embedding", path=str(tmp_path), during=("val",))
    rng = np.random.RandomState(0)
    val = [
        {"image": rng.randn(16, 16).astype("float32"), "label": rng.randint(0, 5, 16)}
    ]
    spj.Trainer(max_epochs=1, callbacks=[writer]).fit(_simclr(), _two_view(), val)
    files = list(tmp_path.glob("val_epoch*.npz"))
    assert len(files) == 1
    data = np.load(files[0])
    assert "embedding" in data and data["embedding"].shape[1] == 16


# --------------------------- LiDAR --------------------------- #
def test_lidar_range_and_collapse():
    """Separable multi-class uses multiple discriminant dims; rank-1 collapses to ~1."""
    rng = np.random.RandomState(0)
    d, c = 16, 5
    centers = rng.randn(c, d).astype("float32") * 10.0
    y = rng.randint(0, c, 200)
    separable = centers[y] + 0.1 * rng.randn(200, d).astype("float32")
    val = lidar(separable, y)
    assert 1.0 < val <= (c - 1) + 0.5  # bounded by min(d, n_classes-1)

    # Rank-1 features (one direction) -> a single discriminant dim -> LiDAR ~ 1.
    direction = rng.randn(1, d).astype("float32")
    rank1 = (rng.randn(200, 1).astype("float32")) * direction
    assert lidar(rank1, y) < 1.5


def test_lidar_single_class_is_nan():
    rng = np.random.RandomState(0)
    x = rng.randn(50, 8).astype("float32")
    assert np.isnan(lidar(x, np.zeros(50, dtype=int)))


def test_lidar_callback_end_to_end():
    lidar_cb = spj.LiDAR()
    rng = np.random.RandomState(0)
    val = [
        {"image": rng.randn(32, 16).astype("float32"), "label": rng.randint(0, 5, 32)}
    ]
    trainer = spj.Trainer(max_epochs=1, callbacks=[lidar_cb])
    trainer.fit(_simclr(), _two_view(), val)
    assert np.isfinite(lidar_cb.value)


# --------------------------- EarlyStopping --------------------------- #
def test_early_stopping_logic_min_mode():
    es = spj.EarlyStopping(monitor="m", mode="min", patience=2)
    fake = types.SimpleNamespace(callback_metrics={}, should_stop=False)
    for v in [1.0, 0.5]:  # improving
        fake.callback_metrics["m"] = v
        es.on_validation_epoch_end(fake, None)
        assert not fake.should_stop
    for v in [0.6, 0.7]:  # not improving for `patience` epochs
        fake.callback_metrics["m"] = v
        es.on_validation_epoch_end(fake, None)
    assert fake.should_stop


def test_early_stopping_halts_trainer():
    """A constant (never-improving) metric must stop the Trainer before max_epochs."""

    class _ConstMetric(spj.Callback):
        def on_validation_epoch_end(self, trainer, module):
            trainer.callback_metrics["plateau"] = 1.0

    es = spj.EarlyStopping(monitor="plateau", mode="min", patience=2)
    rng = np.random.RandomState(0)
    val = [
        {"image": rng.randn(16, 16).astype("float32"), "label": rng.randint(0, 5, 16)}
    ]
    trainer = spj.Trainer(max_epochs=20, callbacks=[_ConstMetric(), es])
    trainer.fit(_simclr(), _two_view(), val)
    assert trainer.should_stop and trainer.current_epoch < 19
