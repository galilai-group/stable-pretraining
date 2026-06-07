"""Logging parity: the JAX backend drives the same logger classes as torch.

Covers the duck-typed logger contract (log_hyperparams/log_metrics/save/finalize)
and the real RegistryLogger end-to-end — JAX runs must show up in ``spt registry``
with the same ``sidecar.json`` the torch path writes.
"""

import json

import numpy as np
import pytest
from flax import nnx

pytestmark = [pytest.mark.unit, pytest.mark.jax]

import stable_pretraining as spt  # noqa: E402
import stable_pretraining.jax as spj  # noqa: E402
from stable_pretraining.jax.forward import supervised  # noqa: E402


def _module(seed=0, hparams=None):
    rngs = nnx.Rngs(seed)
    return spj.Module(
        forward=supervised,
        optim="adamw",
        hparams=hparams or {},
        backbone=spj.backbone.MLP(8, [16], rngs=rngs),
        classifier=nnx.Linear(16, 3, rngs=rngs),
    )


def _data(n=3, b=8, seed=0):
    rng = np.random.RandomState(seed)
    return [
        {"image": rng.randn(b, 8).astype("float32"), "label": rng.randint(0, 3, b)}
        for _ in range(n)
    ]


class _FakeLogger:
    """Records the Lightning-Logger calls the Trainer makes."""

    def __init__(self):
        self.hparams = None
        self.metric_calls = []
        self.finalized = None
        self.saved = 0
        self.checkpoint = None

    def log_hyperparams(self, params, *a, **k):
        self.hparams = dict(params)

    def log_metrics(self, metrics, step=None):
        self.metric_calls.append((dict(metrics), step))

    def save(self):
        self.saved += 1

    def finalize(self, status):
        self.finalized = status

    def after_save_checkpoint(self, cb):
        self.checkpoint = cb.last_model_path


def test_trainer_drives_logger_api():
    lg = _FakeLogger()
    module = _module(hparams={"method": "supervised", "lr": 1e-3})
    trainer = spj.Trainer(max_epochs=3, logger=lg)
    trainer.fit(module, _data())

    assert lg.hparams == {"method": "supervised", "lr": 1e-3}
    assert len(lg.metric_calls) == 3  # one flush per epoch
    metrics0, step0 = lg.metric_calls[0]
    assert "fit/loss" in metrics0 and "epoch" in metrics0
    assert isinstance(step0, int)
    assert lg.finalized == "success" and lg.saved >= 1


def test_logger_finalize_failed_on_exception():
    lg = _FakeLogger()

    class _Boom(spj.Callback):
        def on_train_epoch_end(self, trainer, module):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        spj.Trainer(max_epochs=1, logger=lg, callbacks=[_Boom()]).fit(
            _module(), _data()
        )
    assert lg.finalized == "failed"


def test_epoch_mean_logged_for_forward_metrics():
    lg = _FakeLogger()
    trainer = spj.Trainer(max_epochs=1, logger=lg)
    trainer.fit(_module(), _data(n=4))
    # fit/loss in the flushed metrics is the epoch mean (finite, positive).
    m, _ = lg.metric_calls[0]
    assert np.isfinite(m["fit/loss"]) and m["fit/loss"] > 0


def test_metric_keys_match_torch_convention():
    """The logged metric keys must match the torch backend byte-for-byte.

    torch: forward ``fit/loss``; OnlineProbe ``train/<name>_loss`` +
    ``eval/<name>_acc``; OnlineKNN ``eval/<name>_acc``; RankMe/LiDAR bare ``<name>``.
    """
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
        for _ in range(3)
    ]
    val = [
        {"image": rng.randn(16, d).astype("float32"), "label": rng.randint(0, 5, 16)}
        for _ in range(2)
    ]
    cbs = [
        spj.OnlineProbe("probe", probe=nnx.Linear(d, 5, rngs=rngs)),
        spj.OnlineKNN(num_classes=5, name="knn", k=5),
        spj.RankMe(name="rankme"),
        spj.LiDAR(name="lidar"),
    ]
    trainer = spj.Trainer(max_epochs=2, callbacks=cbs)
    trainer.fit(model, train, val)

    keys = set(trainer.callback_metrics)
    for expected in (
        "fit/loss",
        "train/probe_loss",
        "eval/probe_acc",
        "eval/knn_acc",
        "rankme",
        "lidar",
    ):
        assert expected in keys, f"missing key {expected!r}; have {sorted(keys)}"


def test_registry_sidecar_written_and_queryable(tmp_path, monkeypatch):
    """Manager auto-injects RegistryLogger -> sidecar.json + spt registry sees the run."""
    for k in ("SLURM_JOB_ID", "SLURM_RESTART_COUNT", "SLURM_ARRAY_TASK_ID"):
        monkeypatch.delenv(k, raising=False)
    cfg = spt.get_config()
    old = cfg.cache_dir
    cfg.cache_dir = str(tmp_path)
    try:
        module = _module(hparams={"method": "supervised", "arch": "mlp"})
        mgr = spj.Manager(spj.Trainer(max_epochs=2), module, _data())
        mgr()

        sidecar = mgr.run_dir / "sidecar.json"
        assert sidecar.exists()
        rec = json.loads(sidecar.read_text())
        blob = json.dumps(rec)
        assert "supervised" in blob  # hyperparameters captured
        assert "fit/loss" in blob  # metrics captured
        assert rec.get("checkpoint_path")  # checkpoint path recorded

        # Queryable through the same registry API as torch runs.
        from stable_pretraining.registry import open_registry

        reg = open_registry(cache_dir=str(tmp_path))
        df = reg.to_dataframe()
        assert len(df) >= 1
    finally:
        cfg.cache_dir = old
