"""Edge-case and error-path coverage for the JAX backend."""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

pytestmark = [pytest.mark.unit, pytest.mark.jax]

import stable_pretraining.jax as spj  # noqa: E402
from stable_pretraining.jax.forward import simclr  # noqa: E402


def test_module_requires_callable_forward():
    with pytest.raises(ValueError):
        spj.Module(forward=123)


def test_module_forward_must_return_dict():
    def bad_forward(self, batch, stage):
        return batch["image"]  # not a dict

    rngs = nnx.Rngs(0)
    module = spj.Module(forward=bad_forward, backbone=nnx.Linear(4, 4, rngs=rngs))
    with pytest.raises(ValueError):
        module.compute({"image": jnp.ones((2, 4))}, "fit")


@pytest.mark.parametrize("n_views", [1, 3])
def test_simclr_forward_rejects_wrong_view_count(n_views):
    rngs = nnx.Rngs(0)
    model = spj.SimCLR(
        backbone=spj.backbone.MLP(8, [8], rngs=rngs),
        embed_dim=8,
        rngs=rngs,
        projector_dims=(8, 8),
    )
    rng = np.random.RandomState(0)
    bad = [{"views": [{"image": rng.randn(4, 8).astype("float32")}] * n_views}]
    with pytest.raises(ValueError):
        spj.Trainer(max_epochs=1).fit(model, bad)


@pytest.mark.parametrize(
    "kwargs",
    [{"k": 0}, {"temperature": 0.0}, {"k": -1}, {"temperature": -1.0}],
)
def test_online_knn_rejects_bad_args(kwargs):
    with pytest.raises(ValueError):
        spj.OnlineKNN(num_classes=5, **kwargs)


def test_mlp_requires_nonempty_hidden():
    with pytest.raises(ValueError):
        spj.backbone.MLP(4, [], rngs=nnx.Rngs(0))


def test_vit_requires_divisible_patch():
    with pytest.raises(ValueError):
        spj.backbone.vit_tiny(rngs=nnx.Rngs(0), img_size=30, patch_size=8)


def test_early_stopping_rejects_bad_mode():
    with pytest.raises(ValueError):
        spj.EarlyStopping(monitor="m", mode="sideways")


def test_online_queue_rejects_bad_length():
    with pytest.raises(ValueError):
        spj.OnlineQueue(queue_length=0)


def test_eval_only_module_runs_without_optimizer():
    rngs = nnx.Rngs(0)
    module = spj.Module(
        forward=simclr,
        optim=None,
        backbone=spj.backbone.MLP(8, [8], rngs=rngs),
        projector=spj.backbone.MLP(8, [8], rngs=rngs),
        simclr_loss=spj.losses.NTXEntLoss(0.5),
    )
    rng = np.random.RandomState(0)
    data = [{"image": rng.randn(4, 8).astype("float32"), "label": rng.randint(0, 3, 4)}]
    trainer = spj.Trainer(max_epochs=2)
    trainer.fit(module, data)
    assert trainer.optimizer is None and trainer.global_step == 2


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_knn_handles_bank_smaller_than_k(metric):
    """K larger than the bank must clamp, not crash."""
    from stable_pretraining.jax.callbacks import knn_predict

    rng = np.random.RandomState(0)
    bank = jnp.asarray(rng.randn(3, 8).astype("float32"))
    labels = jnp.asarray(np.array([0, 1, 2]))
    q = jnp.asarray(rng.randn(2, 8).astype("float32"))
    scores = knn_predict(
        q, bank, labels, num_classes=3, k=20, temperature=0.07, metric=metric
    )
    assert scores.shape == (2, 3) and np.isfinite(np.asarray(scores)).all()
