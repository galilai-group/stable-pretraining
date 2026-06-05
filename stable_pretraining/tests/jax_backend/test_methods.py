"""End-to-end tests for the joint-embedding method classes (VICReg, Barlow Twins)."""

import numpy as np
import pytest
from flax import nnx

pytestmark = [pytest.mark.unit, pytest.mark.jax]

import stable_pretraining.jax as spj  # noqa: E402


def _two_view(n_batches=4, b=16, d=32, seed=0):
    rng = np.random.RandomState(seed)
    return [
        {
            "views": [
                {"image": rng.randn(b, d).astype("float32")},
                {"image": rng.randn(b, d).astype("float32")},
            ]
        }
        for _ in range(n_batches)
    ]


@pytest.mark.parametrize("method", ["vicreg", "barlow_twins"])
def test_joint_embedding_loss_decreases(method):
    d = 32
    rngs = nnx.Rngs(0)
    backbone = spj.backbone.MLP(d, [64, d], rngs=rngs)
    if method == "vicreg":
        model = spj.VICReg(
            backbone=backbone, embed_dim=d, rngs=rngs, projector_dims=(64, 64)
        )
    else:
        model = spj.BarlowTwins(
            backbone=backbone, embed_dim=d, rngs=rngs, projector_dims=(64, 64)
        )

    data = _two_view()
    trainer = spj.Trainer(max_epochs=1)
    trainer.fit(model, data)
    first = trainer.callback_metrics["fit/loss"]
    trainer.max_epochs = 30
    trainer.fit(model, data)
    assert trainer.callback_metrics["fit/loss"] < first


def test_simclr_gpu_augmentation_populates_views():
    """GPU augment transform turns a raw-image batch into views and trains."""
    d_img, size = 32, 24
    rngs = nnx.Rngs(0)
    backbone = spj.backbone.resnet18(rngs=rngs, low_resolution=True)
    model = spj.SimCLR(
        backbone=backbone,
        embed_dim=512,
        rngs=rngs,
        projector_dims=(128, 64),
        transform=spj.augment.two_view_transform(size=size),
        aug_seed=0,
    )
    rng = np.random.RandomState(0)
    # Raw images (single "image" key), NOT pre-split into views.
    train = [
        {
            "image": rng.rand(8, d_img, d_img, 3).astype("float32"),
            "label": rng.randint(0, 5, 8),
        }
        for _ in range(3)
    ]
    trainer = spj.Trainer(max_epochs=2)
    trainer.fit(model, train)
    assert trainer.global_step == 6
    assert np.isfinite(trainer.callback_metrics["fit/loss"])


def test_simsiam_loss_decreases():
    d = 32
    rngs = nnx.Rngs(0)
    model = spj.SimSiam(
        backbone=spj.backbone.MLP(d, [64, d], rngs=rngs),
        embed_dim=d,
        rngs=rngs,
        projector_dims=(64, 64),
        predictor_dim=32,
    )
    data = _two_view()
    trainer = spj.Trainer(max_epochs=1)
    trainer.fit(model, data)
    first = trainer.callback_metrics["fit/loss"]
    trainer.max_epochs = 30
    trainer.fit(model, data)
    # Negative cosine similarity: lower (more negative) is better.
    assert trainer.callback_metrics["fit/loss"] < first
