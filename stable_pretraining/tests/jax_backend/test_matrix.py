"""Parametrized coverage matrices: methods × precision, backbones × precision.

Broad-but-bounded sweeps so each method/backbone is exercised under f32 and
bf16, not just the one or two hand-written cases.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

pytestmark = [pytest.mark.unit, pytest.mark.jax]

import stable_pretraining.jax as spj  # noqa: E402

METHODS = ["simclr", "vicreg", "barlow_twins", "simsiam", "byol"]
PRECISIONS = [None, jnp.bfloat16]


def _two_view(n=2, b=16, d=32, seed=0):
    rng = np.random.RandomState(seed)
    return [
        {
            "views": [
                {
                    "image": rng.randn(b, d).astype("float32"),
                    "label": rng.randint(0, 4, b),
                },
                {
                    "image": rng.randn(b, d).astype("float32"),
                    "label": rng.randint(0, 4, b),
                },
            ]
        }
        for _ in range(n)
    ]


def _build_method(name, backbone, d, rngs):
    """Return (model, extra_callbacks) for the named SSL method."""
    if name == "simclr":
        return spj.SimCLR(
            backbone=backbone, embed_dim=d, rngs=rngs, projector_dims=(64, 32)
        ), []
    if name == "vicreg":
        return spj.VICReg(
            backbone=backbone, embed_dim=d, rngs=rngs, projector_dims=(64, 64)
        ), []
    if name == "barlow_twins":
        return spj.BarlowTwins(
            backbone=backbone, embed_dim=d, rngs=rngs, projector_dims=(64, 64)
        ), []
    if name == "simsiam":
        return (
            spj.SimSiam(
                backbone=backbone,
                embed_dim=d,
                rngs=rngs,
                projector_dims=(64, 64),
                predictor_dim=32,
            ),
            [],
        )
    if name == "byol":
        return (
            spj.BYOL(
                backbone=backbone,
                embed_dim=d,
                rngs=rngs,
                projector_dims=(64, 32),
                predictor_hidden=64,
            ),
            [spj.TeacherStudentCallback()],
        )
    raise ValueError(name)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", PRECISIONS)
def test_method_trains_and_loss_improves(method, dtype):
    d = 32
    rngs = nnx.Rngs(0)
    backbone = spj.backbone.MLP(d, [64, d], rngs=rngs, dtype=dtype)
    model, extra = _build_method(method, backbone, d, rngs)
    data = _two_view(d=d)

    t1 = spj.Trainer(max_epochs=1, callbacks=list(extra))
    t1.fit(model, data)
    first = t1.callback_metrics["fit/loss"]
    t2 = spj.Trainer(max_epochs=40, callbacks=list(extra))
    t2.fit(model, data)
    final = t2.callback_metrics["fit/loss"]

    assert np.isfinite(final), f"{method}/{dtype}: non-finite loss"
    assert final < first, f"{method}/{dtype}: loss did not improve ({first} -> {final})"


# --------------------------- backbone matrix --------------------------- #
def _img(b=2, s=32):
    return jnp.asarray(np.random.RandomState(0).randn(b, s, s, 3).astype("float32"))


# (name, builder(rngs, dtype), embed_dim, supports_dtype)
BACKBONES = [
    ("resnet9", lambda r, dt: spj.backbone.ResNet9(rngs=r), 512, False),
    ("resnet18", lambda r, dt: spj.backbone.resnet18(rngs=r, dtype=dt), 512, True),
    ("resnet34", lambda r, dt: spj.backbone.resnet34(rngs=r, dtype=dt), 512, True),
    ("resnet50", lambda r, dt: spj.backbone.resnet50(rngs=r, dtype=dt), 2048, True),
    ("resnet101", lambda r, dt: spj.backbone.resnet101(rngs=r, dtype=dt), 2048, True),
    ("resnet152", lambda r, dt: spj.backbone.resnet152(rngs=r, dtype=dt), 2048, True),
    (
        "convmixer",
        lambda r, dt: spj.backbone.ConvMixer(
            rngs=r, dim=64, depth=2, patch_size=8, kernel_size=5
        ),
        64,
        False,
    ),
    (
        "vit_tiny",
        lambda r, dt: spj.backbone.vit_tiny(
            rngs=r, img_size=32, patch_size=16, dtype=dt
        ),
        192,
        True,
    ),
    (
        "vit_small",
        lambda r, dt: spj.backbone.vit_small(
            rngs=r, img_size=32, patch_size=16, dtype=dt
        ),
        384,
        True,
    ),
    (
        "vit_base",
        lambda r, dt: spj.backbone.vit_base(
            rngs=r, img_size=32, patch_size=16, dtype=dt
        ),
        768,
        True,
    ),
    (
        "vit_large",
        lambda r, dt: spj.backbone.vit_large(
            rngs=r, img_size=32, patch_size=16, dtype=dt
        ),
        1024,
        True,
    ),
]


@pytest.mark.parametrize(
    "name,builder,embed,supports_dtype", BACKBONES, ids=[b[0] for b in BACKBONES]
)
@pytest.mark.parametrize("dtype", PRECISIONS)
def test_backbone_forward(name, builder, embed, supports_dtype, dtype):
    if dtype is not None and not supports_dtype:
        pytest.skip(f"{name} has no dtype/mixed-precision path")
    net = builder(nnx.Rngs(0), dtype)
    net.eval()
    out = np.asarray(net(_img()))
    assert out.shape == (2, embed)
    assert np.isfinite(out).all()
    assert net.embed_dim == embed


# A jit'd train step must flow gradients end-to-end and update the backbone.
# (We assert trainability — non-zero grads + changed params — rather than a
# monotonic loss drop: an artificial ``(output**2).mean()`` objective through
# BatchNorm/residuals is not reliably reduced by plain SGD, which is a property
# of that toy objective, not of whether the backbone trains.)
TRAINABLE = ["resnet9", "resnet18", "convmixer", "vit_tiny"]


@pytest.mark.parametrize("name", TRAINABLE)
def test_backbone_train_step_updates_params(name):
    builder = dict((b[0], b[1]) for b in BACKBONES)[name]
    net = builder(nnx.Rngs(0), None)
    opt = nnx.Optimizer(net, optax.sgd(0.1), wrt=nnx.Param)
    x = _img()
    before = [np.array(p) for p in jax.tree_util.tree_leaves(nnx.state(net, nnx.Param))]

    @nnx.jit
    def step(m, o):
        loss, grads = nnx.value_and_grad(lambda mm: (mm(x) ** 2).mean())(m)
        o.update(m, grads)
        return loss, grads

    loss, grads = step(net, opt)
    assert np.isfinite(float(loss))
    # Gradients reach the parameters (non-zero somewhere in the backbone).
    grad_leaves = jax.tree_util.tree_leaves(grads)
    assert any(float(np.abs(np.asarray(g)).sum()) > 0 for g in grad_leaves)
    # And the optimizer actually moved the parameters.
    after = jax.tree_util.tree_leaves(nnx.state(net, nnx.Param))
    assert any(not np.allclose(b, np.asarray(a)) for b, a in zip(before, after))
