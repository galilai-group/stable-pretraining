"""JAX-specific unit tests for backbones."""

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

pytestmark = [pytest.mark.unit, pytest.mark.jax]

from stable_pretraining.jax.backbone import (  # noqa: E402
    MLP,
    ConvMixer,
    ResNet9,
    resnet18,
    resnet50,
    vit_tiny,
)


def test_mlp_output_shape():
    mlp = MLP(16, [32, 8], rngs=nnx.Rngs(0))
    x = jnp.ones((4, 16))
    assert mlp(x).shape == (4, 8)


def test_mlp_is_deterministic():
    mlp = MLP(16, [32, 8], rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(3, 16).astype("float32"))
    np.testing.assert_array_equal(np.asarray(mlp(x)), np.asarray(mlp(x)))


def test_mlp_single_layer_is_linear():
    """A one-entry hidden_channels list is a bare linear map (no activation)."""
    mlp = MLP(4, [2], rngs=nnx.Rngs(0))
    assert len(mlp.layers) == 1
    x = jnp.asarray([[1.0, -2.0, 3.0, -4.0]])
    # Linear output can be negative — proves no ReLU was applied on the last layer.
    out = np.asarray(mlp(x))
    assert out.shape == (1, 2)


def test_mlp_requires_nonempty_hidden():
    with pytest.raises(ValueError):
        MLP(4, [], rngs=nnx.Rngs(0))


def test_resnet18_output_shape_lowres():
    net = resnet18(rngs=nnx.Rngs(0), low_resolution=True)
    x = jnp.asarray(np.random.RandomState(0).randn(2, 32, 32, 3).astype("float32"))
    net.eval()
    assert net(x).shape == (2, 512)
    assert net.embed_dim == 512


def test_resnet50_output_shape():
    net = resnet50(rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.RandomState(0).randn(2, 64, 64, 3).astype("float32"))
    net.eval()
    assert net(x).shape == (2, 2048)
    assert net.embed_dim == 2048


def test_resnet_batchnorm_train_eval_differ():
    """BatchNorm running-average path must make train() and eval() outputs differ."""
    net = resnet18(rngs=nnx.Rngs(0), low_resolution=True)
    x = jnp.asarray(np.random.RandomState(1).randn(4, 32, 32, 3).astype("float32"))
    net.train()
    y_train = np.asarray(net(x))
    net.eval()
    y_eval = np.asarray(net(x))
    assert not np.allclose(y_train, y_eval)


def test_resnet_trains_under_jit():
    """A jit-compiled SGD step on the ResNet must reduce a trivial loss."""
    net = resnet18(rngs=nnx.Rngs(0), low_resolution=True)
    opt = nnx.Optimizer(net, optax.sgd(0.1), wrt=nnx.Param)
    x = jnp.asarray(np.random.RandomState(2).randn(4, 32, 32, 3).astype("float32"))

    @nnx.jit
    def step(m, o):
        loss, grads = nnx.value_and_grad(lambda mm: (mm(x) ** 2).mean())(m)
        o.update(m, grads)
        return loss

    first = float(step(net, opt))
    for _ in range(3):
        last = float(step(net, opt))
    assert last < first


def test_vit_tiny_output_shape():
    vit = vit_tiny(rngs=nnx.Rngs(0), img_size=32, patch_size=8)
    x = jnp.asarray(np.random.RandomState(0).randn(2, 32, 32, 3).astype("float32"))
    assert vit(x).shape == (2, 192)
    assert vit.embed_dim == 192


def test_vit_requires_divisible_patch():
    with pytest.raises(ValueError):
        vit_tiny(rngs=nnx.Rngs(0), img_size=30, patch_size=8)


def test_resnet_deep_presets():
    from stable_pretraining.jax.backbone import resnet101, resnet152

    x = jnp.asarray(np.random.RandomState(0).randn(1, 64, 64, 3).astype("float32"))
    for fn in (resnet101, resnet152):
        net = fn(rngs=nnx.Rngs(0))
        net.eval()
        assert net(x).shape == (1, 2048)
        assert net.embed_dim == 2048


def test_vit_base_preset():
    from stable_pretraining.jax.backbone import vit_base

    vit = vit_base(rngs=nnx.Rngs(0), img_size=32, patch_size=16)
    x = jnp.asarray(np.random.RandomState(0).randn(1, 32, 32, 3).astype("float32"))
    assert vit(x).shape == (1, 768)
    assert vit.embed_dim == 768


def test_resnet9_output_shape():
    net = ResNet9(rngs=nnx.Rngs(0))
    net.eval()
    x = jnp.asarray(np.random.RandomState(0).randn(2, 32, 32, 3).astype("float32"))
    assert net(x).shape == (2, 512)
    assert net.embed_dim == 512


def test_convmixer_output_shape():
    net = ConvMixer(rngs=nnx.Rngs(0), dim=64, depth=3, patch_size=4, kernel_size=5)
    net.eval()
    x = jnp.asarray(np.random.RandomState(0).randn(2, 32, 32, 3).astype("float32"))
    assert net(x).shape == (2, 64)
    assert net.embed_dim == 64


def test_convmixer_trains_under_jit():
    net = ConvMixer(rngs=nnx.Rngs(0), dim=32, depth=2, patch_size=4, kernel_size=5)
    opt = nnx.Optimizer(net, optax.sgd(0.1), wrt=nnx.Param)
    x = jnp.asarray(np.random.RandomState(0).randn(4, 32, 32, 3).astype("float32"))

    @nnx.jit
    def step(m, o):
        loss, grads = nnx.value_and_grad(lambda mm: (mm(x) ** 2).mean())(m)
        o.update(m, grads)
        return loss

    first = float(step(net, opt))
    for _ in range(3):
        last = float(step(net, opt))
    assert last < first
