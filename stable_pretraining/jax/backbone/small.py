"""Compact native Flax-NNX backbones: ResNet-9 and ConvMixer (channels-last).

ResNet-9 is the DAWNBench-style fast CIFAR network; ConvMixer
:cite:`trockman2022patches` is the all-convolutional patch mixer. Both return
pooled features ``[B, embed_dim]`` and use ``train()``/``eval()`` for BatchNorm.
"""

import jax
import jax.numpy as jnp
from flax import nnx


def _conv_bn(din, dout, rngs, k=3, stride=1):
    return nnx.Sequential(
        nnx.Conv(
            din,
            dout,
            kernel_size=(k, k),
            strides=(stride, stride),
            padding=[(k // 2, k // 2)] * 2,
            use_bias=False,
            rngs=rngs,
        ),
        nnx.BatchNorm(dout, momentum=0.9, rngs=rngs),
    )


class _Residual(nnx.Module):
    """x + conv_bn_relu(conv_bn_relu(x)) — the ResNet-9 residual unit."""

    def __init__(self, dim, rngs):
        self.b1 = _conv_bn(dim, dim, rngs)
        self.b2 = _conv_bn(dim, dim, rngs)

    def __call__(self, x):
        y = jax.nn.relu(self.b1(x))
        y = jax.nn.relu(self.b2(y))
        return x + y


class ResNet9(nnx.Module):
    """ResNet-9 returning pooled features ``[B, 512]``.

    Args:
        rngs: NNX RNG collection.
        in_channels: Input channels. Default ``3``.

    Attributes:
        embed_dim: ``512``.
    """

    def __init__(self, *, rngs: nnx.Rngs, in_channels: int = 3):
        self.stem = _conv_bn(in_channels, 64, rngs)
        self.conv2 = _conv_bn(64, 128, rngs)
        self.res1 = _Residual(128, rngs)
        self.conv3 = _conv_bn(128, 256, rngs)
        self.conv4 = _conv_bn(256, 512, rngs)
        self.res2 = _Residual(512, rngs)
        self.embed_dim = 512

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run ResNet-9 on NHWC images, returning global-avg-pooled features."""
        x = jax.nn.relu(self.stem(x))
        x = nnx.max_pool(jax.nn.relu(self.conv2(x)), (2, 2), strides=(2, 2))
        x = self.res1(x)
        x = nnx.max_pool(jax.nn.relu(self.conv3(x)), (2, 2), strides=(2, 2))
        x = nnx.max_pool(jax.nn.relu(self.conv4(x)), (2, 2), strides=(2, 2))
        x = self.res2(x)
        return jnp.mean(x, axis=(1, 2))


class _ConvMixerBlock(nnx.Module):
    """Depthwise (residual) + pointwise mixing block."""

    def __init__(self, dim, kernel_size, rngs):
        self.depthwise = nnx.Conv(
            dim,
            dim,
            kernel_size=(kernel_size, kernel_size),
            padding="SAME",
            feature_group_count=dim,  # depthwise
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(dim, momentum=0.9, rngs=rngs)
        self.pointwise = nnx.Conv(dim, dim, kernel_size=(1, 1), rngs=rngs)
        self.bn2 = nnx.BatchNorm(dim, momentum=0.9, rngs=rngs)

    def __call__(self, x):
        x = x + self.bn1(jax.nn.gelu(self.depthwise(x)))  # residual depthwise mix
        x = self.bn2(jax.nn.gelu(self.pointwise(x)))  # channel mix
        return x


class ConvMixer(nnx.Module):
    """ConvMixer returning pooled features ``[B, dim]`` :cite:`trockman2022patches`.

    Args:
        rngs: NNX RNG collection.
        dim: Hidden dimension (also the output ``embed_dim``). Default ``256``.
        depth: Number of mixing blocks. Default ``8``.
        kernel_size: Depthwise kernel size. Default ``9``.
        patch_size: Patch-embedding stride. Default ``7``.
        in_channels: Input channels. Default ``3``.

    Attributes:
        embed_dim: Equal to ``dim``.
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        dim: int = 256,
        depth: int = 8,
        kernel_size: int = 9,
        patch_size: int = 7,
        in_channels: int = 3,
    ):
        self.patch = nnx.Conv(
            in_channels,
            dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            rngs=rngs,
        )
        self.patch_bn = nnx.BatchNorm(dim, momentum=0.9, rngs=rngs)
        self.blocks = nnx.List(
            [_ConvMixerBlock(dim, kernel_size, rngs) for _ in range(depth)]
        )
        self.embed_dim = dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run ConvMixer on NHWC images, returning global-avg-pooled features."""
        x = self.patch_bn(jax.nn.gelu(self.patch(x)))
        for block in self.blocks:
            x = block(x)
        return jnp.mean(x, axis=(1, 2))


__all__ = ["ResNet9", "ConvMixer"]
