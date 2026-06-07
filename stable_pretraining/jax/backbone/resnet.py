"""Native Flax-NNX ResNet backbones (channels-last / NHWC).

Ports of the torchvision ResNet family used across the torch backend. They are
*native* Flax modules (no torch bridge): ``nnx.Conv`` + ``nnx.BatchNorm`` with
``train()``/``eval()`` driving the BatchNorm running-average flag, exactly like
``self.training`` on the torch side.

The backbone returns pooled features ``[B, C]`` (``num_classes=0`` semantics) so
it slots straight into SimCLR/VICReg/etc. as ``self.backbone``.

Note:
    BatchNorm ``momentum`` here is the flax *decay* (``0.9``), which corresponds
    to torch's ``momentum=0.1``. For multi-device runs, ``bn_axis_name`` can name
    a mapped axis for cross-replica stat sync (used under ``shard_map``); pure
    SPMD ``jit`` runs keep per-replica stats, which is fine for the large
    per-device batches typical of SSL.
"""

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx


def _conv(din, d_out, k, stride, rngs, padding, dtype=None):
    # ``dtype`` is the *computation* dtype (e.g. bfloat16 for mixed precision);
    # ``param_dtype`` stays float32 so the weights are kept in full precision.
    return nnx.Conv(
        din,
        d_out,
        kernel_size=(k, k),
        strides=(stride, stride),
        padding=padding,
        use_bias=False,
        dtype=dtype,
        rngs=rngs,
    )


def _bn(features, rngs, axis_name):
    # BatchNorm is kept in float32 (its compute dtype is unset) — mirroring
    # torch AMP, which excludes normalization layers from autocast.
    return nnx.BatchNorm(
        features, momentum=0.9, epsilon=1e-5, axis_name=axis_name, rngs=rngs
    )


def _maybe_downsample(din, d_out, stride, rngs, axis_name, dtype=None):
    """1x1-conv + BN projection shortcut, or ``None`` when shapes already match."""
    if stride != 1 or din != d_out:
        return nnx.Sequential(
            _conv(din, d_out, 1, stride, rngs, "VALID", dtype),
            _bn(d_out, rngs, axis_name),
        )
    return None


class BasicBlock(nnx.Module):
    """ResNet basic block (two 3x3 convs); used by ResNet-18/34."""

    expansion = 1

    def __init__(self, din, d_out, stride, rngs, axis_name=None, dtype=None):
        self.conv1 = _conv(din, d_out, 3, stride, rngs, [(1, 1), (1, 1)], dtype)
        self.bn1 = _bn(d_out, rngs, axis_name)
        self.conv2 = _conv(d_out, d_out, 3, 1, rngs, [(1, 1), (1, 1)], dtype)
        self.bn2 = _bn(d_out, rngs, axis_name)
        # Assign exactly once: mixing a static ``None`` then a module would
        # flip the attribute's pytree status and NNX rejects that.
        self.downsample = _maybe_downsample(
            din, d_out * self.expansion, stride, rngs, axis_name, dtype
        )

    def __call__(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = jax.nn.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return jax.nn.relu(x + identity)


class Bottleneck(nnx.Module):
    """ResNet bottleneck block (1x1, 3x3, 1x1); used by ResNet-50/101/152."""

    expansion = 4

    def __init__(self, din, d_out, stride, rngs, axis_name=None, dtype=None):
        self.conv1 = _conv(din, d_out, 1, 1, rngs, "VALID", dtype)
        self.bn1 = _bn(d_out, rngs, axis_name)
        self.conv2 = _conv(d_out, d_out, 3, stride, rngs, [(1, 1), (1, 1)], dtype)
        self.bn2 = _bn(d_out, rngs, axis_name)
        self.conv3 = _conv(d_out, d_out * self.expansion, 1, 1, rngs, "VALID", dtype)
        self.bn3 = _bn(d_out * self.expansion, rngs, axis_name)
        self.downsample = _maybe_downsample(
            din, d_out * self.expansion, stride, rngs, axis_name, dtype
        )

    def __call__(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = jax.nn.relu(self.bn1(self.conv1(x)))
        x = jax.nn.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return jax.nn.relu(x + identity)


class ResNet(nnx.Module):
    """Native Flax-NNX ResNet returning pooled features ``[B, embed_dim]``.

    Args:
        block: ``BasicBlock`` or ``Bottleneck``.
        layers: Number of blocks per stage, e.g. ``[2, 2, 2, 2]`` for ResNet-18.
        rngs: NNX RNG collection.
        in_channels: Input image channels. Default ``3``.
        low_resolution: Use a 3x3 stride-1 stem and skip max-pool (CIFAR/small
            images). Default ``False`` (ImageNet-style 7x7 stem + maxpool).
        bn_axis_name: Optional mapped-axis name for cross-replica BatchNorm.

    Attributes:
        embed_dim: Output feature dimension (``512`` for ResNet-18/34,
            ``2048`` for ResNet-50+).
    """

    def __init__(
        self,
        block,
        layers: Sequence[int],
        *,
        rngs: nnx.Rngs,
        in_channels: int = 3,
        low_resolution: bool = False,
        bn_axis_name: Optional[str] = None,
        dtype=None,
    ):
        self.low_resolution = low_resolution
        width = 64
        if low_resolution:
            self.stem_conv = _conv(
                in_channels, width, 3, 1, rngs, [(1, 1), (1, 1)], dtype
            )
        else:
            self.stem_conv = _conv(
                in_channels, width, 7, 2, rngs, [(3, 3), (3, 3)], dtype
            )
        self.stem_bn = _bn(width, rngs, bn_axis_name)

        stages, din = [], width
        for i, (depth, out) in enumerate(zip(layers, [64, 128, 256, 512])):
            stride = 1 if i == 0 else 2
            blocks = []
            for b in range(depth):
                blocks.append(
                    block(din, out, stride if b == 0 else 1, rngs, bn_axis_name, dtype)
                )
                din = out * block.expansion
            stages.append(nnx.Sequential(*blocks))
        self.stages = nnx.List(stages)
        self.embed_dim = din

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run the ResNet.

        Args:
            x: Input images in NHWC layout, ``[B, H, W, C]``.

        Returns:
            jnp.ndarray: Global-average-pooled features ``[B, embed_dim]``.
        """
        x = jax.nn.relu(self.stem_bn(self.stem_conv(x)))
        if not self.low_resolution:
            # 3x3 stride-2 max pool with SAME padding (torch maxpool stem).
            x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for stage in self.stages:
            x = stage(x)
        return jnp.mean(x, axis=(1, 2))  # global average pool over H, W


def resnet18(*, rngs: nnx.Rngs, **kwargs) -> ResNet:
    """ResNet-18 (BasicBlock, [2, 2, 2, 2]); ``embed_dim=512``."""
    return ResNet(BasicBlock, [2, 2, 2, 2], rngs=rngs, **kwargs)


def resnet34(*, rngs: nnx.Rngs, **kwargs) -> ResNet:
    """ResNet-34 (BasicBlock, [3, 4, 6, 3]); ``embed_dim=512``."""
    return ResNet(BasicBlock, [3, 4, 6, 3], rngs=rngs, **kwargs)


def resnet50(*, rngs: nnx.Rngs, **kwargs) -> ResNet:
    """ResNet-50 (Bottleneck, [3, 4, 6, 3]); ``embed_dim=2048``."""
    return ResNet(Bottleneck, [3, 4, 6, 3], rngs=rngs, **kwargs)


def resnet101(*, rngs: nnx.Rngs, **kwargs) -> ResNet:
    """ResNet-101 (Bottleneck, [3, 4, 23, 3]); ``embed_dim=2048``."""
    return ResNet(Bottleneck, [3, 4, 23, 3], rngs=rngs, **kwargs)


def resnet152(*, rngs: nnx.Rngs, **kwargs) -> ResNet:
    """ResNet-152 (Bottleneck, [3, 8, 36, 3]); ``embed_dim=2048``."""
    return ResNet(Bottleneck, [3, 8, 36, 3], rngs=rngs, **kwargs)


__all__ = [
    "ResNet",
    "BasicBlock",
    "Bottleneck",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]
