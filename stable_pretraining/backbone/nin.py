"""Network-in-Network encoder used by RotNet on CIFAR-10.

Adapted from gidariss/FeatureLearningRotNet at edafaba6561003292017d0a44272414356c959ff
(``architectures/NetworkInNetwork.py``), the official RotNet code, distributed
under the MIT license in ``licenses/ROTNET_LICENSE``. See
``THIRD_PARTY_NOTICES.md``.
"""

import torch
from torch import nn

NIN_CHANNELS = 192


def _conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def nin_block(
    in_channels: int,
    kernel_size: int,
    mid_channels: int = NIN_CHANNELS,
    out_channels: int = NIN_CHANNELS,
    pool: str = None,
) -> nn.Sequential:
    """One NIN block: a ``k x k`` conv followed by two ``1 x 1`` convs.

    Args:
        in_channels: Input channels.
        kernel_size: Kernel size of the first convolution.
        mid_channels: Output channels of the first ``1 x 1`` convolution.
        out_channels: Output channels of the block.
        pool: ``"max"``, ``"avg"`` or ``None`` for a stride-2 ``3 x 3`` pooling
            appended to the block.

    Returns:
        The block as an ``nn.Sequential``.
    """
    layers = [
        _conv_bn_relu(in_channels, NIN_CHANNELS, kernel_size),
        _conv_bn_relu(NIN_CHANNELS, mid_channels, 1),
        _conv_bn_relu(mid_channels, out_channels, 1),
    ]
    if pool == "max":
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    elif pool == "avg":
        layers.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
    elif pool is not None:
        raise ValueError(f"pool must be 'max', 'avg' or None, got {pool!r}")
    return nn.Sequential(*layers)


class NetworkInNetwork(nn.Module):
    """Network-in-Network (Lin et al., 2014) as configured for RotNet.

    Block 1 maps 3 to 96 channels and max-pools, block 2 average-pools, and
    every later block keeps 192 channels at the same resolution. On 32x32
    inputs the block outputs are 96x16x16, 192x8x8, then 192x8x8. This is the
    official CIFAR-10 configuration (no pooling after block 3).

    Args:
        num_blocks: Number of NIN blocks (at least 3). RotNet's CIFAR-10
            results use 4.
        in_channels: Input image channels.
        pool: If ``True``, ``forward`` returns globally average-pooled
            features ``[B, 192]``; otherwise the last feature map.

    Note:
        The official code defines a custom ``weight_initialization`` but never
        calls it, so the published results come from PyTorch 0.2's defaults.
        Conv layers match today's defaults (uniform in +-1/sqrt(fan_in)), but
        batch-norm scales were drawn from U[0, 1) rather than set to 1, so
        they are re-initialised that way here.
    """

    def __init__(
        self,
        num_blocks: int = 4,
        in_channels: int = 3,
        pool: bool = True,
    ):
        super().__init__()
        if num_blocks < 3:
            raise ValueError(f"num_blocks must be at least 3, got {num_blocks}")
        blocks = [
            nin_block(in_channels, 5, mid_channels=160, out_channels=96, pool="max"),
            nin_block(96, 5, pool="avg"),
            nin_block(NIN_CHANNELS, 3),
        ]
        blocks += [nin_block(NIN_CHANNELS, 3) for _ in range(3, num_blocks)]
        self.blocks = nn.ModuleList(blocks)
        self.pool = pool
        self.embed_dim = NIN_CHANNELS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x.mean(dim=(2, 3)) if self.pool else x
