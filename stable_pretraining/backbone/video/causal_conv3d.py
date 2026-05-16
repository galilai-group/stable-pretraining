"""Causal 3D convolution primitive used by causal video encoders.

A standard ``nn.Conv3d`` lets every output frame see ``kernel_t // 2`` future
frames on each side of the temporal kernel. ``CausalConv3d`` removes that
future visibility by left-padding the time axis with ``kernel_t - 1`` zeros
and using zero temporal padding inside the underlying conv. The spatial axes
use ordinary symmetric padding.

This module is implemented as an ``nn.Conv3d`` plus an explicit ``F.pad`` so
that it composes cleanly with ``torch.compile`` (no Python branches inside
``forward``, no custom autograd).
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _triple(x: Union[int, tuple]) -> tuple:
    if isinstance(x, int):
        return (x, x, x)
    if len(x) != 3:
        raise ValueError(f"expected length-3 tuple, got {x!r}")
    return tuple(x)


class CausalConv3d(nn.Module):
    """``nn.Conv3d`` whose temporal receptive field is strictly causal.

    The temporal kernel of size ``kt`` looks at frames ``[t - kt + 1, ..., t]``
    only. Spatial axes use standard symmetric padding so the spatial output
    size matches a same-padded conv.

    Output time length follows the usual Conv3d formula with temporal padding
    treated as ``kt - 1``::

        T_out = floor((T_in + (kt - 1) - dilation_t * (kt - 1) - 1) / stride_t) + 1

    For ``stride_t = 1, dilation_t = 1`` (the common case) this is ``T_in``.

    :param in_channels: Input channel count.
    :param out_channels: Output channel count.
    :param kernel_size: Kernel as ``int`` (cubic) or ``(kt, kh, kw)`` tuple.
    :param stride: Stride as ``int`` or ``(st, sh, sw)``.
    :param dilation: Dilation as ``int`` or ``(dt, dh, dw)``.
    :param groups: Conv groups (for depthwise pass ``in_channels``).
    :param bias: Whether the underlying conv has a bias term.

    Example::

        conv = CausalConv3d(64, 128, kernel_size=(3, 3, 3))
        x = torch.randn(2, 64, 16, 32, 32)
        y = conv(x)  # (2, 128, 16, 32, 32)
        # Perturbing x[:, :, k+1:] leaves y[:, :, :k+1] unchanged.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple] = 3,
        stride: Union[int, tuple] = 1,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        kt, kh, kw = _triple(kernel_size)
        st, sh, sw = _triple(stride)
        dt, dh, dw = _triple(dilation)

        self.kernel_size = (kt, kh, kw)
        self.stride = (st, sh, sw)
        self.dilation = (dt, dh, dw)

        # Causal: full kernel_t - 1 (× dilation) padding on the *left* of time;
        # spatial: same-style symmetric pad applied via the conv's own padding.
        self._time_pad_left = (kt - 1) * dt
        pad_h = ((kh - 1) * dh) // 2
        pad_w = ((kw - 1) * dw) // 2

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(kt, kh, kw),
            stride=(st, sh, sw),
            padding=(0, pad_h, pad_w),
            dilation=(dt, dh, dw),
            groups=groups,
            bias=bias,
        )

    @property
    def weight(self) -> nn.Parameter:
        return self.conv.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.conv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal 3D convolution.

        :param x: ``(B, C_in, T, H, W)``
        :return: ``(B, C_out, T', H', W')`` per Conv3d output-size formula.
        """
        if self._time_pad_left > 0:
            # F.pad order for 5D is (W_left, W_right, H_left, H_right, T_left, T_right).
            x = F.pad(x, (0, 0, 0, 0, self._time_pad_left, 0))
        return self.conv(x)
