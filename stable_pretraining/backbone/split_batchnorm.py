"""Split Batch Normalization for self-supervised learning.

This module provides batch normalization that computes statistics
independently across equally sized splits of the input batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SplitBatchNorm(nn.BatchNorm2d):
    """Batch normalization with independent statistics for batch splits.

    During training, divides the input batch into equally sized splits
    and normalizes each split using its own batch statistics. Running
    statistics are averaged across splits for use during evaluation.

    During evaluation, applies standard BatchNorm2d using the
    accumulated running statistics.

    Parameters
    ----------
    num_features : int
        Number of input channels.
    num_splits : int, default=8
        Number of equally sized batch splits.
    **kwargs
        Additional arguments passed to nn.BatchNorm2d.

    Raises:
    ------
    ValueError
        If num_splits is not a positive integer, the training batch size
        is not divisible by num_splits, or unsupported BatchNorm2d
        options are requested.

    Notes:
    -----
    Input and output tensors have shape (N, C, H, W).
    This implementation requires affine=True and track_running_stats=True.
    """

    def __init__(self, num_features: int, num_splits: int = 8, **kwargs):
        if (
            not isinstance(num_splits, int)
            or isinstance(num_splits, bool)
            or num_splits <= 0
        ):
            raise ValueError("num_splits must be a positive integer")
        super().__init__(num_features, **kwargs)
        self.num_splits = num_splits

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Normalize each batch split independently during training."""
        if not self.training:
            return super().forward(input)

        n, c, h, w = input.shape
        if n % self.num_splits != 0:
            raise ValueError(
                f"Batch size {n} must be divisible by num_splits={self.num_splits}"
            )

        running_mean_split = self.running_mean.repeat(self.num_splits)
        running_var_split = self.running_var.repeat(self.num_splits)

        output = F.batch_norm(
            input.reshape(-1, c * self.num_splits, h, w),
            running_mean_split,
            running_var_split,
            self.weight.repeat(self.num_splits),
            self.bias.repeat(self.num_splits),
            True,
            self.momentum,
            self.eps,
        ).reshape(n, c, h, w)

        with torch.no_grad():
            self.running_mean.copy_(
                running_mean_split.reshape(self.num_splits, c).mean(dim=0)
            )
            self.running_var.copy_(
                running_var_split.reshape(self.num_splits, c).mean(dim=0)
            )

        return output
