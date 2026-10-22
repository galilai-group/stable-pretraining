"""SSL-EY objective from Unconstrained Stochastic CCA (ICLR 2024)."""

from contextlib import nullcontext

import torch
from torch import nn

from stable_pretraining.utils import all_gather


class SSLEYLoss(nn.Module):
    """Centered two-view SSL-EY objective.

    For centered projections X and Y, let C = 2 X.T Y / (N - 1) and
    V = (X.T X + Y.T Y) / (N - 1). The loss is -2 tr(C) + tr(V V).
    This follows https://github.com/jameschapman19/SSL-EY and uses the actual
    global batch size. Negative losses are expected; no dimension averaging
    or feature normalization is applied.

    Args:
        force_float32: Compute covariance matrices outside autocast in float32
            for half inputs. Set False to reproduce the released author code's
            arithmetic and autocast behavior (used by the CIFAR recipe).

    Note:
        Distributed execution gathers projections with autograd support and
        requires equal local batch shapes on every rank. Use drop_last=True.
        By default covariance calculations use float32 for half/bfloat16
        inputs, even inside autocast; the reference recipe disables this. Float64 inputs retain their precision.
    """

    def __init__(self, force_float32: bool = True) -> None:
        super().__init__()
        self.force_float32 = force_float32

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the scalar loss for paired projections.

        Args:
            x: Floating-point tensor of shape (local batch size, features).
            y: Matching projections from the second augmentation.

        Returns:
            Differentiable scalar SSL-EY loss.

        Raises:
            ValueError: Shapes, dtypes, devices, or global sample counts differ
                from the required paired floating-point matrix contract.
        """
        if x.ndim != 2 or x.shape != y.shape or x.shape[1] == 0:
            raise ValueError(
                "SSL-EY requires matching nonempty [batch, features] tensors"
            )
        if not x.is_floating_point() or x.dtype != y.dtype or x.device != y.device:
            raise ValueError(
                "SSL-EY requires matching floating-point dtypes and devices"
            )
        context = (
            torch.autocast(device_type=x.device.type, enabled=False)
            if self.force_float32
            else nullcontext()
        )
        with context:
            if self.force_float32 and x.dtype in (torch.float16, torch.bfloat16):
                x, y = x.float(), y.float()
            x = torch.cat(all_gather(x), dim=0)
            y = torch.cat(all_gather(y), dim=0)
            n = x.shape[0]
            if n < 2:
                raise ValueError("SSL-EY requires at least two global samples")
            x, y = x - x.mean(dim=0), y - y.mean(dim=0)
            cross_covariance = 2 * (x.T @ y) / (n - 1)
            covariance = (x.T @ x) / (n - 1) + (y.T @ y) / (n - 1)
            return -2 * torch.trace(cross_covariance) + torch.trace(
                covariance @ covariance
            )
