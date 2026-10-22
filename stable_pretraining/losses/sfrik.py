"""SFRIK's two-view objective from Zheng et al. (ICLR 2023)."""

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..utils import all_gather


class SFRIKLoss(nn.Module):
    """Match views and regularize their distribution on the unit sphere.

    Implements the degree-three rotation-invariant kernel of
    https://arxiv.org/abs/2208.00789. The objective is ``sim_coeff * MSE``
    plus ``mmd_coeff`` times the average of the two views' kernel means.
    Each kernel mean includes diagonal entries (the biased MMD estimator),
    matching the authors' two-view implementation. Inputs are normalized
    internally; the MSE averages over both samples and feature dimensions.

    Args:
        sim_coeff: Weight of view agreement. The default is the authors'
            8192-dimensional small-dataset setting, not a dimension-independent
            recommendation. Retune when changing the projection width.
        mmd_coeff: Weight of the uniformity penalty.
        kernel_weights: Nonnegative weights for degrees one, two, and three.
        gather_distributed: Gather both views with autograd before computing
            the loss. Distributed workers must have equal, nonempty batch
            sizes and matching feature dimensions (use ``drop_last=True``).
        eps: Positive denominator floor for L2 normalization.

    Note:
        Uses O(B²) memory for global batch size B. Float16/bfloat16 inputs
        are promoted to float32 and loss computation disables autocast.
        Float64 is preserved for numerical checks. With distributed gathering,
        every worker evaluates the same global loss; the gather backward and
        DDP's gradient averaging together reproduce the single-process gradient.
        This implements two views, not the paper's multi-crop variants.
    """

    def __init__(
        self,
        sim_coeff: float = 4000.0,
        mmd_coeff: float = 1.0,
        kernel_weights: Sequence[float] = (1.0, 40.0, 40.0),
        gather_distributed: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if any(not math.isfinite(c) or c < 0 for c in (sim_coeff, mmd_coeff)):
            raise ValueError("sim_coeff and mmd_coeff must be finite and nonnegative")
        if sim_coeff + mmd_coeff == 0:
            raise ValueError("At least one loss coefficient must be positive")
        if len(kernel_weights) != 3 or any(
            not math.isfinite(w) or w < 0 for w in kernel_weights
        ):
            raise ValueError(
                "kernel_weights must contain three finite nonnegative weights"
            )
        if sum(kernel_weights) == 0:
            raise ValueError("At least one kernel weight must be positive")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.sim_coeff = float(sim_coeff)
        self.mmd_coeff = float(mmd_coeff)
        self.kernel_weights = tuple(float(w) for w in kernel_weights)
        self.gather_distributed = gather_distributed
        self.eps = float(eps)

    def _uniformity(self, z: torch.Tensor) -> torch.Tensor:
        d = z.shape[1]
        cosine = z @ z.T
        w1, w2, w3 = self.kernel_weights
        kernel = (
            w1 * cosine
            + w2 * (d * cosine.square() - 1) / (d - 1)
            + w3 * ((d + 2) * cosine.pow(3) - 3 * cosine) / (d - 1)
        )
        return kernel.mean()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute the scalar loss for corresponding projected views.

        Args:
            z1: Floating-point projections of shape ``(batch, dimension)``.
            z2: Corresponding projections with the same shape, dtype and device.

        Returns:
            Scalar SFRIK objective, differentiable with respect to both inputs.

        Raises:
            ValueError: Shapes, devices or dtypes differ, the batch is empty,
                or the feature dimension is less than two.
            TypeError: Inputs are not floating-point tensors.
        """
        if z1.ndim != 2 or z1.shape != z2.shape:
            raise ValueError("SFRIK expects two matching (batch, dimension) tensors")
        if z1.shape[0] == 0 or z1.shape[1] < 2:
            raise ValueError("SFRIK requires a nonempty batch and dimension >= 2")
        if z1.device != z2.device or z1.dtype != z2.dtype:
            raise ValueError("SFRIK inputs must have the same device and dtype")
        if not z1.is_floating_point():
            raise TypeError("SFRIK inputs must be floating-point tensors")

        with torch.autocast(device_type=z1.device.type, enabled=False):
            if z1.dtype in (torch.float16, torch.bfloat16):
                z1, z2 = z1.float(), z2.float()
            z1 = F.normalize(z1, dim=1, eps=self.eps)
            z2 = F.normalize(z2, dim=1, eps=self.eps)
            if self.gather_distributed:
                z1 = torch.cat(all_gather(z1), dim=0)
                z2 = torch.cat(all_gather(z2), dim=0)
            agreement = F.mse_loss(z1, z2)
            uniformity = (self._uniformity(z1) + self._uniformity(z2)) / 2
            return self.sim_coeff * agreement + self.mmd_coeff * uniformity
