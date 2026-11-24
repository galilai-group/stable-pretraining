"""FroSSL multiview joint-embedding objective."""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


class FroSSLLoss(torch.nn.Module):
    """Frobenius norm minimization with multiview invariance.

    Args:
        invariance_weight: Weight of the invariance term. The default of 1.0
            follows the authors' multiview implementation; paper experiments
            tune this weight for each number of views.

    Note:
        Following the authors' multiview code, each ``[batch, features]`` view
        is L2-normalized along the batch axis without mean-centering. The paper
        describes a centered covariance; this implementation uses the
        trace-normalized Gram objective from the released code.
        The per-view regularizer is the log of the squared Frobenius norm of
        its trace-normalized Gram matrix. The invariance term compares every
        view with the mean view, following Equation 3 of the FroSSL paper.
        Statistics are local to each process under distributed training.
        See Skean et al., ECCV 2024, https://arxiv.org/abs/2310.02903.
    """

    def __init__(self, invariance_weight: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(invariance_weight) or invariance_weight < 0:
            raise ValueError("invariance_weight must be finite and non-negative")
        self.invariance_weight = invariance_weight

    def forward(self, projections: Sequence[torch.Tensor]) -> torch.Tensor:
        """Compute the FroSSL loss for aligned views.

        Args:
            projections: Two or more aligned tensors of shape
                ``[batch, features]``. Row ``i`` must identify the same sample
                in every view.

        Returns:
            Scalar FroSSL loss. Half-precision inputs are evaluated in float32.

        Raises:
            ValueError: The views have invalid shapes, nonfinite values, or
                zero total projection energy.
        """
        if isinstance(projections, torch.Tensor) or len(projections) < 2:
            raise ValueError("FroSSL requires a sequence of at least two views")

        shape = projections[0].shape
        if len(shape) != 2 or shape[0] < 2 or shape[1] < 1:
            raise ValueError("each FroSSL view must have shape [batch>=2, features>=1]")

        device = projections[0].device
        dtype = projections[0].dtype
        normalized = []
        eps = 1e-12
        for view in projections:
            if (
                view.shape != shape
                or view.device != device
                or view.dtype != dtype
                or not view.is_floating_point()
            ):
                raise ValueError(
                    "FroSSL views must share shape, device, and floating dtype"
                )
            if not torch.isfinite(view).all():
                raise ValueError("FroSSL views must contain only finite values")
            z = view if view.dtype == torch.float64 else view.float()
            # Rescaling bounds the norm calculation while preserving the original
            # epsilon floor and its gradient in projection units.
            scale = z.detach().abs().amax(dim=0, keepdim=True).clamp_min(eps)
            z = z / scale
            norm = torch.linalg.vector_norm(z, dim=0, keepdim=True)
            normalized.append(z / norm.clamp_min(eps / scale))

        mean_view = torch.stack(normalized).mean(dim=0)
        num_views = len(normalized)
        feature_dim = shape[1]
        loss = mean_view.new_zeros(())
        for z in normalized:
            # Trace normalization cancels a common scale. Rescale before the
            # product so tiny views retain nonzero Gram entries.
            scale = z.detach().abs().amax()
            if scale == 0:
                raise ValueError("FroSSL views must have nonzero projection energy")
            scaled = z / scale
            # The two Gram matrices have the same nonzero spectrum; use the
            # smaller one to avoid quadratic work in the larger dimension.
            with torch.autocast(device_type=device.type, enabled=False):
                gram = (
                    scaled.T @ scaled if feature_dim <= shape[0] else scaled @ scaled.T
                )
            regularizer = (gram / gram.diagonal().sum()).square().sum().log()
            invariance = num_views * feature_dim * F.mse_loss(z, mean_view)
            loss = loss + regularizer + self.invariance_weight * invariance
        return loss
