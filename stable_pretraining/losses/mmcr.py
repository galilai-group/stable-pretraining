"""Maximum Manifold Capacity Representations (Yerxa et al., NeurIPS 2023)."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.distributed import all_gather, get_world_size


class MMCRLoss(nn.Module):
    """Compress view manifolds while expanding their centroids' nuclear norm.

    For normalized projections ``Z[b, v, d]`` and centroids ``C = mean_v Z``,
    the objective is ``lmbda * mean_b ||Z[b]||_* - ||C||_*``. The global term
    is a sum of singular values, without division by batch size. Centroids
    are not normalized again. This follows the authors' CIFAR/STL objective:
    https://github.com/ThomasYerxa/mmcr/blob/master/mmcr/cifar_stl/loss_mmcr.py

    Args:
        lmbda: Nonnegative local compression weight. The default of zero
            matches the authors' CIFAR/STL training entry point.
        eps: Positive denominator floor for per-view L2 normalization.
        gather_distributed: Gather projections with autograd when a process
            group is initialized. Ranks must supply equal nonempty shapes.
            Set to False to explicitly optimize rank-local objectives.

    Note:
        Half and bfloat16 inputs are promoted to float32 before normalization
        and SVD; float64 is preserved for numerical checks. The nuclear norm
        is nonsmooth at rank deficiency; PyTorch selects a subgradient there.
        With distributed gathering every rank computes the same global loss;
        autograd gathering plus DDP gradient averaging gives the gradient of
        the global objective, without an extra world-size loss multiplier.
    """

    def __init__(
        self,
        lmbda: float = 0.0,
        eps: float = 1e-12,
        gather_distributed: bool = True,
    ) -> None:
        super().__init__()
        if not math.isfinite(lmbda) or lmbda < 0:
            raise ValueError("lmbda must be finite and nonnegative")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.lmbda = float(lmbda)
        self.eps = float(eps)
        self.gather_distributed = gather_distributed

    def forward(self, projections: torch.Tensor) -> torch.Tensor:
        """Compute MMCR on aligned views of a batch.

        Args:
            projections: Floating tensor of shape ``[B, V, D]``. Each row
                ``projections[b]`` contains at least two views of one sample.

        Returns:
            Scalar loss in float32 or float64 on the input device.

        Raises:
            ValueError: For invalid dimensions, nonfinite input, or unequal
                distributed shapes.
            TypeError: If the input is not a real floating tensor.
        """
        if (
            not isinstance(projections, torch.Tensor)
            or not projections.is_floating_point()
        ):
            raise TypeError("projections must be a real floating tensor")
        if projections.ndim != 3:
            raise ValueError("projections must have shape [B, V, D]")
        batch, views, dim = projections.shape
        if batch < 1 or views < 2 or dim < 1:
            raise ValueError("MMCR requires B >= 1, V >= 2, and D >= 1")
        if not torch.isfinite(projections).all():
            raise ValueError("projections must contain only finite values")

        # Autocast must not lower normalization/centroid arithmetic before SVD.
        with torch.autocast(device_type=projections.device.type, enabled=False):
            z = (
                projections
                if projections.dtype == torch.float64
                else projections.float()
            )
            z = F.normalize(z, p=2, dim=-1, eps=self.eps)
            if self.gather_distributed and get_world_size() > 1:
                shape = torch.tensor(z.shape, device=z.device)
                if any(not torch.equal(shape, other) for other in all_gather(shape)):
                    raise ValueError(
                        "Distributed MMCR requires equal [B, V, D] on all ranks"
                    )
                z = torch.cat(all_gather(z.contiguous()), dim=0)
            centroids = z.mean(dim=1)
            global_norm = torch.linalg.svdvals(centroids).sum()
            if self.lmbda == 0:
                return -global_norm
            local_norm = torch.linalg.svdvals(z).sum(dim=-1).mean()
            return self.lmbda * local_norm - global_norm
