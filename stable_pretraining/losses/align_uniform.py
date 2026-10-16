"""Alignment and uniformity loss for normalized representations."""

import math

import torch
import torch.nn.functional as F


class AlignUniformLoss(torch.nn.Module):
    """Alignment and uniformity objective from Wang and Isola.

    Args:
        alignment_weight: Weight applied to the alignment term.
        uniformity_weight: Weight applied to the uniformity term.
        alpha: Exponent used for the positive-pair distance.
        t: Temperature in the Gaussian potential used for uniformity.

    Note:
        Both terms are computed on L2-normalized embeddings. Uniformity is
        computed within each view over distinct samples, then averaged across
        the two views.
    """

    def __init__(
        self,
        alignment_weight: float = 1.0,
        uniformity_weight: float = 1.0,
        alpha: float = 2.0,
        t: float = 2.0,
    ):
        super().__init__()
        if (
            not math.isfinite(alignment_weight)
            or not math.isfinite(uniformity_weight)
            or alignment_weight < 0
            or uniformity_weight < 0
        ):
            raise ValueError(
                "alignment_weight and uniformity_weight must be non-negative"
            )
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be positive")
        if not math.isfinite(t) or t <= 0:
            raise ValueError("t must be positive")

        self.alignment_weight = alignment_weight
        self.uniformity_weight = uniformity_weight
        self.alpha = alpha
        self.t = t

    def loss_and_terms(
        self, z_i: torch.Tensor, z_j: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the weighted loss and its alignment/uniformity components.

        Args:
            z_i: Projected embeddings from the first view, shaped ``[B, D]``.
            z_j: Projected embeddings from the second view, shaped ``[B, D]``.

        Returns:
            A tuple containing the weighted scalar loss, alignment term, and
            uniformity term.
        """
        if z_i.ndim != 2 or z_j.ndim != 2:
            raise ValueError("z_i and z_j must be two-dimensional [B, D] tensors")
        if z_i.shape != z_j.shape:
            raise ValueError(
                f"z_i and z_j must have the same shape, got {z_i.shape} and {z_j.shape}"
            )
        if z_i.shape[0] == 0 or z_i.shape[1] == 0:
            raise ValueError(
                "z_i and z_j must have non-empty batch and feature dimensions"
            )
        if z_i.shape[0] < 2:
            raise ValueError("uniformity requires at least two samples per view")

        dtype = torch.promote_types(
            torch.promote_types(z_i.dtype, z_j.dtype), torch.float32
        )
        z_i = F.normalize(z_i.to(dtype=dtype), dim=-1)
        z_j = F.normalize(z_j.to(dtype=dtype), dim=-1)

        alignment = (z_i - z_j).norm(p=2, dim=-1).pow(self.alpha).mean()

        uniformities = []
        for embeddings in (z_i, z_j):
            pairwise_distances = torch.pdist(embeddings, p=2).square()
            log_mean_potential = torch.logsumexp(
                -self.t * pairwise_distances, dim=0
            ) - math.log(pairwise_distances.numel())
            uniformities.append(log_mean_potential)
        uniformity = torch.stack(uniformities).mean()

        loss = self.alignment_weight * alignment + self.uniformity_weight * uniformity
        return loss, alignment, uniformity

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute the weighted alignment and uniformity loss.

        Args:
            z_i: Projected embeddings from the first view, shaped ``[B, D]``.
            z_j: Projected embeddings from the second view, shaped ``[B, D]``.

        Returns:
            The weighted scalar loss.
        """
        loss, _, _ = self.loss_and_terms(z_i, z_j)
        return loss
