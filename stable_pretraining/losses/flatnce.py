"""FlatNCE: self-normalized contrastive learning with negative-only normalization."""

import math

import torch
import torch.nn.functional as F

from ..utils import all_gather, get_rank, get_world_size


class FlatNCELoss(torch.nn.Module):
    """Compute the symmetric, two-view FlatNCE objective.

    Implements Algorithm 1 and Equation 6 of Chen et al.,
    https://arxiv.org/abs/2107.01152. Both augmented views serve as anchors;
    self-pairs and positive pairs are excluded from the negative partition.

    Args:
        temperature: Finite, strictly positive cosine-similarity temperature.

    Note:
        The forward value is one, but the detached normalization gives a
        nonzero gradient. Loss-value convergence and finite-difference checks
        of this expression do not measure learning. Distributed candidates
        are gathered with autograd; all ranks must have equal batch sizes.
        Half-precision inputs and autocast use float32 loss arithmetic.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and strictly positive")
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Contrast paired projections against all other images' views.

        Args:
            z_i: First-view projections of shape ``[B, D]``.
            z_j: Second-view projections of the same shape, dtype, and device.

        Returns:
            Scalar mean FlatNCE loss, numerically equal to one.

        Raises:
            ValueError: If shapes, dtypes, or devices differ, dimensions are
                empty, inputs are not floating point, or no negatives exist.
        """
        if z_i.ndim != 2 or z_i.shape != z_j.shape:
            raise ValueError("projections must have the same [B, D] shape")
        if z_i.size(0) == 0 or z_i.size(1) == 0:
            raise ValueError("projection batch and feature dimensions must be nonempty")
        if not z_i.is_floating_point() or not z_j.is_floating_point():
            raise ValueError("projections must be floating-point tensors")
        if z_i.device != z_j.device or z_i.dtype != z_j.dtype:
            raise ValueError("projections must have the same device and dtype")
        if z_i.size(0) * get_world_size() < 2:
            raise ValueError(
                "FlatNCE requires at least two images globally for negatives"
            )

        # Disable autocast around matmul as well as logsumexp: casting inputs
        # alone would still let autocast reduce the similarity precision.
        with torch.autocast(device_type=z_i.device.type, enabled=False):
            anchors = torch.cat([z_i, z_j], dim=0)
            if anchors.dtype in (torch.float16, torch.bfloat16):
                anchors = anchors.float()
            anchors = F.normalize(anchors, dim=-1)
            candidates = torch.cat(all_gather(anchors), dim=0)
            logits = (anchors @ candidates.T) / self.temperature

            two_b = anchors.size(0)
            rows = torch.arange(two_b, device=anchors.device)
            offset = get_rank() * two_b
            positives = offset + (rows + z_i.size(0)) % two_b
            contrasts = logits - logits[rows, positives].unsqueeze(1)
            excluded = torch.zeros_like(logits, dtype=torch.bool)
            excluded[rows, offset + rows] = True
            excluded[rows, positives] = True
            log_partition = contrasts.masked_fill(excluded, -torch.inf).logsumexp(dim=1)
            return torch.exp(log_partition - log_partition.detach()).mean()
