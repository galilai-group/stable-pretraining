"""Cross-view contrastive objective from DirectCLR (Jing et al., ICLR 2022)."""

import math

import torch
import torch.distributed as dist

from ..utils.distributed import all_gather, get_rank, get_world_size
from .joint_embedding import InfoNCELoss


class DirectCLRLoss(InfoNCELoss):
    """Symmetric InfoNCE with candidates from the opposite view only.

    Unlike NT-Xent, this objective has no same-view negatives. Inputs are the
    selected backbone subspaces; feature selection belongs to the method.

    Args:
        temperature: Positive finite softmax temperature, 0.1 in the reference.

    Note:
        DDP uses local anchors and autograd-aware global candidates. Every rank
        must supply the same nonempty batch size and feature width. With DDP's
        gradient averaging this matches the global-batch objective. Half and
        bfloat16 inputs are evaluated in float32, including under autocast.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        super().__init__(temperature=temperature)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute the average of the two cross-view cross-entropies.

        Args:
            z1: First-view features of shape ``[N, D]``.
            z2: Aligned second-view features of shape ``[N, D]``.

        Returns:
            Scalar contrastive loss; labels are sample positions, not classes.

        Raises:
            ValueError: Shapes, devices, dtypes, or distributed sizes disagree.
        """
        if z1.ndim != 2 or z1.shape != z2.shape or min(z1.shape) < 1:
            raise ValueError("features must have identical nonempty [N, D] shapes")
        if z1.device != z2.device or z1.dtype != z2.dtype:
            raise ValueError("features must have matching devices and dtypes")
        if not z1.is_floating_point():
            raise ValueError("features must be floating point")
        if get_world_size() > 1:
            shape = torch.tensor(z1.shape, device=z1.device)
            shapes = [torch.empty_like(shape) for _ in range(get_world_size())]
            dist.all_gather(shapes, shape)
            if any(not torch.equal(shape, other) for other in shapes):
                raise ValueError("DirectCLR requires equal [N, D] shapes on every rank")

        # Disabling autocast prevents the similarity matrix from silently using
        # half precision even after explicit input promotion.
        with torch.autocast(device_type=z1.device.type, enabled=False):
            if z1.dtype in (torch.float16, torch.bfloat16):
                z1, z2 = z1.float(), z2.float()
            candidates = torch.cat(all_gather(torch.stack((z1, z2), dim=1)), dim=0)
            targets = torch.arange(z1.shape[0], device=z1.device)
            targets = targets + get_rank() * z1.shape[0]
            return (
                self._compute(z1, candidates[:, 1], targets)
                + self._compute(z2, candidates[:, 0], targets)
            ) / 2
