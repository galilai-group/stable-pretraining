"""Hierarchical contrastive loss used by TS2Vec."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TS2VecLoss(nn.Module):
    """Hierarchical instance and temporal contrastive objective.

    Args:
        alpha: Weight assigned to instance contrast at each scale. The
            complementary weight ``1 - alpha`` is assigned to temporal
            contrast.
        temporal_unit: Lowest hierarchy level at which temporal contrast is
            evaluated.

    The loss uses raw dot-product similarities, without feature normalization
    or temperature scaling, as in the TS2Vec reference implementation.

    Note:
        Instance negatives are local to each rank: embeddings are never gathered
        across processes. With per-rank batch size B, each anchor has 2(B - 1)
        instance negatives and one positive at each timestamp. DDP averages
        parameter gradients across ranks, but does not enlarge this negative
        pool. Temporal negatives are timestamps within the same local series.
        Thus DDP is not equivalent to a single-device global-batch objective;
        a per-rank batch size of one has zero instance loss.
    """

    def __init__(self, alpha: float = 0.5, temporal_unit: int = 0) -> None:
        super().__init__()
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a real number in [0, 1].")
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be finite and in [0, 1].")
        if (
            isinstance(temporal_unit, bool)
            or not isinstance(temporal_unit, int)
            or temporal_unit < 0
        ):
            raise ValueError("temporal_unit must be a non-negative integer.")
        self.alpha = float(alpha)
        self.temporal_unit = temporal_unit

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute hierarchical contrast between two aligned views.

        Args:
            z1: First representation with shape ``[batch, time, features]``.
            z2: Second aligned representation with the same shape.

        Returns:
            A scalar differentiable loss tensor.
        """
        if z1.ndim != 3 or z2.ndim != 3:
            raise ValueError("z1 and z2 must have shape [batch, time, features].")
        if z1.shape != z2.shape:
            raise ValueError("z1 and z2 must have identical shapes.")
        if any(size < 1 for size in z1.shape):
            raise ValueError("Batch, time, and feature dimensions must be non-empty.")
        if z1.device != z2.device:
            raise ValueError("z1 and z2 must be on the same device.")
        if z1.dtype != z2.dtype:
            raise ValueError("z1 and z2 must have the same dtype.")
        if not z1.is_floating_point() or not z2.is_floating_point():
            raise TypeError("z1 and z2 must have floating-point dtypes.")

        if z1.dtype in (torch.float16, torch.bfloat16) or z2.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            z1 = z1.float()
            z2 = z2.float()

        with torch.autocast(device_type=z1.device.type, enabled=False):
            return self._compute_loss(z1, z2)

    def _compute_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute the hierarchy with autocast disabled for raw dot products."""
        loss = (z1.sum() + z2.sum()) * 0
        scales = 0
        while z1.size(1) > 1:
            if self.alpha != 0:
                loss = loss + self.alpha * self._instance_contrastive_loss(z1, z2)
            if scales >= self.temporal_unit and self.alpha != 1:
                loss = loss + (1 - self.alpha) * self._temporal_contrastive_loss(z1, z2)
            scales += 1
            z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2, stride=2).transpose(
                1, 2
            )
            z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2, stride=2).transpose(
                1, 2
            )
        if self.alpha != 0:
            loss = loss + self.alpha * self._instance_contrastive_loss(z1, z2)
        scales += 1
        return loss / scales

    @staticmethod
    def _instance_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Contrast paired views across instances at every timestamp."""
        batch, _, _ = z1.shape
        if batch == 1:
            return (z1.sum() + z2.sum()) * 0
        z = torch.cat((z1, z2), dim=0).transpose(0, 1)
        similarity = torch.matmul(z, z.transpose(1, 2))
        logits = (
            torch.tril(similarity, diagonal=-1)[:, :, :-1]
            + torch.triu(similarity, diagonal=1)[:, :, 1:]
        )
        log_probs = -F.log_softmax(logits, dim=-1)
        indices = torch.arange(batch, device=z1.device)
        return (
            log_probs[:, indices, batch + indices - 1].mean()
            + log_probs[:, batch + indices, indices].mean()
        ) / 2

    @staticmethod
    def _temporal_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Contrast paired views across timestamps for every instance."""
        _, length, _ = z1.shape
        if length == 1:
            return (z1.sum() + z2.sum()) * 0
        z = torch.cat((z1, z2), dim=1)
        similarity = torch.matmul(z, z.transpose(1, 2))
        logits = (
            torch.tril(similarity, diagonal=-1)[:, :, :-1]
            + torch.triu(similarity, diagonal=1)[:, :, 1:]
        )
        log_probs = -F.log_softmax(logits, dim=-1)
        indices = torch.arange(length, device=z1.device)
        return (
            log_probs[:, indices, length + indices - 1].mean()
            + log_probs[:, length + indices, indices].mean()
        ) / 2
