# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from facebookresearch/capi at 98b4fa17ee8eec8810c17022df9a27a44845368b.
# Licensed under Apache-2.0; see ../backbone/licenses/CAPI-Apache-2.0.txt.
"""CAPI position-wise online clustering and masked prediction loss."""

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def positionwise_sinkhorn(
    logits: torch.Tensor, iterations: int = 3, eps: float = 1e-8
) -> torch.Tensor:
    """Balance each patch position across samples and distributed ranks.

    Args:
        logits: Temperature-scaled logits of shape (batch, positions, clusters).
        iterations: Alternating normalization count.
        eps: Reference denominator stabilizer.

    Returns:
        Detached soft assignments with the input shape, in float32.

    Note:
        A per-cluster maximum shift and exponent offset of 50 reproduce the
        reference modified Sinkhorn, while avoiding reduced-precision overflow.
        All ranks must have the same spatial and cluster dimensions.
    """
    if logits.ndim != 3 or min(logits.shape) < 1 or iterations < 1:
        raise ValueError("Expected nonempty (batch, positions, clusters) logits")
    distributed = torch.distributed.is_initialized()
    values = logits.detach().float().transpose(0, 1)
    shift = values.amax(dim=1, keepdim=True)
    if distributed:
        torch.distributed.all_reduce(shift, op=torch.distributed.ReduceOp.MAX)
    q = (values - shift + 50).exp()
    for _ in range(iterations):
        mass = q.sum(dim=1, keepdim=True)
        if distributed:
            torch.distributed.all_reduce(mass)
        q = q / (mass + eps)
        q = q / (q.sum(dim=-1, keepdim=True) + eps)
    return q.transpose(0, 1).contiguous()


class CAPILoss(nn.Module):
    """Separate clustering and prediction cross-entropies with detached targets.

    Args:
        student_temp: Prediction head temperature.
        target_temp: Balanced assignment temperature.
        clustering_temp: Clustering head prediction temperature.
        sinkhorn_iterations: Position-wise balancing iterations.
    """

    def __init__(
        self,
        student_temp: float = 0.12,
        target_temp: float = 0.06,
        clustering_temp: float = 0.12,
        sinkhorn_iterations: int = 3,
    ) -> None:
        super().__init__()
        if min(student_temp, target_temp, clustering_temp) <= 0:
            raise ValueError("Temperatures must be positive")
        if sinkhorn_iterations < 1:
            raise ValueError("sinkhorn_iterations must be positive")
        self.student_temp = student_temp
        self.target_temp = target_temp
        self.clustering_temp = clustering_temp
        self.sinkhorn_iterations = sinkhorn_iterations

    def clustering(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return balanced assignments and the clustering-only loss.

        Args:
            logits: Clustering logits (batch, positions, clusters).

        Returns:
            Detached assignments and scalar clustering cross-entropy.
        """
        targets = positionwise_sinkhorn(
            logits.float() / self.target_temp, self.sinkhorn_iterations
        )
        loss = (
            -(targets * F.log_softmax(logits.float() / self.clustering_temp, -1))
            .sum(-1)
            .mean()
        )
        return targets, loss

    def prediction(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute masked prediction cross-entropy without target gradients.

        Args:
            logits: Student predictions (..., clusters).
            targets: Selected teacher assignments, matching logits.

        Returns:
            Scalar prediction loss.
        """
        return (
            -(
                targets.detach().float()
                * F.log_softmax(logits.float() / self.student_temp, -1)
            )
            .sum(-1)
            .mean()
        )

    def forward(
        self,
        predictions: torch.Tensor,
        clustering_logits: torch.Tensor,
        predict_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate both terms using flattened batch-position prediction indices.

        Args:
            predictions: Student logits (predicted positions, clusters).
            clustering_logits: Teacher clustering logits (batch, positions, clusters).
            predict_indices: Flattened batch-position indices.

        Returns:
            Total and component losses plus detached assignments.
        """
        targets, clustering_loss = self.clustering(clustering_logits)
        selected = targets.flatten(0, 1)[predict_indices]
        prediction_loss = self.prediction(predictions, selected)
        return {
            "loss": prediction_loss + clustering_loss,
            "prediction_loss": prediction_loss,
            "clustering_loss": clustering_loss,
            "assignments": targets,
        }


__all__ = ["CAPILoss", "positionwise_sinkhorn"]
