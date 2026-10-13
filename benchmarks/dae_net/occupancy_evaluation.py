"""Counts for sampled occupancy reconstruction evaluation."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class OccupancyCounts:
    """Accumulate intersections and unions before computing a shape's IoU.

    Note:
        Query chunks need not have equal sizes. Both-empty occupancy has IoU 1.
    """

    intersection: int = 0
    union: int = 0
    queries: int = 0
    predicted_occupied: int = 0
    target_occupied: int = 0

    def update(
        self, prediction: Tensor, target: Tensor, threshold: float = 0.5
    ) -> None:
        """Add one chunk of occupancy predictions and binary targets.

        Args:
            prediction: Finite occupancy values with shape [Q, 1].
            target: Binary targets with the same shape and device.
            threshold: Strict lower bound for predicted occupied points.

        Raises:
            ValueError: If shapes, values, devices, or threshold are invalid.
        """
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        if (
            prediction.ndim != 2
            or prediction.shape[1] != 1
            or prediction.shape != target.shape
            or prediction.numel() == 0
            or prediction.device != target.device
        ):
            raise ValueError("Expected matching nonempty [Q, 1] tensors on one device")
        if (
            not torch.isfinite(prediction).all()
            or not ((prediction >= 0) & (prediction <= 1)).all()
        ):
            raise ValueError("Predictions must be finite occupancy values in [0, 1]")
        if not ((target == 0) | (target == 1)).all():
            raise ValueError("Targets must be binary")
        occupied = prediction > threshold
        truth = target.bool()
        self.intersection += int((occupied & truth).sum().item())
        self.union += int((occupied | truth).sum().item())
        self.queries += target.numel()
        self.predicted_occupied += int(occupied.sum().item())
        self.target_occupied += int(truth.sum().item())

    @property
    def iou(self) -> float:
        """Return accumulated IoU, with 1 for a nonempty all-background query set.

        Raises:
            ValueError: If no queries have been evaluated.
        """
        if not self.queries:
            raise ValueError("No queries have been evaluated")
        return self.intersection / self.union if self.union else 1.0
