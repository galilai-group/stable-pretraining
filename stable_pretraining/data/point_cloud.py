"""Reusable XYZ point-cloud preprocessing."""

import torch
from torch.utils.data import default_collate

from ..backbone.point_cloud import farthest_point_indices


class PointCloudTransform:
    """Subsample, scale, center, normalize, and optionally rotate XYZ points.

    Args:
        num_points: Number of points returned per object.
        training: Enable random scaling and vertical-axis rotation.
    """

    def __init__(self, num_points: int = 1024, training: bool = True) -> None:
        self.num_points = num_points
        self.training = training

    def __call__(self, sample: dict) -> dict:
        """Preprocess a point-cloud sample.

        Args:
            sample: Dictionary containing XYZ ``points`` [N,3] or [B,N,3].

        Returns:
            Sample with normalized ``points`` [num_points,3].
        """
        points = sample["points"]
        single_cloud = points.ndim == 2
        if single_cloud:
            points = points.unsqueeze(0)
        if points.shape[1] < self.num_points:
            raise ValueError("Too few points for the requested FPS subsampling")
        if points.shape[1] > self.num_points:
            indices = farthest_point_indices(points.float(), self.num_points)
            points = points[
                torch.arange(len(points), device=points.device)[:, None], indices
            ]
        if self.training:
            scale = torch.empty(3, device=points.device).uniform_(0.8, 1.2)
            points = points * scale
        points = points - points.mean(1, keepdim=True)
        points = points / points.norm(dim=-1).max(1, keepdim=True).values.unsqueeze(-1)
        if self.training:
            angle = torch.rand((), device=points.device) * (2 * torch.pi) - torch.pi
            c, s = torch.cos(angle), torch.sin(angle)
            zero, one = torch.zeros_like(c), torch.ones_like(c)
            rotation = torch.stack(
                [
                    torch.stack([c, zero, s]),
                    torch.stack([zero, one, zero]),
                    torch.stack([-s, zero, c]),
                ]
            )
            points = points @ rotation.T
        sample["points"] = points[0] if single_cloud else points
        return sample

    def collate(self, samples: list[dict]) -> dict:
        """Collate then transform so scaling and rotation are shared by a batch.

        Args:
            samples: Dataset samples with equally sized clouds.

        Returns:
            Collated dictionary with FPS-subsampled and normalized XYZ clouds.
        """
        return self(default_collate(samples))
