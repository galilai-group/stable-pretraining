"""All4One objectives from Estepa et al. (ICCV 2023).

Reference: https://github.com/ImaGonEs/all4one/blob/main/solo/methods/all4one.py
"""

import math

import torch
import torch.nn.functional as F

from ..utils import all_gather, get_rank
from .joint_embedding import InfoNCELoss
from .utils import off_diagonal


class All4OneLoss(torch.nn.Module):
    """Combine neighbour, centroid, and feature contrast objectives.

    Args:
        temperature: Neighbour contrast temperature. Default is 0.2.
        centroid_temperature: Centroid contrast temperature. Default is 0.1.
        neighbour_weight: Weight of the symmetric neighbour loss.
        centroid_weight: Weight of the symmetric centroid loss.
        feature_diagonal_weight: Weight of the diagonal RMS error.
        feature_off_diagonal_weight: Weight of the off-diagonal RMS error.

    Note:
        Defaults match the released code: ``0.5 * neighbour + 0.5 * centroid
        + 2.5 * diagonal_rms + 2.5 * off_diagonal_rms``. Equations 3--4 in
        the paper instead imply a diagonal weight of 5.0.

        Contrastive candidates are gathered with gradients across ranks;
        equal local batch sizes are required. Feature correlations use the
        local batch, matching the reference implementation. Momentum projections
        and queue neighbours are detached here, but neither centroid branch is.
    """

    def __init__(
        self,
        temperature: float = 0.2,
        centroid_temperature: float = 0.1,
        neighbour_weight: float = 0.5,
        centroid_weight: float = 0.5,
        feature_diagonal_weight: float = 2.5,
        feature_off_diagonal_weight: float = 2.5,
    ) -> None:
        super().__init__()
        for name, value in (
            ("temperature", temperature),
            ("centroid_temperature", centroid_temperature),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("neighbour_weight", neighbour_weight),
            ("centroid_weight", centroid_weight),
            ("feature_diagonal_weight", feature_diagonal_weight),
            ("feature_off_diagonal_weight", feature_off_diagonal_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.neighbour_contrast = InfoNCELoss(temperature)
        self.centroid_contrast = InfoNCELoss(centroid_temperature)
        self.neighbour_weight = neighbour_weight
        self.centroid_weight = centroid_weight
        self.feature_diagonal_weight = feature_diagonal_weight
        self.feature_off_diagonal_weight = feature_off_diagonal_weight

    def _contrast(
        self,
        anchors: torch.Tensor,
        predictions: torch.Tensor,
        objective: InfoNCELoss,
    ) -> torch.Tensor:
        candidates = torch.cat(all_gather(predictions), dim=0)
        targets = torch.arange(anchors.size(0), device=anchors.device)
        targets = targets + get_rank() * anchors.size(0)
        return objective(anchors, candidates, targets)

    def forward(
        self,
        *,
        projections: tuple[torch.Tensor, torch.Tensor],
        momentum_projections: tuple[torch.Tensor, torch.Tensor],
        neighbours: tuple[torch.Tensor, torch.Tensor],
        predictions: tuple[torch.Tensor, torch.Tensor],
        centroids: tuple[torch.Tensor, torch.Tensor],
        shifted_centroids: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute the objective from two views' representations.

        Args:
            projections: Online projector outputs, in view order.
            momentum_projections: Momentum projector outputs, in view order.
                Row normalization is performed here before feature normalization.
            neighbours: The single closest queue vector for each view.
            predictions: Neighbour predictor outputs for each view.
            centroids: Transformer outputs for the unshifted neighbour sequences.
            shifted_centroids: Transformer outputs for sequences beginning with
                the online centroid prediction.

        Returns:
            Dictionary with the total ``loss``, unweighted ``neighbour_loss``
            and ``centroid_loss``, weighted ``feature_loss``, and unweighted
            ``feature_diagonal`` and ``feature_off_diagonal`` RMS errors.

        Note:
            Each argument contains two tensors of the same shape ``[N, D]``,
            with ``N >= 1`` and ``D >= 2``. Half precision inputs are promoted
            to float32 and autocast is disabled for the loss computation.
            RMS errors use vector norms, selecting a finite zero subgradient
            at zero error instead of the undefined gradient of ``sqrt(0)``.
        """
        pairs = (
            projections,
            momentum_projections,
            neighbours,
            predictions,
            centroids,
            shifted_centroids,
        )
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError("Each input must contain exactly two views")
        shape = projections[0].shape
        if len(shape) != 2 or shape[0] < 1 or shape[1] < 2:
            raise ValueError("Inputs must have shape [N, D] with N >= 1 and D >= 2")
        if any(tensor.shape != shape for pair in pairs for tensor in pair):
            raise ValueError("All inputs must have the same [N, D] shape")

        with torch.autocast(device_type=projections[0].device.type, enabled=False):
            pairs = tuple(
                tuple(
                    tensor.float()
                    if tensor.dtype in (torch.float16, torch.bfloat16)
                    else tensor
                    for tensor in pair
                )
                for pair in pairs
            )
            (z1, z2), momentum, nearest, (p1, p2), (c1, c2), (s1, s2) = pairs
            m1, m2 = (F.normalize(z.detach(), dim=-1) for z in momentum)
            n1, n2 = (z.detach() for z in nearest)

            neighbour_loss = 0.5 * (
                self._contrast(n1, p2, self.neighbour_contrast)
                + self._contrast(n2, p1, self.neighbour_contrast)
            )
            centroid_loss = 0.5 * (
                self._contrast(c1, s2, self.centroid_contrast)
                + self._contrast(c2, s1, self.centroid_contrast)
            )
            correlations = (
                F.normalize(m1, dim=0).T @ F.normalize(z2, dim=0),
                F.normalize(m2, dim=0).T @ F.normalize(z1, dim=0),
            )
            diagonal = torch.cat([c.diagonal() - 1 for c in correlations])
            off = torch.cat([off_diagonal(c) for c in correlations])
            diagonal_rms = torch.linalg.vector_norm(diagonal) / math.sqrt(
                diagonal.numel()
            )
            off_diagonal_rms = torch.linalg.vector_norm(off) / math.sqrt(off.numel())
            feature_loss = (
                self.feature_diagonal_weight * diagonal_rms
                + self.feature_off_diagonal_weight * off_diagonal_rms
            )
            loss = (
                self.neighbour_weight * neighbour_loss
                + self.centroid_weight * centroid_loss
                + feature_loss
            )
        return {
            "loss": loss,
            "neighbour_loss": neighbour_loss,
            "centroid_loss": centroid_loss,
            "feature_loss": feature_loss,
            "feature_diagonal": diagonal_rms,
            "feature_off_diagonal": off_diagonal_rms,
        }
