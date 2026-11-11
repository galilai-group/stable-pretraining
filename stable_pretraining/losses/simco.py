"""SimCo's symmetric dual-temperature contrastive objective."""

import math

import torch
import torch.nn.functional as F

from ..utils import all_gather


class SimCoLoss(torch.nn.Module):
    """Compute the dual-temperature InfoNCE objective from paper equation (9).

    Each projection is L2-normalized. An anchor's positive is the corresponding
    row in the other view, and its negatives are the other rows of that view.
    The two directions are averaged, with gradients through both views.

    Args:
        intra_temperature: Positive temperature controlling the relative
            weights of negative samples for each anchor.
        inter_temperature: Positive temperature controlling the relative
            weights assigned to different anchors.

    Note:
        The ratio between the inter- and intra-temperature negative probability
        masses is detached, as required by equation (9). This is different from
        changing the temperature of an ordinary cross-entropy loss. Distributed
        training uses the entire differentiably gathered batch in both
        directions. The collective requires equal local batch sizes.

    References:
        Zhang et al., "Dual Temperature Helps Contrastive Learning Without Many
        Negative Samples: Towards Understanding and Simplifying MoCo", CVPR
        2022, https://arxiv.org/abs/2203.17248.
        Author implementation: https://github.com/ChaoningZhang/Dual-temperature,
        commit ``e8d5e768255721b23643d948631f80c0b8a0851b``.
    """

    def __init__(
        self,
        intra_temperature: float = 0.1,
        inter_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.intra_temperature = float(intra_temperature)
        self.inter_temperature = float(inter_temperature)
        if any(
            not math.isfinite(temperature) or temperature <= 0
            for temperature in (self.intra_temperature, self.inter_temperature)
        ):
            raise ValueError("SimCo temperatures must be finite and positive")

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Return the symmetric dual-temperature loss for paired projections.

        Args:
            z1: Raw first-view projections with shape ``(N, D)``.
            z2: Raw second-view projections with the same nonempty shape.

        Returns:
            A scalar loss. Half-precision inputs are computed in float32, and
            float64 inputs retain their precision.

        Raises:
            ValueError: If input shapes or devices differ, either axis is
                empty, or the gathered batch contains fewer than two samples.
            TypeError: If inputs are not real floating-point tensors.
        """
        if z1.ndim != 2 or z2.ndim != 2 or z1.shape != z2.shape:
            raise ValueError(
                "SimCo inputs must have equal two-dimensional (N, D) shapes"
            )
        if z1.shape[0] == 0 or z1.shape[1] == 0:
            raise ValueError("SimCo inputs must have nonempty batch and feature axes")
        if not z1.is_floating_point() or not z2.is_floating_point():
            raise TypeError("SimCo inputs must be real floating-point tensors")
        if z1.device != z2.device:
            raise ValueError("SimCo inputs must be on the same device")

        dtype = torch.promote_types(z1.dtype, z2.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        # The probability ratio is sensitive to autocast rounding near p_pos=1.
        with torch.autocast(device_type=z1.device.type, enabled=False):
            x = F.normalize(z1.to(dtype=dtype), dim=1)
            y = F.normalize(z2.to(dtype=dtype), dim=1)
            x = torch.cat(all_gather(x), dim=0)
            y = torch.cat(all_gather(y), dim=0)
            if x.shape[0] < 2:
                raise ValueError(
                    "SimCo requires at least two samples in the global batch"
                )
            similarities = x @ y.T
            loss = (self._direction(similarities) + self._direction(similarities.T)) / 2
        return loss

    def _direction(self, similarities: torch.Tensor) -> torch.Tensor:
        """Evaluate one direction without subtracting a saturated probability."""
        positive = similarities.diagonal()
        mask = torch.eye(
            similarities.shape[0], device=similarities.device, dtype=torch.bool
        )
        negatives = similarities.masked_fill(mask, -torch.inf)
        intra_gap = torch.logsumexp(negatives / self.intra_temperature, dim=1)
        intra_gap = intra_gap - positive / self.intra_temperature
        inter_gap = torch.logsumexp(negatives / self.inter_temperature, dim=1)
        inter_gap = inter_gap - positive / self.inter_temperature
        log_ratio = (F.logsigmoid(inter_gap) - F.logsigmoid(intra_gap)).detach()

        # Below log(eps), log(softplus(x)) equals x to working precision.
        # Keeping the weighted CE in log space also avoids inf * 0 for easy pairs.
        threshold = math.log(torch.finfo(intra_gap.dtype).eps)
        log_cross_entropy = torch.where(
            intra_gap < threshold,
            intra_gap,
            F.softplus(intra_gap.clamp_min(threshold)).log(),
        )
        return (log_ratio + log_cross_entropy).exp().mean()
