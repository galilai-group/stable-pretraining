"""Multimodal SSL losses.

This module contains losses for multimodal self-supervised learning,
particularly for image-text contrastive learning like CLIP.
"""

import math
from typing import Optional

import torch
import torch.distributed as dist

from ..utils.distributed import all_gather, get_rank, get_world_size
from .joint_embedding import InfoNCELoss


class CLIPLoss(InfoNCELoss):
    """CLIP loss (symmetric bidirectional InfoNCE).

    As used in CLIP :cite:`radford2021learning`.
    Computes symmetric cross-entropy over image-text and text-image logits.

    Args:
        temperature (float, optional): Softmax temperature. Default is 0.07.
            (If you use a learnable logit_scale in your model, pass it to
            forward(...) and this temperature will be ignored.)
    """

    def __init__(self, temperature: float = 0.07):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        super().__init__(temperature=temperature)

    def forward(
        self,
        feats_i: torch.Tensor,
        feats_j: torch.Tensor,
        logit_scale: Optional[torch.Tensor | float] = None,
    ) -> torch.Tensor:
        """Score aligned pairs against all opposite-view candidates.

        Args:
            feats_i: First-view floating point features with shape [N, D].
            feats_j: Second-view features of the same shape.
            logit_scale: Optional differentiable denominator overriding temperature.

        Returns:
            Mean of the two directional cross-entropies.
        """
        if (
            feats_i.ndim != 2
            or feats_i.shape != feats_j.shape
            or min(feats_i.shape) < 1
        ):
            raise ValueError("features must have identical nonempty [N, D] shapes")
        if not feats_i.is_floating_point() or not feats_j.is_floating_point():
            raise ValueError("features must be floating point")
        if feats_i.device != feats_j.device:
            raise ValueError("features must be on the same device")

        world = get_world_size()
        if world > 1:
            shape = torch.tensor(feats_i.shape, device=feats_i.device)
            shapes = [torch.empty_like(shape) for _ in range(world)]
            dist.all_gather(shapes, shape)
            if any(not torch.equal(shape, other) for other in shapes):
                raise ValueError("CLIPLoss requires equal [N, D] shapes on every rank")

        # Half-precision leaves also receive half-precision gradients. A tiny
        # normalization floor can overflow that cast even with float32 logits.
        half = torch.float16 in (feats_i.dtype, feats_j.dtype)
        eps = torch.finfo(torch.float16).eps if half else 1e-12
        dtype = torch.promote_types(
            torch.float32, torch.promote_types(feats_i.dtype, feats_j.dtype)
        )
        with torch.autocast(device_type=feats_i.device.type, enabled=False):
            first, second = feats_i.to(dtype), feats_j.to(dtype)
            global_first, global_second = first, second
            targets = torch.arange(len(first), device=first.device)
            if world > 1:
                pairs = torch.stack((first, second), dim=1)
                global_first, global_second = torch.cat(all_gather(pairs)).unbind(dim=1)
                targets = targets + get_rank() * len(first)
            first_loss = self._compute(
                first, global_second, targets, logit_scale=logit_scale, eps=eps
            )
            second_loss = self._compute(
                second, global_first, targets, logit_scale=logit_scale, eps=eps
            )
            return (first_loss + second_loss) / 2
