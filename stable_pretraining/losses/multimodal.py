"""Multimodal SSL losses.

This module contains losses for multimodal self-supervised learning,
particularly for image-text contrastive learning like CLIP.
"""

import torch
from typing import Optional

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
        super().__init__(temperature=temperature)

    def forward(
        self,
        feats_i: torch.Tensor,
        feats_j: torch.Tensor,
        logit_scale: Optional[torch.Tensor | float] = None,
    ) -> torch.Tensor:
        # _compute all_gathers both sides -> logits [world*B, world*B]; targets must be length world*B.
        b = feats_i.size(0)
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            w = torch.distributed.get_world_size()
            targets = torch.arange(w * b, device=feats_i.device, dtype=torch.long)
        else:
            targets = torch.arange(b, device=feats_i.device, dtype=torch.long)

        # calculate loss in both directions
        loss_i = self._compute(
            anchors=feats_i,
            candidates=feats_j,
            targets=targets,
            logit_scale=logit_scale,
        )
        loss_j = self._compute(
            anchors=feats_j,
            candidates=feats_i,
            targets=targets,
            logit_scale=logit_scale,
        )

        return 0.5 * (loss_i + loss_j)
