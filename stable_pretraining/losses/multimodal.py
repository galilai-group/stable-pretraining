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
        # for CLIP, targets are always the diagonal
        targets = torch.arange(feats_i.size(0), device=feats_i.device)
        if torch.distributed.is_initialized():
            targets = targets + torch.distributed.get_rank() * feats_i.size(0)

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
