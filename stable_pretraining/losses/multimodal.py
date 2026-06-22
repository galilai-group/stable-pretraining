"""Multimodal SSL losses.

This module contains losses for multimodal self-supervised learning,
particularly for image-text contrastive learning like CLIP.
"""

import torch
from typing import Optional

from ..utils import all_gather, get_rank
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
        # Under DDP, each local anchor is scored against the candidates gathered
        # from all ranks; the positive (its paired feature) sits at the diagonal
        # offset into this rank's block of the global candidate set. In a single
        # process this is just the diagonal (rank 0, gather is a no-op).
        batch = feats_i.size(0)
        offset = get_rank() * batch
        targets = offset + torch.arange(batch, device=feats_i.device)

        candidates_j = torch.cat(all_gather(feats_j), dim=0)
        candidates_i = torch.cat(all_gather(feats_i), dim=0)

        # calculate loss in both directions
        loss_i = self._compute(
            anchors=feats_i,
            candidates=candidates_j,
            targets=targets,
            logit_scale=logit_scale,
        )
        loss_j = self._compute(
            anchors=feats_j,
            candidates=candidates_i,
            targets=targets,
            logit_scale=logit_scale,
        )

        return 0.5 * (loss_i + loss_j)
