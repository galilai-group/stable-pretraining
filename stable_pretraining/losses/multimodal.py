"""Multimodal SSL losses.

This module contains losses for multimodal self-supervised learning,
particularly for image-text contrastive learning like CLIP.
"""

import math

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
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


class SigLIPLoss(torch.nn.Module):
    """Sigmoid Loss for Language Image Pre-Training.

    Computes the pairwise SigLIP objective from
    :cite:`zhai2023sigmoid`. Positive image-text pairs are expected on
    the batch diagonal; all off-diagonal pairs are treated as negatives.

    Args:
        init_logit_scale: Initial value for the learnable log-space scale
            parameter. The effective multiplier is ``exp(logit_scale)``, so
            it remains positive during training. Defaults to ``log(10)``.
        init_logit_bias: Initial value for the learnable additive bias applied
            to every image-text logit. Defaults to ``-10``.
        normalize: Whether to L2-normalize image and text features before
            computing logits. Defaults to ``True``.
        gather_distributed: Whether to gather image and text features across
            distributed workers before computing the loss. Defaults to ``True``.
    """

    def __init__(
        self,
        init_logit_scale: float = math.log(10.0),
        init_logit_bias: float = -10.0,
        normalize: bool = True,
        gather_distributed: bool = True,
    ):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(init_logit_scale))
        self.logit_bias = torch.nn.Parameter(torch.tensor(init_logit_bias))
        self.normalize = normalize
        self.gather_distributed = gather_distributed

    def _gather_features(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gather_distributed and dist.is_available() and dist.is_initialized():
            image_features = torch.cat(dist_nn.all_gather(image_features), dim=0)
            text_features = torch.cat(dist_nn.all_gather(text_features), dim=0)
        return image_features, text_features

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        if image_features.shape[0] != text_features.shape[0]:
            raise ValueError(
                "SigLIPLoss expects paired image/text batches with the same batch size."
            )

        if self.normalize:
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)

        image_features, text_features = self._gather_features(
            image_features, text_features
        )

        scale = self.logit_scale.exp()
        logits = scale * (image_features @ text_features.T) + self.logit_bias

        labels = -torch.ones_like(logits)
        labels.fill_diagonal_(1)

        loss = -torch.nn.functional.logsigmoid(labels * logits).sum(dim=-1).mean()
        return loss
