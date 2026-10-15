"""Multimodal SSL losses.

This module contains losses for multimodal self-supervised learning,
particularly for image-text contrastive learning like CLIP.
"""

import math

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


class SigLIPLoss(torch.nn.Module):
    """Sigmoid loss for language-image pre-training (SigLIP).

    As used in SigLIP :cite:`zhai2023sigmoid`. Every image-text pair is an
    independent binary classification: positives (matching pairs, aligned on
    the batch diagonal) get label ``+1`` and all other pairs get ``-1``. The
    loss is ``-sum(log sigmoid(label * logit)) / batch_size`` with
    ``logit = exp(logit_scale) * <img, txt> + logit_bias``.

    Under DDP, each rank scores its local images against the text features
    gathered from all ranks. Because the loss is a sum over independent pairs,
    the mean of the per-rank losses equals the single-process loss on the
    global batch, and each pair is computed exactly once across ranks.

    Args:
        init_logit_scale: Initial value of the learnable log-space scale. The
            effective multiplier is ``exp(logit_scale)``. Defaults to
            ``log(10)`` (paper default).
        init_logit_bias: Initial value of the learnable logit bias. Defaults to
            ``-10`` (paper default), which keeps the initial loss from being
            dominated by the many negatives.
        normalize: L2-normalize image and text features before computing
            logits. Defaults to ``True``.
        gather_distributed: Gather text features across ranks so every image
            sees negatives from the global batch. Defaults to ``True``.
    """

    def __init__(
        self,
        init_logit_scale: float = math.log(10.0),
        init_logit_bias: float = -10.0,
        normalize: bool = True,
        gather_distributed: bool = True,
    ):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(float(init_logit_scale)))
        self.logit_bias = torch.nn.Parameter(torch.tensor(float(init_logit_bias)))
        self.normalize = normalize
        self.gather_distributed = gather_distributed

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the SigLIP loss.

        Args:
            image_features: Image embeddings ``[B, D]``.
            text_features: Text embeddings ``[B, D]``, paired row-wise with
                ``image_features``.

        Returns:
            Scalar loss: summed pairwise sigmoid loss divided by the local
            batch size.
        """
        if image_features.shape[0] != text_features.shape[0]:
            raise ValueError(
                "SigLIPLoss expects paired image/text batches with the same batch size."
            )

        if self.normalize:
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)

        batch = image_features.size(0)
        if self.gather_distributed:
            # The positive for local image i sits at this rank's offset into
            # the gathered text block. In a single process the gather is a
            # no-op and the offset is 0.
            candidates = torch.cat(all_gather(text_features), dim=0)
            offset = get_rank() * batch
        else:
            candidates = text_features
            offset = 0

        logits = (
            self.logit_scale.exp() * (image_features @ candidates.T) + self.logit_bias
        )

        labels = -torch.ones_like(logits)
        rows = torch.arange(batch, device=logits.device)
        labels[rows, offset + rows] = 1.0

        return -torch.nn.functional.logsigmoid(labels * logits).sum() / batch
