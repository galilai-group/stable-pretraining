"""Learning From Randomness sample-wise redundancy-reduction loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import all_gather, get_rank, get_world_size


class BatchwiseBarlowTwinsLoss(nn.Module):
    """Match corresponding samples and suppress cross-sample similarity.

    Implements Eq. 5 of :cite:`sui2024lfr` for one projector. With cosine
    similarities ``c_ij`` between prediction ``i`` and target ``j``::

        loss = sum_i (1 - c_ii)^2 + lambd * sum_{i != j} c_ij^2

    Unlike :class:`BarlowTwinsLoss`, the similarity matrix is ``B x B``
    (samples), not ``D x D`` (features). Average over projectors for the full
    LFR objective, as the authors' code does. Similarities are computed in
    float32 so the ``B^2`` sum cannot overflow under mixed precision.

    Under DDP with ``gather=True`` (training), local predictions are compared
    against targets gathered from every rank, so negatives span the **global**
    batch. The local value is scaled by the world size: averaging it across
    ranks gives the global-batch loss, and DDP's averaged gradients equal those
    of a single process seeing the whole batch. Local batch sizes must then be
    equal across ranks. ``gather=False`` uses only the local batch, which is
    safe when ranks see different batch sizes (validation, testing), and
    accepts a single sample, for which only the alignment term remains.

    Args:
        lambd: Weight of the off-diagonal squared cosine similarities.
    """

    def __init__(self, lambd: float = 0.01) -> None:
        super().__init__()
        if lambd < 0:
            raise ValueError("lambd must be non-negative")
        self.lambd = lambd

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, gather: bool = True
    ) -> torch.Tensor:
        """Compute the sample-wise Barlow Twins objective.

        Args:
            prediction: Predictor outputs with shape ``[B, D]``.
            target: Random-projector outputs with shape ``[B, D]``.
            gather: Use targets from every rank as negatives (training).

        Returns:
            Scalar float32 loss, summed over samples.
        """
        if prediction.ndim != 2 or prediction.shape != target.shape:
            raise ValueError("prediction and target must have matching [B, D] shapes")
        world_size = get_world_size() if gather else 1
        if gather and prediction.shape[0] * world_size < 2:
            raise ValueError("the training batch must contain at least 2 samples")
        # Autocast would run the matmul below in half precision despite
        # ``.float()``; the B x B similarities and their sums stay float32.
        with torch.autocast(prediction.device.type, enabled=False):
            return self._loss(prediction, target, gather, world_size)

    def _loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        gather: bool,
        world_size: int,
    ) -> torch.Tensor:
        prediction = F.normalize(prediction.float(), dim=1)
        targets = F.normalize(target.float(), dim=1)
        offset = 0
        if gather:
            targets = torch.cat(all_gather(targets), dim=0)
            offset = get_rank() * prediction.shape[0]
        similarity = prediction @ targets.T
        local = torch.arange(prediction.shape[0], device=prediction.device)
        diagonal = similarity[local, offset + local]
        loss = (1 - diagonal).square().sum() + self.lambd * (
            similarity.square().sum() - diagonal.square().sum()
        )
        return loss * world_size
