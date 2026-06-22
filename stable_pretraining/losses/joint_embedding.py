"""Joint embedding SSL losses.

This module contains joint embedding methods that learn to embed different views
of the same image close together in representation space. Includes both contrastive
(NTXentLoss) and non-contrastive (BYOL, VICReg, Barlow Twins) methods.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from ..utils import all_gather, all_reduce, get_rank, get_world_size
from .utils import off_diagonal, VCRegLoss


class BYOLLoss(torch.nn.Module):
    """Normalized MSE objective used in BYOL :cite:`grill2020bootstrap`.

    Computes the mean squared error between L2-normalized online predictions
    and L2-normalized target projections.
    """

    def forward(
        self, online_pred: torch.Tensor, target_proj: torch.Tensor
    ) -> torch.Tensor:
        """Compute BYOL loss.

        Args:
            online_pred: Predictions from the online network predictor.
            target_proj: Projections from the target network (no gradient).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        online_pred = F.normalize(online_pred, dim=-1, p=2)
        target_proj = F.normalize(target_proj, dim=-1, p=2)
        loss = 2 - 2 * (online_pred * target_proj).sum(dim=-1)
        return loss.mean()


class VICRegLoss(torch.nn.Module):
    """SSL objective used in VICReg :cite:`bardes2021vicreg`.

    Args:
        sim_coeff (float, optional): The weight of the similarity loss (attractive term).
            Default is 25.
        std_coeff (float, optional): The weight of the standard deviation loss.
            Default is 25.
        cov_coeff (float, optional): The weight of the covariance loss.
            Default is 1.
        epsilon (float, optional): Small value to avoid division by zero.
            Default is 1e-4.
    """

    def __init__(
        self,
        sim_coeff: float = 25,
        std_coeff: float = 25,
        cov_coeff: float = 1,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.vcreg_loss = VCRegLoss(std_coeff, cov_coeff, epsilon)

    def forward(self, z_i, z_j):
        """Compute the loss of the VICReg model.

        Args:
            z_i (torch.Tensor): Latent representation of the first augmented view of the batch.
            z_j (torch.Tensor): Latent representation of the second augmented view of the batch.

        Returns:
            float: The computed loss.
        """
        repr_loss = F.mse_loss(z_i, z_j)

        loss = self.sim_coeff * repr_loss + self.vcreg_loss(z_i) + self.vcreg_loss(z_j)
        return loss


class BarlowTwinsLoss(torch.nn.Module):
    """SSL objective used in Barlow Twins :cite:`zbontar2021barlow`.

    Args:
        lambd (float, optional): The weight of the off-diagonal terms in the loss.
            Default is 5e-3.
    """

    def __init__(self, lambd: float = 5e-3):
        super().__init__()
        self.lambd = lambd

    @staticmethod
    def _normalize(z: torch.Tensor) -> torch.Tensor:
        """Standardize along the batch dimension (zero mean, unit std)."""
        return (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-5)

    def forward(self, z_i, z_j):
        """Compute the loss of the Barlow Twins model.

        Args:
            z_i (torch.Tensor): Latent representation of the first augmented view of the batch.
            z_j (torch.Tensor): Latent representation of the second augmented view of the batch.

        Returns:
            float: The computed loss.
        """
        c = self._normalize(z_i).T @ self._normalize(
            z_j
        )  # normalize along the batch dimension
        # Normalize by the GLOBAL batch size, then sum the per-rank partial
        # cross-correlation matrices. In a single process get_world_size() == 1
        # and all_reduce is a no-op, so this is identical to the previous
        # ``c / z_i.size(0)``; under DDP it recovers the true global
        # cross-correlation (matching the reference Barlow Twins
        # implementation), whereas the old ``all_reduce(c)`` discarded its
        # result and was a silent no-op.
        c = c / (z_i.size(0) * get_world_size())
        c = all_reduce(c)

        on_diag = (torch.diagonal(c) - 1).pow(2).sum()
        off_diag = off_diagonal(c).pow(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


class SwAVLoss(torch.nn.Module):
    """Computes the SwAV loss, optionally using a feature queue.

    This loss function contains the core components of the SwAV algorithm, including
    the Sinkhorn-Knopp algorithm for online clustering and the swapped-prediction
    contrastive task.

    Args:
        temperature (float, optional): The temperature scaling factor for the softmax
            in the swapped prediction task. Default is 0.1.
        sinkhorn_iterations (int, optional): The number of iterations for the
            Sinkhorn-Knopp algorithm. Default is 3.
        epsilon (float, optional): A small value for numerical stability in the
            Sinkhorn-Knopp algorithm. Default is 0.05.

    Note:
        Introduced in the SwAV paper :cite:`caron2020unsupervised`.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        epsilon: float = 0.05,
    ):
        super().__init__()
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon

    def forward(self, proj1, proj2, prototypes, queue_feats=None):
        """Compute the SwAV loss.

        Args:
        proj1 (torch.Tensor): Raw projections of the first view.
        proj2 (torch.Tensor): Raw projections of the second view.
        prototypes (torch.nn.Module): The prototype vectors.
        queue_feats (torch.Tensor, optional): Raw features from the queue.
        """
        proj1 = F.normalize(proj1, dim=1, p=2)
        proj2 = F.normalize(proj2, dim=1, p=2)
        if queue_feats is not None:
            queue_feats = F.normalize(queue_feats, dim=1, p=2)

        with torch.no_grad():
            w = prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            prototypes.weight.copy_(w)

        scores1 = prototypes(proj1)
        scores2 = prototypes(proj2)

        with torch.no_grad():
            if queue_feats is not None:
                combined_feats = torch.cat([proj1, proj2, queue_feats])
                combined_scores = prototypes(combined_feats)
                all_q = self.sinkhorn(combined_scores)
                q1 = all_q[: proj1.shape[0]]
                q2 = all_q[proj1.shape[0] : proj1.shape[0] * 2]
            else:
                batch_scores = torch.cat([scores1, scores2])
                all_q = self.sinkhorn(batch_scores)
                q1, q2 = all_q.chunk(2)

        loss = self.swapped_prediction(scores1, q2) + self.swapped_prediction(
            scores2, q1
        )
        return loss / 2.0

    def swapped_prediction(self, scores, q):
        """Computes the cross-entropy loss for the swapped prediction task."""
        scores = scores.float()
        loss = -torch.mean(
            torch.sum(q * F.log_softmax(scores / self.temperature, dim=1), dim=1)
        )
        return loss

    @torch.no_grad()
    def sinkhorn(self, scores):
        """Applies the Sinkhorn-Knopp algorithm."""
        scores = scores.float()
        Q = torch.exp(scores / self.epsilon).T
        Q /= torch.sum(Q)
        K, B = Q.shape
        r = torch.ones(K, device=Q.device) / K
        c = torch.ones(B, device=Q.device) / B
        for _ in range(self.sinkhorn_iterations):
            u = torch.sum(Q, dim=1)
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).T


class InfoNCELoss(torch.nn.Module):
    """InfoNCE contrastive loss (one-directional).

    This module computes the cross-entropy loss between anchor embeddings
    and a set of candidate embeddings, given the ground-truth targets. It
    forms the core mathematical operation for losses like those in CLIP
    and SimCLR.

    Args:
        temperature (float, optional): The temperature scaling factor.
            Default is 0.07.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def _compute(
        self,
        anchors: torch.Tensor,
        candidates: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        logit_scale: Optional[torch.Tensor | float] = None,
    ) -> torch.Tensor:
        logit_scale = self.temperature if logit_scale is None else logit_scale

        # ``_compute`` is a pure local ``anchors @ candidates.T`` cross-entropy.
        # Cross-GPU gathering of ``candidates`` (and the matching rank-offset of
        # ``targets`` / ``mask``) is the caller's responsibility — see
        # ``NTXEntLoss.forward`` / ``CLIPLoss.forward`` — so anchors stay local
        # while candidates may span all ranks.
        anchors = F.normalize(anchors, dim=-1)
        candidates = F.normalize(candidates, dim=-1)

        logits = (anchors @ candidates.T) / logit_scale

        if mask is not None:
            logits = logits.masked_fill(mask, -torch.inf)

        return F.cross_entropy(logits, targets)

    def forward(
        self,
        anchors: torch.Tensor,
        candidates: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        logit_scale: Optional[torch.Tensor | float] = None,
    ) -> torch.Tensor:
        """Computes the contrastive loss.

        Args:
            anchors (torch.Tensor): The primary set of embeddings (queries) of shape `[N, D]`.
            candidates (torch.Tensor): The set of embeddings to contrast against (keys)
                of shape `[M, D]`.
            targets (torch.Tensor): A 1D tensor of ground-truth indices of shape `[N]`,
                where `targets[i]` is the index of the positive candidate for `anchors[i]`.
            mask (torch.Tensor, optional): A boolean mask of shape `[N, M]` to exclude
                certain anchor-candidate pairs from the loss calculation. Values set to
                `True` will be ignored.
            logit_scale (torch.Tensor | float, optional): The temperature scaling factor.
                Default is `self.temperature`.

        Returns:
            torch.Tensor: A scalar loss value.
        """
        return self._compute(anchors, candidates, targets, mask, logit_scale)


class NTXEntLoss(InfoNCELoss):
    """Normalized temperature-scaled cross entropy loss.

    Introduced in the SimCLR paper :cite:`chen2020simple`.
    Also used in MoCo :cite:`he2020momentum`.

    Args:
        temperature (float, optional): The temperature scaling factor.
            Default is 0.5.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__(temperature=temperature)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute the NT-Xent loss.

        Under DDP the negatives span **all** ranks: the local anchors
        (``[z_i; z_j]``) are scored against the candidates gathered from every
        process, with ``targets`` / the self-mask offset by ``rank`` so each
        anchor's positive (its other view) and self-exclusion point at the right
        rows of the global candidate set. In a single process this reduces
        exactly to the classic SimCLR formulation (``rank == 0``,
        ``world_size == 1``, gather is a no-op).

        Args:
            z_i (torch.Tensor): Latent representation of the first augmented view of the batch.
            z_j (torch.Tensor): Latent representation of the second augmented view of the batch.

        Returns:
            float: The computed contrastive loss.
        """
        anchors = torch.cat([z_i, z_j], dim=0)  # local: [2N, D]
        # Gather every rank's anchors as candidates (autograd-aware), so each
        # local anchor is contrasted against the GLOBAL set of negatives.
        candidates = torch.cat(all_gather(anchors), dim=0)  # global: [2N * W, D]

        two_n = anchors.size(0)
        offset = get_rank() * two_n  # where this rank's block sits in `candidates`
        device = z_i.device

        # Positive of local anchor a is its other view; in the global candidate
        # block for this rank that is index ``offset + (a + N) % 2N``.
        local_idx = torch.arange(two_n, device=device)
        n = z_i.size(0)
        targets = offset + (local_idx + n) % two_n

        # Exclude only the anchor's own copy (global index ``offset + a``) — not
        # its positive. ``masked_fill`` with this boolean mask sends those logits
        # to ``-inf``.
        mask = torch.zeros(two_n, candidates.size(0), dtype=torch.bool, device=device)
        mask[local_idx, offset + local_idx] = True

        return self._compute(anchors, candidates, targets, mask=mask)
