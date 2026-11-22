"""Joint embedding SSL losses.

This module contains joint embedding methods that learn to embed different views
of the same image close together in representation space. Includes both contrastive
(NTXentLoss) and non-contrastive (BYOL, VICReg, Barlow Twins) methods.
"""

import math
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


class CorInfoMaxLoss(torch.nn.Module):
    """Agreement and running-covariance log-determinant objective.

    Args:
        projection_dim: Width of both projected views.
        sim_coeff: Weight of the mean squared agreement loss.
        cov_coeff: Weight of the covariance log-determinant loss.
        mean_momentum: Weight of the previous running mean, in ``[0, 1)``.
        covariance_momentum: Weight of the previous covariance, in ``[0, 1)``.
        initial_covariance: Positive initial covariance diagonal.
        epsilon: Positive diagonal regularizer used only inside the determinant.
        normalize: L2-normalize projections before both objective terms.
        statistics_dtype: Float64 (reference default) or float32 running state.

    Note:
        Current statistics remain differentiable; only retained history is
        detached. Evaluation computes candidate statistics without storing them.
        DDP gathers global projections and requires equal local batch sizes and
        synchronized calls. History is checkpointed as persistent buffers.
        Explicit ``.float()`` changes the statistics precision; ``.half()`` and
        ``.bfloat16()`` history are unsupported. Use mixed precision instead.
    """

    def __init__(
        self,
        projection_dim: int,
        sim_coeff: float = 250.0,
        cov_coeff: float = 1.0,
        mean_momentum: float = 0.01,
        covariance_momentum: float = 0.01,
        initial_covariance: float = 1.0,
        epsilon: float = 1e-8,
        normalize: bool = True,
        statistics_dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if not isinstance(projection_dim, int) or projection_dim < 1:
            raise ValueError("projection_dim must be a positive integer")
        for name, value in (("sim_coeff", sim_coeff), ("cov_coeff", cov_coeff)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            ("mean_momentum", mean_momentum),
            ("covariance_momentum", covariance_momentum),
        ):
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        for name, value in (
            ("initial_covariance", initial_covariance),
            ("epsilon", epsilon),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if statistics_dtype not in (torch.float32, torch.float64):
            raise ValueError("statistics_dtype must be torch.float32 or torch.float64")
        self.projection_dim = projection_dim
        self.sim_coeff = sim_coeff
        self.cov_coeff = cov_coeff
        self.mean_momentum = mean_momentum
        self.covariance_momentum = covariance_momentum
        self.epsilon = epsilon
        self.normalize = normalize
        for view in (1, 2):
            self.register_buffer(
                f"mean_{view}", torch.zeros(projection_dim, dtype=statistics_dtype)
            )
            self.register_buffer(
                f"covariance_{view}",
                initial_covariance * torch.eye(projection_dim, dtype=statistics_dtype),
            )

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute the objective and retain detached history during training.

        Args:
            z_i: Raw first-view projections, shape ``[N, projection_dim]``.
            z_j: Paired second-view projections, with matching shape and dtype.

        Returns:
            Scalar weighted CorInfoMax loss.

        Raises:
            ValueError: Inputs or running-state precision are incompatible.
        """
        if (
            z_i.ndim != 2
            or z_j.shape != z_i.shape
            or z_i.shape[0] == 0
            or z_i.shape[1] != self.projection_dim
        ):
            raise ValueError("Expected matching nonempty [N, projection_dim] views")
        if (
            not z_i.is_floating_point()
            or z_i.dtype != z_j.dtype
            or z_i.device != z_j.device
            or z_i.device != self.mean_1.device
        ):
            raise ValueError("Views must have matching floating dtype and loss device")
        dtype = self.mean_1.dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("CorInfoMax history requires float32 or float64; use AMP")
        if get_world_size() > 1:
            shape = torch.tensor(z_i.shape, device=z_i.device)
            if any(not torch.equal(shape, other) for other in all_gather(shape)):
                raise ValueError("CorInfoMax DDP requires equal local view shapes")
            z_i = torch.cat(all_gather(z_i), dim=0)
            z_j = torch.cat(all_gather(z_j), dim=0)

        # Autocast must not lower the precision of covariance matrix products.
        with torch.autocast(device_type=z_i.device.type, enabled=False):
            if z_i.dtype in (torch.float16, torch.bfloat16):
                z_i, z_j = z_i.float(), z_j.float()
            if self.normalize:
                z_i, z_j = F.normalize(z_i, dim=-1), F.normalize(z_j, dim=-1)
            similarity = F.mse_loss(z_i, z_j)
            identity = torch.eye(self.projection_dim, device=z_i.device, dtype=dtype)
            updates = []
            determinants = []
            for view, z in enumerate((z_i, z_j), start=1):
                old_mean = getattr(self, f"mean_{view}")
                old_covariance = getattr(self, f"covariance_{view}")
                mean = self.mean_momentum * old_mean + (
                    1 - self.mean_momentum
                ) * z.mean(dim=0).to(dtype)
                centered = z.to(dtype) - mean
                covariance = self.covariance_momentum * old_covariance + (
                    1 - self.covariance_momentum
                ) * (centered.T @ centered / z.shape[0])
                determinants.append(torch.logdet(covariance + self.epsilon * identity))
                updates.append((mean, covariance))
            covariance_loss = -sum(determinants) / self.projection_dim
            loss = self.sim_coeff * similarity + self.cov_coeff * covariance_loss

        if self.training:
            for view, (mean, covariance) in enumerate(updates, start=1):
                # Replacing buffers preserves tensors saved by the current graph.
                setattr(self, f"mean_{view}", mean.detach().clone())
                setattr(self, f"covariance_{view}", covariance.detach().clone())
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
