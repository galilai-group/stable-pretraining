"""Loss terms for Clustering-based Mutual Targeting (CluMuT)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CluMuTLoss(nn.Module):
    """Compute the mutual-targeting and identitization losses.

    Args:
        lambd: Weight applied to off-diagonal correlations before measuring
            their squared distance from the identity matrix.
        eps: Minimum denominator used when computing correlations.

    Note:
        The mutual-targeting KL terms are summed over batch elements and
        prototypes to match the paper's displayed batch objective. Compared
        with ``batchmean``, this scales the mutual-targeting term linearly
        with batch size; the identitization term is unchanged.
        Correlations require at least two examples per view. Constant cluster
        dimensions are assigned zero correlation, keeping the loss finite.
    """

    def __init__(self, lambd: float = 0.005, eps: float = 1e-6):
        super().__init__()
        if lambd < 0:
            raise ValueError("lambd must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.lambd = lambd
        self.eps = eps

    def mutual_targeting_loss(
        self, q_a: torch.Tensor, q_b: torch.Tensor
    ) -> torch.Tensor:
        """Compute symmetric cross-view KL with detached targets."""
        log_q_a = q_a.clamp_min(torch.finfo(q_a.dtype).tiny).log()
        log_q_b = q_b.clamp_min(torch.finfo(q_b.dtype).tiny).log()
        return 0.5 * (
            F.kl_div(log_q_a, q_b.detach(), reduction="sum")
            + F.kl_div(log_q_b, q_a.detach(), reduction="sum")
        )

    def correlation(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        """Return the batch-wise Pearson cross-correlation matrix."""
        if h_a.ndim != 2 or h_b.ndim != 2 or h_a.shape != h_b.shape:
            raise ValueError("h_a and h_b must have the same [batch, clusters] shape")
        if h_a.shape[0] < 2:
            raise ValueError(
                "CluMuT identitization requires a batch size of at least 2"
            )

        a = h_a.float() - h_a.float().mean(dim=0, keepdim=True)
        b = h_b.float() - h_b.float().mean(dim=0, keepdim=True)
        numerator = a.T @ b
        a_norm = (a.square().sum(dim=0) + self.eps**2).sqrt()
        b_norm = (b.square().sum(dim=0) + self.eps**2).sqrt()
        denominator = (a_norm[:, None] * b_norm[None, :]).clamp_min(self.eps)
        return numerator / denominator

    def identitization_loss(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        correlation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Match scaled cluster cross-correlations to an identity matrix."""
        corr = self.correlation(h_a, h_b) if correlation is None else correlation
        identity = torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)
        scale = torch.full_like(corr, self.lambd)
        scale.fill_diagonal_(1.0)
        return (corr * scale - identity).square().sum()
