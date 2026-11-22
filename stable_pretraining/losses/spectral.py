"""Spectral contrastive loss from HaoChen et al. (NeurIPS 2021)."""

from typing import Optional

import torch
import torch.nn as nn

from ..utils import all_gather, get_rank


def _project_to_ball(z: torch.Tensor, radius: float) -> torch.Tensor:
    """Rescale rows whose norm exceeds ``radius`` onto the sphere of that radius.

    Rows already inside the ball are left untouched, which is what makes this a
    projection onto the ball rather than a normalisation onto its boundary.
    """
    # Clamp before division: a reciprocal of a tiny norm can overflow in
    # backward even when the resulting scale is later capped at one.
    norm = z.norm(dim=-1, keepdim=True).clamp_min(radius)
    return z * (radius / norm)


class SpectralContrastiveLoss(nn.Module):
    """Spectral contrastive loss over two views of a batch.

    For a batch of ``N`` positive pairs ``(z_i, z'_i)`` the loss is::

        L = -2/N * sum_i z_i . z'_i  +  1/(N(N-1)) * sum_{i != j} (z_i . z'_j)^2

    Unlike InfoNCE it has no temperature and no log-sum-exp: the second term is
    the mean of squared inner products of independent pairs.

    Following the authors' implementation, embeddings are first projected onto
    the ball of radius ``sqrt(mu)`` (rows that already lie inside it are kept
    as they are) and the loss is divided by ``mu``. This follows the authors'
    normalization; rescaling both views changes the positive term with the
    second power and the negative term with the fourth power of the scale.

    Under distributed training the ``z'`` candidates are gathered from every
    rank, so each local anchor is contrasted against the global batch. With a
    single process this is exactly the estimator in Algorithm 1 of the paper.

    Args:
        mu: Squared radius of the ball the embeddings are projected into, and the
            divisor of the loss. ``None`` disables both and uses the embeddings
            and the loss exactly as given.

    Note:
        Reference: HaoChen et al., NeurIPS 2021, https://arxiv.org/abs/2106.04156.
        The ball projection and the division by ``mu`` follow the authors' code,
        https://github.com/jhaochenz96/spectral_contrastive_learning.
    """

    def __init__(self, mu: Optional[float] = 1.0) -> None:
        super().__init__()
        if mu is not None and mu <= 0:
            raise ValueError(f"mu must be positive or None, got {mu}")
        self.mu = mu

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute the spectral contrastive loss.

        Args:
            z_i: Embeddings of the first view, shape ``[N, D]``.
            z_j: Embeddings of the second view, shape ``[N, D]``. Row ``k`` of
                ``z_j`` is the positive of row ``k`` of ``z_i``.

        Returns:
            Scalar loss in float32.

        Raises:
            ValueError: If the views differ in shape, are not 2-D, or the global
                batch holds fewer than two samples (no negative pairs exist).
        """
        if z_i.ndim != 2 or z_i.shape != z_j.shape:
            raise ValueError(
                "z_i and z_j must be 2-D tensors of the same shape, got "
                f"{tuple(z_i.shape)} and {tuple(z_j.shape)}"
            )

        # Squared inner products can overflow in fp16, so this computation must
        # not run under an enclosing autocast region.
        with torch.autocast(device_type=z_i.device.type, enabled=False):
            z_i = z_i.float()
            z_j = z_j.float()
            if self.mu is not None:
                radius = self.mu**0.5
                z_i = _project_to_ball(z_i, radius)
                z_j = _project_to_ball(z_j, radius)

            candidates = torch.cat(all_gather(z_j), dim=0)
            n, m = z_i.size(0), candidates.size(0)
            if m < 2:
                raise ValueError("SpectralContrastiveLoss needs at least 2 samples")

            positive = (z_i * z_j).sum(dim=-1).mean()

            # Row a of the local block sits at column ``offset + a`` of the
            # gathered candidates; that entry is its positive, not a negative.
            offset = get_rank() * n
            own = torch.arange(n, device=z_i.device)
            is_positive = torch.zeros(n, m, dtype=torch.bool, device=z_i.device)
            is_positive[own, offset + own] = True

            squared = (z_i @ candidates.T).pow(2).masked_fill(is_positive, 0.0)
            negative = squared.sum() / (n * (m - 1))

            loss = -2.0 * positive + negative
            if self.mu is not None:
                loss = loss / self.mu
        return loss
