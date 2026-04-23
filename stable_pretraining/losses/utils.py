"""Utilities for SSL losses.

This module provides helper functions and utilities used by various SSL losses,
such as Sinkhorn-Knopp optimal transport algorithm for DINO and iBOT.
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from loguru import logger as logging
from ..utils import all_gather


def off_diagonal(x):
    """Return a flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m, logging.error("Input tensor must be square.")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class NegativeCosineSimilarity(torch.nn.Module):
    """Negative cosine similarity objective.

    This objective is used for instance in BYOL :cite:`grill2020bootstrap`
    or SimSiam :cite:`chen2021exploring`.
    """

    def forward(self, z_i, z_j):
        """Compute the loss of the BYOL model.

        Args:
            z_i (torch.Tensor): Latent representation of the first augmented view of the batch.
            z_j (torch.Tensor): Latent representation of the second augmented view of the batch.

        Returns:
            float: The computed loss.
        """
        sim = torch.nn.CosineSimilarity(dim=1)
        return -sim(z_i, z_j).mean()


class VCRegLoss(torch.nn.Module):
    """Variance-Covariance regularization loss..

    Args:
        std_coeff (float, optional): The weight of the standard deviation loss.
            Default is 25.
        cov_coeff (float, optional): The weight of the covariance loss.
            Default is 1.
        epsilon (float, optional): Small value to avoid division by zero.
            Default is 1e-4.
    """

    def __init__(
        self,
        std_coeff: float = 25,
        cov_coeff: float = 1,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.epsilon = epsilon

    def forward(self, z_i):
        """Compute the VCReg loss.

        Args:
            z_i (torch.Tensor): Latent representation of the batch.

        Returns:
            float: The computed loss.
        """
        z_i = torch.cat(all_gather(z_i), 0)

        z_i = z_i - z_i.mean(dim=0)

        std_i = torch.sqrt(z_i.var(dim=0) + self.epsilon)
        std_loss = torch.mean(F.relu(1 - std_i)) / 2

        cov_i = (z_i.T @ z_i) / (z_i.size(0) - 1)
        cov_loss = off_diagonal(cov_i).pow_(2).sum().div(z_i.size(1))

        loss = self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss


@torch.no_grad()
def sinkhorn_knopp(
    teacher_output: torch.Tensor,
    teacher_temp: float,
    num_samples: int | torch.Tensor,
    n_iterations: int = 3,
) -> torch.Tensor:
    """Sinkhorn-Knopp algorithm for optimal transport normalization.

    This is an alternative to simple centering used in DINO/iBOT losses.
    It performs optimal transport to assign samples to prototypes, ensuring
    a more uniform distribution across prototypes.

    Reference: DINOv3 implementation
    https://github.com/facebookresearch/dinov3

    Args:
        teacher_output: Teacher predictions [batch, prototypes] or [n_samples, prototypes]
        teacher_temp: Temperature for softmax
        num_samples: Number of samples to assign. Can be:
            - int: Fixed batch size (e.g., batch_size * world_size for DINO)
            - torch.Tensor: Variable count (e.g., n_masked_patches for iBOT)
        n_iterations: Number of Sinkhorn iterations (default: 3)

    Returns:
        Normalized probabilities [batch, prototypes] summing to 1 over prototypes

    Examples:
        # DINO CLS token loss (fixed batch size)
        Q = sinkhorn_knopp(teacher_cls_output, temp=0.04,
                          num_samples=batch_size * world_size)

        # iBOT patch loss (variable number of masked patches)
        Q = sinkhorn_knopp(teacher_patch_output, temp=0.04,
                          num_samples=n_masked_patches_tensor)
    """
    teacher_output = teacher_output.float()

    # Q is K-by-B for consistency with paper notations
    Q = torch.exp(teacher_output / teacher_temp).t()
    K = Q.shape[0]  # number of prototypes

    # Handle num_samples as tensor or int
    if isinstance(num_samples, torch.Tensor):
        num_samples = num_samples.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_samples)

    # Make the matrix sum to 1
    sum_Q = torch.sum(Q)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sum_Q)
    Q /= sum_Q

    # Sinkhorn iterations
    for _ in range(n_iterations):
        # Normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # Normalize each column: total weight per sample must be 1/num_samples
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= num_samples

    Q *= num_samples  # the columns must sum to 1 so that Q is an assignment
    return Q.t()
