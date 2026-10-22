"""BayesNCL loss and gate primitives.

Reference: https://arxiv.org/abs/2605.28441
"""

import math

import torch

from .joint_embedding import NTXEntLoss


def straight_through_gate(
    logits: torch.Tensor,
    temperature: float = 1.0,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute binary gates with sigmoid gradients.

    Args:
        logits: Gating head outputs, shaped [batch, features].
        temperature: Finite positive sigmoid temperature.
        threshold: Probability threshold in [0, 1]. Values strictly above it
            produce an open gate.

    Returns:
        Binary gates and sigmoid probabilities, both shaped like logits.
        The gates use sigmoid derivatives during backpropagation. The
        probabilities are used separately for KL regularization.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")

    probabilities = torch.sigmoid(logits / temperature)
    hard_gates = (probabilities > threshold).to(probabilities.dtype)
    gates = hard_gates + (probabilities - probabilities.detach())
    return gates, probabilities


def bernoulli_kl(probabilities: torch.Tensor, prior_probability: float) -> torch.Tensor:
    """Compute the BayesNCL Bernoulli KL penalty.

    Args:
        probabilities: Posterior gate probabilities, shaped [batch, features].
        prior_probability: Prior probability of an open gate, in [0, 1].

    Returns:
        Scalar KL(posterior || prior), summed over features and averaged over
        the batch. Both distributions' probabilities are clamped to
        [0.01, 0.99], following the reference implementation.
    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [batch, features]")
    if not 0 <= prior_probability <= 1:
        raise ValueError("prior_probability must be in [0, 1]")

    posterior = probabilities.clamp(0.01, 0.99)
    prior = posterior.new_tensor(prior_probability).clamp(0.01, 0.99)
    kl = posterior * (posterior.log() - prior.log())
    kl = kl + (1 - posterior) * ((1 - posterior).log() - (1 - prior).log())
    return kl.sum(dim=1).mean()


class BayesNCLLoss(torch.nn.Module):
    """Contrastive loss with Bernoulli KL regularization for BayesNCL.

    Args:
        temperature: Finite positive temperature for the contrastive loss.
        kl_weight: Finite nonnegative weight for the KL penalty.
        prior_probability: Prior probability of an open gate, in [0, 1].

    Note:
        Inputs are already gated representations. Gate probabilities remain
        differentiable and are regularized before binary thresholding. The
        contrastive term uses one positive per anchor and global negatives
        under distributed training, as in NTXEntLoss.
    """

    def __init__(
        self,
        temperature: float = 0.2,
        kl_weight: float = 3e-5,
        prior_probability: float = 0.8,
    ):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(kl_weight) or kl_weight < 0:
            raise ValueError("kl_weight must be finite and nonnegative")
        if not 0 <= prior_probability <= 1:
            raise ValueError("prior_probability must be in [0, 1]")

        self.contrastive_loss = NTXEntLoss(temperature=temperature)
        self.kl_weight = kl_weight
        self.prior_probability = prior_probability

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        probabilities_i: torch.Tensor,
        probabilities_j: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the two-view BayesNCL objective.

        Args:
            z_i: Gated representations of the first view, shaped [batch, features].
            z_j: Gated representations of the second view, shaped [batch, features].
            probabilities_i: First-view gate probabilities, shaped [batch, features].
            probabilities_j: Second-view gate probabilities, shaped [batch, features].

        Returns:
            Scalar contrastive loss plus the weighted KL penalty, averaged
            over both views. NTXEntLoss normalizes the gated representations.
        """
        contrastive = self.contrastive_loss(z_i, z_j)
        probabilities = torch.cat([probabilities_i, probabilities_j], dim=0)
        regularization = bernoulli_kl(probabilities, self.prior_probability)
        return contrastive + self.kl_weight * regularization
