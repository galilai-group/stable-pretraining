"""Gate and regularization primitives for BayesNCL.

Reference: https://arxiv.org/abs/2605.28441
"""

import math

import torch


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
