"""Deep InfoMax (DIM) losses.

This module contains the objectives of local Deep InfoMax
:cite:`hjelm2019learningdeeprepresentationsmutual`: a Jensen-Shannon mutual
information estimator between a global feature vector and every location of
a local feature map, and the optional adversarial prior matching term on the
global feature.
"""

import math

import torch
import torch.nn.functional as F
from torch.func import functional_call


class LocalDIMLoss(torch.nn.Module):
    """Jensen-Shannon local DIM objective :cite:`hjelm2019learningdeeprepresentationsmutual`.

    Takes a tensor of critic scores ``scores[a, b, i]`` comparing the global
    feature of image ``a`` with location ``i`` of the local feature map of
    image ``b``. Diagonal pairs (``a == b``) are positives (samples from the
    joint distribution); off-diagonal pairs are negatives (samples from the
    product of marginals). The loss is the negated JSD mutual information
    estimator (Eq. 4 of the paper) averaged over all locations (Eq. 6).

    Note:
        Positives and negatives are averaged separately, as in the reference
        implementation, so the ``B`` positives per location are not drowned
        out by the ``B * (B - 1)`` negatives. The ``log 2`` shifts of the
        reference implementation are kept so that ``-loss`` matches its logged
        JSD estimate and lies in ``(-inf, 2 log 2]``. Negatives are drawn from
        the local batch only; no cross-device gathering is performed.
    """

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute the local DIM loss.

        Args:
            scores: Critic scores of shape ``[B, B, M]`` where ``B`` is the
                batch size and ``M`` the number of local feature locations.

        Returns:
            torch.Tensor: Scalar loss value.

        Raises:
            ValueError: If ``scores`` is not ``[B, B, M]`` or ``B < 2`` (no
                negative pairs).
        """
        if scores.ndim != 3 or scores.size(0) != scores.size(1):
            raise ValueError(
                f"scores must have shape [B, B, M], got {tuple(scores.shape)}"
            )
        batch_size = scores.size(0)
        if batch_size < 2:
            raise ValueError("LocalDIMLoss needs a batch size >= 2 to form negatives")

        pos_mask = torch.eye(batch_size, dtype=torch.bool, device=scores.device)
        log_2 = math.log(2.0)
        e_pos = log_2 - F.softplus(-scores[pos_mask]).mean()
        e_neg = F.softplus(scores[~pos_mask]).mean() - log_2
        return e_neg - e_pos


class DIMPriorLoss(torch.nn.Module):
    """Adversarial prior matching term of DIM :cite:`hjelm2019learningdeeprepresentationsmutual`.

    A discriminator learns to tell samples of a uniform prior on ``[0, 1]^D``
    from ``sigmoid(embedding)``, while the encoder is trained to fool it
    (Eq. 7 of the paper). Both terms are returned so they can be added to a
    single loss and optimized with one backward pass: the discriminator term
    only reaches the discriminator parameters and the encoder term only
    reaches the embedding.

    Args:
        discriminator: Network mapping ``[N, D]`` to ``[N, 1]`` logits, e.g.
            ``spt.backbone.MLP(D, [1000, 200, 1])`` as in the paper (Table 10).
        gradient_penalty: Weight of the gradient penalty on the
            discriminator. ``0`` disables it.

    Note:
        Details not stated in the paper follow the reference implementation
        (https://github.com/rdevon/DIM): the embedding is squashed with a
        sigmoid before being compared to the prior, the discriminator uses the
        standard GAN loss, the encoder uses the non-saturating generator loss,
        and the discriminator is regularized with the gradient penalty of
        Mescheder et al. (2018), averaged over the prior and encoder samples.
        The penalty reproduces the reference ``contrastive_gradient_penalty``
        exactly, including its indexing of the discriminator output with
        ``[-1]``: only the input gradient of the last sample of the batch is
        non-zero, and its squared norm is divided by the batch size.
    """

    def __init__(self, discriminator: torch.nn.Module, gradient_penalty: float = 1.0):
        super().__init__()
        self.discriminator = discriminator
        self.gradient_penalty = gradient_penalty

    def _penalty(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.detach().requires_grad_(True)
        with torch.enable_grad():
            score = self.discriminator(samples)[-1]
            (grad,) = torch.autograd.grad(score.sum(), samples, create_graph=True)
        return grad.pow(2).flatten(1).sum(dim=1).mean()

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the prior matching terms.

        Args:
            embedding: Global features of shape ``[N, D]`` (before sigmoid).

        Returns:
            dict[str, torch.Tensor]: ``"encoder"``, the generator loss whose
            gradient only reaches ``embedding``, and ``"discriminator"``, the
            discriminator loss (including the gradient penalty) whose gradient
            only reaches the discriminator parameters.
        """
        fake = torch.sigmoid(embedding)
        real = torch.rand_like(fake)

        disc_loss = (
            F.softplus(-self.discriminator(real)).mean()
            + F.softplus(self.discriminator(fake.detach())).mean()
        )
        if self.gradient_penalty > 0:
            penalty = 0.5 * (self._penalty(real) + self._penalty(fake))
            disc_loss = disc_loss + self.gradient_penalty * penalty

        # Frozen copy of the discriminator so that the encoder term does not
        # train the discriminator to be fooled.
        frozen = {
            name: tensor.detach()
            for name, tensor in self.discriminator.state_dict(keep_vars=True).items()
        }
        enc_loss = F.softplus(
            -functional_call(self.discriminator, frozen, (fake,))
        ).mean()
        return {"encoder": enc_loss, "discriminator": disc_loss}
