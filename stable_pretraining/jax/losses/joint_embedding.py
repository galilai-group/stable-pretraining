"""Joint-embedding SSL losses for the JAX backend.

JAX/jnp ports of the torch losses in
:mod:`stable_pretraining.losses.joint_embedding`. They are intentionally
numerically faithful to the torch versions so the parity regression tests can
assert bit-close agreement for identical inputs. Losses are plain stateless
callables (not ``nnx.Module``) — they hold only scalar hyper-parameters.

Note:
    These are single-device implementations. The torch versions gather across
    the distributed group (``all_gather``); the multi-host JAX equivalent
    (``jax.lax.all_gather`` over a mesh axis) is a follow-up.
"""

import jax.numpy as jnp
import optax

from .utils import off_diagonal, vcreg_loss

_NORM_EPS = 1e-12  # matches torch.nn.functional.normalize default eps


def l2_normalize(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """L2-normalize along ``axis`` matching ``F.normalize`` semantics.

    Args:
        x: Input array.
        axis: Axis to normalize over.

    Returns:
        jnp.ndarray: ``x / max(||x||_2, eps)``.
    """
    norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=True))
    return x / jnp.maximum(norm, _NORM_EPS)


class NTXEntLoss:
    """Normalized temperature-scaled cross entropy (NT-Xent / InfoNCE) loss.

    JAX port of :class:`stable_pretraining.losses.NTXEntLoss`. Introduced in
    SimCLR :cite:`chen2020simple`.

    Args:
        temperature: Temperature scaling factor. Default ``0.5``.
    """

    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature

    def __call__(self, z_i: jnp.ndarray, z_j: jnp.ndarray) -> jnp.ndarray:
        """Compute the NT-Xent loss for two views.

        Args:
            z_i: Latent representations of the first view, shape ``[N, D]``.
            z_j: Latent representations of the second view, shape ``[N, D]``.

        Returns:
            jnp.ndarray: Scalar contrastive loss.
        """
        anchors = jnp.concatenate([z_i, z_j], axis=0)
        anchors = l2_normalize(anchors, axis=-1)
        candidates = anchors

        n = z_i.shape[0]
        two_n = 2 * n
        # Positive of view-i sample i is view-j sample i and vice-versa.
        targets = jnp.concatenate([jnp.arange(n, two_n), jnp.arange(0, n)], axis=0)

        logits = (anchors @ candidates.T) / self.temperature
        # Mask self-similarity on the diagonal with -inf so softmax ignores it.
        logits = jnp.where(jnp.eye(two_n, dtype=bool), -jnp.inf, logits)

        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()


class InfoNCELoss:
    """InfoNCE contrastive loss (one-directional).

    JAX port of :class:`stable_pretraining.losses.InfoNCELoss` (single-device).
    Cross-entropy between L2-normalized anchors and candidates given integer
    targets — the core operation behind CLIP and SimCLR.

    Args:
        temperature: Temperature scaling factor. Default ``0.07``.
    """

    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature

    def __call__(self, anchors, candidates, targets, mask=None, logit_scale=None):
        """Compute the InfoNCE loss.

        Args:
            anchors: Query embeddings ``[N, D]``.
            candidates: Key embeddings ``[M, D]``.
            targets: Integer indices ``[N]`` of the positive candidate per anchor.
            mask: Optional boolean ``[N, M]``; ``True`` entries are excluded.
            logit_scale: Override for the temperature divisor.

        Returns:
            jnp.ndarray: Scalar loss.
        """
        scale = self.temperature if logit_scale is None else logit_scale
        anchors = l2_normalize(anchors, axis=-1)
        candidates = l2_normalize(candidates, axis=-1)
        logits = (anchors @ candidates.T) / scale
        if mask is not None:
            logits = jnp.where(mask, -jnp.inf, logits)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()


class NegativeCosineSimilarity:
    """Negative cosine similarity objective (BYOL/SimSiam :cite:`chen2021exploring`).

    JAX port of :class:`stable_pretraining.losses.utils.NegativeCosineSimilarity`.
    """

    def __call__(self, z_i: jnp.ndarray, z_j: jnp.ndarray) -> jnp.ndarray:
        """Return ``-mean(cosine_similarity(z_i, z_j))``.

        Args:
            z_i: First embeddings ``[N, D]``.
            z_j: Second embeddings ``[N, D]``.

        Returns:
            jnp.ndarray: Scalar loss in ``[-1, 1]`` (lower is better).
        """
        zi = l2_normalize(z_i, axis=-1)
        zj = l2_normalize(z_j, axis=-1)
        return -jnp.mean(jnp.sum(zi * zj, axis=-1))


class BYOLLoss:
    """Normalized-MSE objective from BYOL :cite:`grill2020bootstrap`.

    JAX port of :class:`stable_pretraining.losses.BYOLLoss`.
    """

    def __call__(
        self, online_pred: jnp.ndarray, target_proj: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the BYOL loss between online predictions and target projections.

        Args:
            online_pred: Predictions from the online predictor ``[N, D]``.
            target_proj: Target-network projections ``[N, D]`` (no gradient).

        Returns:
            jnp.ndarray: Scalar loss.
        """
        online_pred = l2_normalize(online_pred, axis=-1)
        target_proj = l2_normalize(target_proj, axis=-1)
        return jnp.mean(2 - 2 * jnp.sum(online_pred * target_proj, axis=-1))


class VICRegLoss:
    """Variance-Invariance-Covariance objective :cite:`bardes2021vicreg`.

    JAX port of :class:`stable_pretraining.losses.VICRegLoss`.

    Args:
        sim_coeff: Weight of the invariance (MSE) term. Default ``25``.
        std_coeff: Weight of the variance term. Default ``25``.
        cov_coeff: Weight of the covariance term. Default ``1``.
        epsilon: Stability epsilon. Default ``1e-4``.
    """

    def __init__(
        self,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        epsilon: float = 1e-4,
    ):
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.epsilon = epsilon

    def __call__(self, z_i: jnp.ndarray, z_j: jnp.ndarray) -> jnp.ndarray:
        """Compute the VICReg loss for two views.

        Args:
            z_i: First view embeddings ``[N, D]``.
            z_j: Second view embeddings ``[N, D]``.

        Returns:
            jnp.ndarray: Scalar loss.
        """
        repr_loss = jnp.mean((z_i - z_j) ** 2)
        reg = vcreg_loss(
            z_i, self.std_coeff, self.cov_coeff, self.epsilon
        ) + vcreg_loss(z_j, self.std_coeff, self.cov_coeff, self.epsilon)
        return self.sim_coeff * repr_loss + reg


class BarlowTwinsLoss:
    """Barlow Twins cross-correlation objective :cite:`zbontar2021barlow`.

    JAX port of :class:`stable_pretraining.losses.BarlowTwinsLoss`.

    Args:
        lambd: Weight of the off-diagonal redundancy term. Default ``5e-3``.
    """

    def __init__(self, lambd: float = 5e-3):
        self.lambd = lambd

    @staticmethod
    def _standardize(z: jnp.ndarray) -> jnp.ndarray:
        # ddof=1 matches torch.Tensor.std default (unbiased).
        return (z - jnp.mean(z, axis=0)) / (jnp.std(z, axis=0, ddof=1) + 1e-5)

    def __call__(self, z_i: jnp.ndarray, z_j: jnp.ndarray) -> jnp.ndarray:
        """Compute the Barlow Twins loss for two views.

        Args:
            z_i: First view embeddings ``[N, D]``.
            z_j: Second view embeddings ``[N, D]``.

        Returns:
            jnp.ndarray: Scalar loss.
        """
        c = (self._standardize(z_i).T @ self._standardize(z_j)) / z_i.shape[0]
        on_diag = jnp.sum((jnp.diagonal(c) - 1) ** 2)
        off_diag = jnp.sum(off_diagonal(c) ** 2)
        return on_diag + self.lambd * off_diag


__all__ = [
    "NTXEntLoss",
    "InfoNCELoss",
    "NegativeCosineSimilarity",
    "BYOLLoss",
    "VICRegLoss",
    "BarlowTwinsLoss",
    "l2_normalize",
]
