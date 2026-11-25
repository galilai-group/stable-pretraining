"""Sub-JEPA objective (Zhao et al., 2026, https://arxiv.org/abs/2605.09241)."""

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn.functional import all_reduce
from torch.nn import functional as F


class SubJEPALoss(nn.Module):
    """Prediction MSE plus frozen-subspace Gaussian regularization.

    Args:
        embed_dim: Latent dimension, 192 in the paper.
        num_subspaces: Number of independent frozen subspaces (32, or 16 for PushT).
        subspace_dim: Subspace dimension; defaults to ``embed_dim / num_subspaces``.
        regularization_weight: Weight of the Gaussian statistic.
        num_projections: Random unit directions per subspace and forward call.
        num_knots: Trapezoidal quadrature nodes over [0, 3].
        projection_seed: Seed for resampled directions, synchronized across DDP ranks.

    Note:
        Matches the released orthogonal_frozen implementation: characteristic
        functions are estimated across the batch separately at each time step,
        then averaged over time, directions, and subspaces. The paper's equation
        6 instead describes pooling time and batch. Targets retain gradients.
        DDP uses global batch statistics and supports unequal local batch sizes.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_subspaces: int = 32,
        subspace_dim: int | None = None,
        regularization_weight: float = 0.1,
        num_projections: int = 1024,
        num_knots: int = 17,
        projection_seed: int = 0,
    ) -> None:
        super().__init__()
        if embed_dim < 1 or num_subspaces < 1:
            raise ValueError("embed_dim and num_subspaces must be positive")
        if subspace_dim is None:
            if embed_dim % num_subspaces:
                raise ValueError("embed_dim must be divisible by num_subspaces")
            subspace_dim = embed_dim // num_subspaces
        if not 1 <= subspace_dim <= embed_dim:
            raise ValueError("subspace_dim must be between 1 and embed_dim")
        if num_projections < 1 or num_knots < 2:
            raise ValueError("num_projections must be positive and num_knots >= 2")
        if not math.isfinite(regularization_weight) or regularization_weight < 0:
            raise ValueError("regularization_weight must be finite and nonnegative")
        self.embed_dim = embed_dim
        self.num_projections = num_projections
        self.regularization_weight = regularization_weight
        self.projection_seed = projection_seed
        bases = torch.linalg.qr(
            torch.randn(num_subspaces, embed_dim, subspace_dim), mode="reduced"
        ).Q.transpose(-1, -2)
        self.register_buffer("projection_matrices", bases.contiguous())
        knots = torch.linspace(0, 3, num_knots)
        phi = torch.exp(-knots.square() / 2)
        weights = torch.full_like(knots, 2 * 3 / (num_knots - 1))
        weights[[0, -1]] /= 2
        self.register_buffer("knots", knots)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)
        self.register_buffer("projection_step", torch.zeros((), dtype=torch.long))

    def regularization(self, embedding: torch.Tensor) -> torch.Tensor:
        """Compute the Epps–Pulley statistic in frozen orthogonal subspaces.

        Args:
            embedding: Encoded trajectories of shape [batch, time, embed_dim].

        Returns:
            Scalar Gaussian regularization, scaled by the global batch size.
        """
        if embedding.ndim != 3 or embedding.shape[-1] != self.embed_dim:
            raise ValueError("embedding must have shape [batch, time, embed_dim]")
        if embedding.shape[0] < 1 or embedding.shape[1] < 1:
            raise ValueError("embedding batch and time dimensions must be nonempty")
        # Float32 avoids unstable trigonometric statistics under mixed precision.
        dtype = torch.float64 if embedding.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=embedding.device.type, enabled=False):
            embedding = embedding.to(dtype)
            bases = self.projection_matrices.to(dtype)
            generator = torch.Generator(device=embedding.device).manual_seed(
                self.projection_seed + int(self.projection_step)
            )
            directions = torch.randn(
                bases.shape[0],
                bases.shape[1],
                self.num_projections,
                device=embedding.device,
                dtype=dtype,
                generator=generator,
            )
            directions = F.normalize(directions, dim=1)
            self.projection_step.add_(1)
            projected = torch.einsum("btd,ksd,ksm->ktbm", embedding, bases, directions)
            angles = projected.unsqueeze(-1) * self.knots.to(dtype)
            cos_sum = angles.cos().sum(dim=2)
            sin_sum = angles.sin().sum(dim=2)
            count = embedding.new_tensor(embedding.shape[0])
            if dist.is_available() and dist.is_initialized():
                cos_sum = all_reduce(cos_sum)
                sin_sum = all_reduce(sin_sum)
                dist.all_reduce(count)
            error = (cos_sum / count - self.phi.to(dtype)).square()
            error = error + (sin_sum / count).square()
            return (error @ self.weights.to(dtype)).mean() * count

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the total objective and its components without detaching targets.

        Args:
            prediction: Predicted next latents [batch, time - 1, embed_dim].
            target: Encoded next observations with the same shape.
            embedding: All encoded observations [batch, time, embed_dim].

        Returns:
            Dict containing loss, prediction_loss, and regularization_loss.
        """
        if prediction.shape != target.shape or prediction.numel() == 0:
            raise ValueError(
                "prediction and target must have identical nonempty shapes"
            )
        prediction_loss = F.mse_loss(prediction.float(), target.float())
        regularization_loss = self.regularization(embedding)
        return {
            "loss": prediction_loss + self.regularization_weight * regularization_loss,
            "prediction_loss": prediction_loss,
            "regularization_loss": regularization_loss,
        }
