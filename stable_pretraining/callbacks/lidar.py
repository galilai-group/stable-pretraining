"""LiDAR (Linear Discriminant Analysis Rank) callback for monitoring representation quality.

Based on:
    Thilak et al. "LiDAR: Sensing Linear Probing Performance in Joint Embedding SSL Architectures"
    arXiv:2312.04000 (2023)
"""

from typing import Iterable, Optional, Union

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from .registry import log as _spt_log

from .queue import find_or_create_queue_callback
from .utils import log_header


class LiDAR(Callback):
    """LiDAR (Linear Discriminant Analysis Rank) monitor using queue discovery.

    LiDAR measures the effective rank of learned representations using Linear Discriminant
    Analysis (LDA). It computes the exponential of the entropy of the eigenvalue distribution
    from the LDA transformation, providing a metric between 1 and min(d, n_classes - 1) where
    d is the feature dimension, indicating how many dimensions are effectively being used.

    This implementation is based on Thilak et al. "LiDAR: Sensing Linear Probing Performance
    in Joint Embedding SSL Architectures" (arXiv:2312.04000).

    IMPORTANT: Surrogate Class Formation Requirement
    -------------------------------------------------
    The LiDAR paper requires that each "surrogate class" consists of q augmented views
    of the same clean sample. The current implementation chunks the queue sequentially
    into groups of size samples_per_class. For faithful reproduction of the paper:

    - Ensure the upstream queue pushes q contiguous augmentations of each clean sample
    - OR implement ID-based grouping to ensure each group contains views of the same sample

    Without proper grouping, the metric may not accurately reflect the paper's methodology.

    The metric helps detect:
    - Dimensional collapse in self-supervised learning
    - Loss of representational capacity
    - Over-regularization effects

    Args:
        name: Unique identifier for this callback instance
        target: Key in batch dict containing the feature embeddings to monitor
        queue_length: Size of the circular buffer for caching embeddings
        target_shape: Shape of the target embeddings (e.g., 768 for 768-dim features)
        n_classes: Number of surrogate classes (clean samples) for LDA computation
        samples_per_class: Number of augmented samples per class
        delta: Regularization constant added to within-class covariance (default: 1e-4)
        epsilon: Small constant for numerical stability (default: 1e-8)
    """

    def __init__(
        self,
        name: str,
        target: str,
        queue_length: int,
        target_shape: Union[int, Iterable[int]],
        n_classes: int = 100,
        samples_per_class: int = 10,
        delta: float = 1e-4,
        epsilon: float = 1e-8,
        verbose: bool = None,
    ) -> None:
        super().__init__()

        # Convert target_shape to int if needed
        if isinstance(target_shape, (list, tuple)):
            if len(target_shape) == 1:
                target_shape = target_shape[0]
            else:
                target_shape = int(torch.prod(torch.tensor(target_shape)))

        self.name = name
        self.target = target
        self.queue_length = queue_length
        self.target_shape = target_shape
        self.n_classes = n_classes
        self.samples_per_class = samples_per_class
        self.delta = delta
        self.epsilon = epsilon
        from .utils import resolve_verbose

        self.verbose = resolve_verbose(verbose)

        self._target_queue = None

        # Validate queue length adequacy
        min_required_samples = n_classes * samples_per_class
        if queue_length < min_required_samples:
            logging.warning(
                f"! queue length ({queue_length}) is less than required "
                f"samples ({min_required_samples} = {n_classes} classes x {samples_per_class} samples/class). "
                f"LiDAR computation may use fewer classes than specified."
            )

        log_header("LiDAR")
        logging.info(f"  name: {name}")
        logging.info(f"  target: {target}")
        logging.info(f"  queue_length: {queue_length}")
        logging.info(f"  feature_dimension: {target_shape}")
        logging.info(
            f"  n_classes: {n_classes}, samples_per_class: {samples_per_class}"
        )

    @property
    def state_key(self) -> str:
        """Unique identifier for this callback's state during checkpointing."""
        return f"LiDAR[name={self.name}]"

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Find or create the queue callback for target features."""
        if self._target_queue is None:
            self._target_queue = find_or_create_queue_callback(
                trainer,
                self.target,
                self.queue_length,
                self.target_shape,
                torch.float32,
                gather_distributed=True,
                create_if_missing=True,
            )
            logging.info(f"  target queue: {self.target}")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute LiDAR metric on the first validation batch only."""
        if batch_idx > 0:
            return

        logging.info("  computing LiDAR on first validation batch")

        embeddings = self._target_queue.data

        if embeddings is None:
            logging.warning(f"! {self.name}: queue data not available")
            return

        if embeddings.numel() == 0:
            logging.warning(f"! {self.name}: queue data is empty")
            return

        # The queue already handles gathering across GPUs if gather_distributed=True
        # So embeddings here already contains data from all GPUs
        lidar_value = self._compute_lidar(embeddings)

        if lidar_value is not None:
            pl_module.log(
                self.name,
                lidar_value["lidar"],
                rank_zero_only=True,  # Only log from rank 0 to avoid duplicates
                sync_dist=False,  # No need to sync since we compute same value on all ranks
            )
            if self.verbose:
                _spt_log(
                    f"{self.name}/entropy",
                    lidar_value["entropy"],
                    rank_zero_only=True,
                    sync_dist=False,
                )
                _spt_log(
                    f"{self.name}/top_eigenvalue",
                    lidar_value["top_eigenvalue"],
                    rank_zero_only=True,
                    sync_dist=False,
                )
            if trainer.global_rank == 0:
                logging.info(f"  LiDAR = {lidar_value['lidar']:.4f}")

    def _compute_lidar(self, embeddings: torch.Tensor) -> Optional[dict]:
        """Compute the LiDAR metric from embeddings.

        Args:
            embeddings: Tensor of shape (n_samples, feature_dim)

        Returns:
            Dict with 'lidar', 'entropy', and 'top_eigenvalue' keys, or None if computation fails.
        """
        n_samples, d = embeddings.shape

        # Determine how many classes we can form
        actual_n_classes = min(self.n_classes, n_samples // self.samples_per_class)

        if actual_n_classes < 2:
            logging.warning(
                f"! {self.name}: not enough samples for LiDAR computation. "
                f"Need at least {2 * self.samples_per_class} samples, got {n_samples}"
            )
            return None

        # Reshape embeddings to (n_classes, samples_per_class, feature_dim)
        # WARNING: This assumes the queue contains contiguous groups of augmentations
        # from the same clean sample. If the queue mixes samples randomly, the
        # surrogate classes won't match the paper's methodology.
        # Take only the samples we need
        n_used = actual_n_classes * self.samples_per_class
        embeddings = embeddings[:n_used].view(
            actual_n_classes, self.samples_per_class, d
        )

        with torch.no_grad():
            class_means = embeddings.mean(dim=1)  # (n_classes, d)
            grand_mean = class_means.mean(dim=0)  # (d,)

            device = embeddings.device

            # Sb = sum((mu_i - mu) @ (mu_i - mu)^T) / (n_classes - 1)
            centered_means = class_means - grand_mean.unsqueeze(0)
            Sb = (centered_means.T @ centered_means) / (actual_n_classes - 1)

            # First center all samples by their class means
            class_means_expanded = class_means.unsqueeze(1).expand_as(embeddings)
            centered_samples = embeddings - class_means_expanded

            centered_samples_flat = centered_samples.reshape(-1, d)

            # Sw = sum((x_ij - mu_i) @ (x_ij - mu_i)^T) / (n_classes * (samples_per_class - 1))
            # This is the unbiased estimate of class-averaged within-class covariance
            # as described in the LiDAR paper (arXiv:2312.04000)
            Sw = (centered_samples_flat.T @ centered_samples_flat) / (
                actual_n_classes * (self.samples_per_class - 1)
            )

            # Add regularization to within-class covariance
            Sw = Sw + self.delta * torch.eye(d, device=device)

            # Compute Sw^(-1/2) using eigendecomposition
            eigvals_w, eigvecs_w = torch.linalg.eigh(Sw)
            eigvals_w = torch.clamp(eigvals_w, min=self.epsilon)

            # Sw^(-1/2) = V * D^(-1/2) * V^T
            sqrt_inv_eigvals = 1.0 / torch.sqrt(eigvals_w)
            Sw_invsqrt = (eigvecs_w * sqrt_inv_eigvals.unsqueeze(0)) @ eigvecs_w.T

            # Compute LiDAR matrix: Σ_lidar = Sw^(-1/2) * Sb * Sw^(-1/2)
            Sigma_lidar = Sw_invsqrt @ Sb @ Sw_invsqrt

            # Handle numerical errors by ensuring symmetry
            Sigma_lidar = 0.5 * (Sigma_lidar + Sigma_lidar.T)

            # Compute eigenvalues of LiDAR matrix
            eigvals_lidar = torch.linalg.eigvalsh(Sigma_lidar)
            eigvals_lidar = torch.clamp(eigvals_lidar, min=0.0)

            # Normalize eigenvalues to get probability distribution
            # Following the paper: p_i = (lambda_i + epsilon) / sum_j(lambda_j + epsilon)
            eigvals_with_eps = eigvals_lidar + self.epsilon
            eigvals_sum = eigvals_with_eps.sum()
            if eigvals_sum <= 0:
                logging.warning(f"! {self.name}: all eigenvalues are zero or negative")
                return {"lidar": 1.0, "entropy": 0.0, "top_eigenvalue": 0.0}

            p = eigvals_with_eps / eigvals_sum

            # Compute entropy and LiDAR metric
            entropy = -(p * torch.log(p)).sum()
            lidar = torch.exp(entropy).item()

            return {
                "lidar": lidar,
                "entropy": entropy.item(),
                "top_eigenvalue": eigvals_lidar.max().item(),
            }
