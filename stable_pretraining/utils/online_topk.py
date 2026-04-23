import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple


class StreamingTopKEigen(nn.Module):
    """Hyperparameter-free streaming estimator for top-K eigenvectors.

    This module maintains running estimates of the top-K eigenvectors and
    eigenvalues of the covariance matrix of streaming data. It requires no
    tuning - learning rates and update schedules are derived automatically
    from theoretical considerations.

    **DDP Support**: When used with DistributedDataParallel, this module
    automatically aggregates statistics across all processes using efficient
    all-reduce operations. No data gathering to a single GPU is performed.

    Key Features:
    -------------
    - **No hyperparameters**: Learning rates adapt based on sample count and
      current eigenvalue estimates.
    - **Memory efficient**: O(dk) storage, no need to store covariance matrix.
    - **Numerically stable**: QR orthogonalization + Welford's algorithm.
    - **Fast**: Single-pass update per batch, optimized for GPU.
    - **DDP-aware**: Automatically synchronizes across processes when distributed.

    Mathematical Background:
    ------------------------
    Given streaming data x₁, x₂, ..., xₙ ∈ ℝᵈ, we want to estimate the top-K
    eigenvectors of the covariance matrix:

        C = E[(x - μ)(x - μ)ᵀ]

    The algorithm maintains:
    - Running mean μ̂ (Welford's algorithm)
    - Eigenvector matrix V ∈ ℝᵈˣᵏ (columns are eigenvectors)
    - Eigenvalue estimates λ̂ ∈ ℝᵏ

    Updates use Sanger's rule with adaptive learning rates:

        V ← V + diag(η) · (E[x̃ yᵀ] - V · tril(E[y yᵀ]))

    where:
    - x̃ = x - μ̂ (centered data)
    - y = Vᵀx̃ (projections)
    - η_i = (1/√n) · (σ²_total / λ̂_i) (adaptive per-component learning rate)
    - tril(·) extracts lower triangular part (for deflation)

    Example Usage:
    --------------
    >>> # Initialize estimator
    >>> estimator = StreamingTopKEigen(dim=512, k=16, device="cuda")
    >>>
    >>> # Training loop - just call forward on each batch
    >>> for epoch in range(num_epochs):
    ...     for batch in dataloader:
    ...         x = batch["features"]  # (batch_size, 512)
    ...         eigenvalues, eigenvectors = estimator(x)
    ...
    ...         # Optional: use for dimensionality reduction
    ...         x_reduced = estimator.project(x)  # (batch_size, 16)
    >>>
    >>> # After training, eigenvectors are available
    >>> print(f"Top eigenvalue: {estimator.eigenvalues[0]:.4f}")
    >>> print(f"Variance explained: {estimator.explained_variance_ratio.sum():.2%}")

    DDP Usage:
    ----------
    >>> # Works automatically with DDP - no changes needed!
    >>> model = MyModel()
    >>> model.pca = StreamingTopKEigen(dim=256, k=16)
    >>> model = DDP(model, device_ids=[local_rank])
    >>>
    >>> for batch in dataloader:
    ...     # Statistics are automatically aggregated across all GPUs
    ...     eigenvalues, eigenvectors = model.module.pca(features)

    Parameters
    ----------
    dim : int
        Dimensionality of input features.
    k : int
        Number of top eigenvectors to estimate. Must satisfy k ≤ dim.
    device : torch.device, optional
        Device for tensors. If None, uses default device.
    dtype : torch.dtype, optional
        Data type for computations. Default is float32.
        Use float64 for higher precision if needed.
    sync_distributed : bool, optional
        Whether to synchronize across processes in distributed training.
        Default is True. Set to False if you want per-GPU estimates.

    Attributes:
    ----------
    V : torch.Tensor
        Current eigenvector estimates, shape (dim, k).
        Columns are eigenvectors, sorted by eigenvalue (descending).
    eigenvalues : torch.Tensor
        Current eigenvalue estimates, shape (k,), sorted descending.
    mean : torch.Tensor
        Running mean estimate, shape (dim,).
    n_samples : torch.Tensor
        Total number of samples seen so far.
    total_variance : torch.Tensor
        Estimate of total variance (trace of covariance matrix).
    initialized : torch.Tensor
        Boolean flag indicating if first batch has been processed.

    Notes:
    -----
    - The estimator uses `register_buffer` for all state, so it will be
      properly saved/loaded with `state_dict()` and moved with `.to(device)`.
    - All updates are performed in `torch.no_grad()` context - this module
      does not participate in gradient computation.
    - For very small batches (< k samples), consider accumulating batches
      before calling forward for better initialization.
    - In DDP mode, all processes will have identical state after each update.

    See Also:
    --------
    torch.pca_lowrank : PyTorch's built-in randomized PCA (non-streaming)
    sklearn.decomposition.IncrementalPCA : Scikit-learn's incremental PCA
    """

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        dim: int,
        k: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        sync_distributed: bool = True,
    ) -> None:
        """Initialize the streaming eigenvector estimator.

        Parameters
        ----------
        dim : int
            Input feature dimensionality.
        k : int
            Number of top eigenvectors to track.
        device : torch.device, optional
            Computation device (cpu/cuda).
        dtype : torch.dtype, optional
            Tensor dtype, default float32.
        sync_distributed : bool, optional
            Whether to sync across distributed processes. Default True.
        """
        super().__init__()

        # Validate inputs
        if k > dim:
            raise ValueError(
                f"k ({k}) cannot exceed dim ({dim}). "
                f"Requested more eigenvectors than dimensions."
            )
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")

        self.dim = dim
        self.k = k
        self.sync_distributed = sync_distributed

        # ---------------------------------------------------------------------
        # State buffers (will be saved with state_dict, moved with .to())
        # ---------------------------------------------------------------------

        # Eigenvector matrix: columns are eigenvectors, shape (dim, k)
        self.register_buffer("V", torch.empty(dim, k, device=device, dtype=dtype))

        # Running mean estimate, shape (dim,)
        self.register_buffer("mean", torch.zeros(dim, device=device, dtype=dtype))

        # Eigenvalue estimates (variance along each principal direction)
        self.register_buffer("eigenvalues", torch.ones(k, device=device, dtype=dtype))

        # Total sample count (as float for smooth division)
        self.register_buffer("n_samples", torch.tensor(0.0, device=device, dtype=dtype))

        # Initialization flag
        self.register_buffer("initialized", torch.tensor(False, device=device))

        # Total variance estimate (trace of covariance matrix)
        self.register_buffer(
            "total_variance", torch.tensor(1.0, device=device, dtype=dtype)
        )

    # =========================================================================
    # Distributed Utilities
    # =========================================================================

    def _is_distributed(self) -> bool:
        """Check if we should use distributed operations."""
        return self.sync_distributed and dist.is_available() and dist.is_initialized()

    @torch.no_grad()
    def _all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce tensor with SUM operation across processes.

        This is the core primitive for distributed aggregation.
        Each process contributes its local tensor, and all processes
        receive the sum.

        Parameters
        ----------
        tensor : torch.Tensor
            Local tensor to aggregate.

        Returns:
        -------
        torch.Tensor
            Sum across all processes (in-place modification of input).
        """
        if self._is_distributed():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    # =========================================================================
    # Initialization from First Batch (DDP-aware)
    # =========================================================================

    @torch.no_grad()
    def _init_from_batch(self, x: torch.Tensor) -> None:
        """Initialize eigenvector estimates from the first batch.

        In distributed mode, aggregates statistics across all processes
        before computing eigendecomposition. Uses covariance estimation
        rather than gathering raw data for memory efficiency.

        Strategy for DDP:
        1. Aggregate mean across processes via all-reduce
        2. Aggregate scatter matrix (X^T X) across processes via all-reduce
        3. Compute covariance from aggregated statistics
        4. Eigendecompose covariance to get initial eigenvectors

        This avoids gathering O(n*d) data to a single GPU, instead
        only communicating O(d^2) covariance statistics.
        """
        local_batch_size = x.shape[0]

        # ---------------------------------------------------------------------
        # Step 1: Compute global batch size and mean
        # ---------------------------------------------------------------------
        local_sum = x.sum(dim=0)  # (dim,)
        local_count = torch.tensor(
            float(local_batch_size), device=x.device, dtype=x.dtype
        )

        # Aggregate across processes
        if self._is_distributed():
            self._all_reduce_sum(local_sum)
            self._all_reduce_sum(local_count)

        global_batch_size = int(local_count.item())
        global_mean = local_sum / local_count

        # Set mean
        self.mean.copy_(global_mean)

        # Center local data with global mean
        x_centered = x - global_mean

        # ---------------------------------------------------------------------
        # Step 2: Compute global covariance via scatter matrix
        # ---------------------------------------------------------------------
        # Local scatter matrix: X^T X (unnormalized covariance contribution)
        local_scatter = x_centered.T @ x_centered  # (dim, dim)

        # Local sum of squared norms for total variance
        local_sq_norm_sum = (x_centered**2).sum()

        # Aggregate across processes
        if self._is_distributed():
            self._all_reduce_sum(local_scatter)
            self._all_reduce_sum(local_sq_norm_sum)

        # Global covariance estimate
        global_cov = local_scatter / local_count

        # Global total variance: trace(Cov) = E[||x - μ||²]
        self.total_variance.copy_(local_sq_norm_sum / local_count + 1e-8)

        # ---------------------------------------------------------------------
        # Step 3: Eigendecomposition of covariance
        # ---------------------------------------------------------------------
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(global_cov)
            # Reverse to get descending order (eigh returns ascending)
            eigenvalues = eigenvalues.flip(0)
            eigenvectors = eigenvectors.flip(1)
        except RuntimeError:
            # Fallback to SVD if eigh fails
            U, S, Vh = torch.linalg.svd(global_cov, full_matrices=False)
            eigenvalues = S
            eigenvectors = U

        # Take top-k
        k_valid = min(self.k, len(eigenvalues))

        # Filter out near-zero eigenvalues
        valid_mask = eigenvalues[:k_valid] > 1e-10
        k_valid = valid_mask.sum().item()

        if k_valid > 0:
            self.V[:, :k_valid] = eigenvectors[:, :k_valid]
            self.eigenvalues[:k_valid] = eigenvalues[:k_valid].clamp(min=1e-10)

        # ---------------------------------------------------------------------
        # Step 4: Handle remaining components (if k_valid < k)
        # ---------------------------------------------------------------------
        if k_valid < self.k:
            remaining_count = self.k - k_valid

            # Use deterministic seed for consistency across ranks
            generator = torch.Generator(device=x.device)
            generator.manual_seed(42 + global_batch_size)

            random_vecs = torch.randn(
                self.dim,
                remaining_count,
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )

            if k_valid > 0:
                # Project out existing eigenvectors
                existing = self.V[:, :k_valid]
                random_vecs = random_vecs - existing @ (existing.T @ random_vecs)

            # QR decomposition to get orthonormal vectors
            Q, R = torch.linalg.qr(random_vecs)

            n_from_qr = min(Q.shape[1], remaining_count)
            self.V[:, k_valid : k_valid + n_from_qr] = Q[:, :n_from_qr]

            # Handle edge case
            for i in range(k_valid + n_from_qr, self.k):
                v = torch.randn(self.dim, device=x.device, dtype=x.dtype)
                v = v - self.V[:, :i] @ (self.V[:, :i].T @ v)
                norm = v.norm()
                self.V[:, i] = v / norm if norm > 1e-8 else torch.randn_like(v)
                self.V[:, i] /= self.V[:, i].norm()

            # Set eigenvalues for remaining components
            if k_valid > 0:
                min_eigenvalue = self.eigenvalues[:k_valid].min() / 2
            else:
                min_eigenvalue = self.total_variance / self.dim
            self.eigenvalues[k_valid:] = min_eigenvalue

        # Update sample count and flag
        self.n_samples.fill_(float(global_batch_size))
        self.initialized.fill_(True)

    # =========================================================================
    # Adaptive Learning Rate Computation
    # =========================================================================

    @torch.no_grad()
    def _compute_adaptive_lr(self) -> torch.Tensor:
        """Compute per-component adaptive learning rates.

        The learning rate for component i is:

            η_i = base_lr × (total_variance / λ_i)

        where:
        - base_lr = 1/√n decays with sample count
        - total_variance / λ_i provides natural gradient scaling

        Returns:
        -------
        torch.Tensor
            Learning rates for each component, shape (k,).
        """
        base_lr = 1.0 / torch.sqrt(self.n_samples + 1.0)

        min_eigenvalue = self.total_variance * 1e-6
        lambda_safe = self.eigenvalues.clamp(min=min_eigenvalue)

        lr_per_component = base_lr * (self.total_variance / lambda_safe)

        return lr_per_component.clamp(min=0.001, max=1.0)

    # =========================================================================
    # Running Mean Update (DDP-aware)
    # =========================================================================

    @torch.no_grad()
    def _update_mean(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Update running mean estimate and return centered data.

        In distributed mode, aggregates across all processes to compute
        the true global mean.

        Parameters
        ----------
        x : torch.Tensor
            Input batch, shape (batch_size, dim).

        Returns:
        -------
        Tuple[torch.Tensor, int]
            - Centered data (x - updated_mean), shape (batch_size, dim).
            - Global batch size across all processes.
        """
        local_batch_size = x.shape[0]

        # Compute local statistics
        local_sum = x.sum(dim=0)
        local_count = torch.tensor(
            float(local_batch_size), device=x.device, dtype=x.dtype
        )

        # Aggregate across processes
        if self._is_distributed():
            self._all_reduce_sum(local_sum)
            self._all_reduce_sum(local_count)

        global_batch_size = int(local_count.item())
        batch_mean = local_sum / local_count

        n_total = self.n_samples + global_batch_size

        # Incremental mean update (Welford's algorithm)
        self.mean.mul_(self.n_samples / n_total)
        self.mean.add_(batch_mean, alpha=global_batch_size / n_total)

        # Return centered data using the NEW mean
        return x - self.mean, global_batch_size

    # =========================================================================
    # Core Update: Sanger's Rule (DDP-aware)
    # =========================================================================

    @torch.no_grad()
    def _sanger_update(self, x_centered: torch.Tensor, global_batch_size: int) -> None:
        """Perform one step of Sanger's rule (Generalized Hebbian Algorithm).

        In distributed mode, aggregates gradient statistics across all
        processes before applying the update.

        Distributed Aggregation Strategy:
        ---------------------------------
        Instead of gathering raw data (O(n*d) communication), we aggregate:
        - Hebbian term: sum of x̃ yᵀ across processes (O(d*k))
        - Projection covariance: sum of y yᵀ across processes (O(k²))
        - Eigenvalue statistics: sum of y² across processes (O(k))
        - Total variance: sum of ||x̃||² across processes (O(1))

        Parameters
        ----------
        x_centered : torch.Tensor
            Centered input data, shape (local_batch_size, dim).
        global_batch_size : int
            Total batch size across all processes.
        """
        # Get adaptive learning rates
        lr = self._compute_adaptive_lr()

        # ---------------------------------------------------------------------
        # Step 1: Compute local projections
        # ---------------------------------------------------------------------
        proj = x_centered @ self.V  # (local_batch_size, k)

        # ---------------------------------------------------------------------
        # Step 2: Compute local statistics (unnormalized - sums, not means)
        # ---------------------------------------------------------------------
        local_hebbian_sum = x_centered.T @ proj  # (dim, k)
        local_proj_cov_sum = proj.T @ proj  # (k, k)
        local_eigenvalue_sum = (proj**2).sum(dim=0)  # (k,)
        local_total_var_sum = (x_centered**2).sum()  # scalar

        # ---------------------------------------------------------------------
        # Step 3: Aggregate across processes (if distributed)
        # ---------------------------------------------------------------------
        if self._is_distributed():
            self._all_reduce_sum(local_hebbian_sum)
            self._all_reduce_sum(local_proj_cov_sum)
            self._all_reduce_sum(local_eigenvalue_sum)
            self._all_reduce_sum(local_total_var_sum)

        # ---------------------------------------------------------------------
        # Step 4: Normalize by global batch size to get expectations
        # ---------------------------------------------------------------------
        hebbian_term = local_hebbian_sum / global_batch_size
        proj_cov = local_proj_cov_sum / global_batch_size
        batch_eigenvalues = local_eigenvalue_sum / global_batch_size
        batch_total_var = local_total_var_sum / global_batch_size

        # ---------------------------------------------------------------------
        # Step 5: Compute deflation term
        # ---------------------------------------------------------------------
        lower_tri = torch.tril(proj_cov)
        deflation_term = self.V @ lower_tri

        # ---------------------------------------------------------------------
        # Step 6: Compute gradient and apply update
        # ---------------------------------------------------------------------
        gradient = hebbian_term - deflation_term
        scaled_gradient = gradient * lr.unsqueeze(0)
        self.V.add_(scaled_gradient)

        # ---------------------------------------------------------------------
        # Step 7: Re-orthogonalize using QR decomposition
        # ---------------------------------------------------------------------
        Q, R = torch.linalg.qr(self.V)
        signs = torch.diag(R).sign()
        signs[signs == 0] = 1
        self.V.copy_(Q * signs.unsqueeze(0))

        # ---------------------------------------------------------------------
        # Step 8: Update eigenvalue estimates
        # ---------------------------------------------------------------------
        eigenvalue_lr = 2.0 / torch.sqrt(self.n_samples + 1.0)
        eigenvalue_lr = eigenvalue_lr.clamp(max=0.5)
        self.eigenvalues.lerp_(batch_eigenvalues, eigenvalue_lr)

        # ---------------------------------------------------------------------
        # Step 9: Update total variance estimate
        # ---------------------------------------------------------------------
        self.total_variance.lerp_(batch_total_var, eigenvalue_lr)

        # ---------------------------------------------------------------------
        # Step 10: Update sample count (with global count!)
        # ---------------------------------------------------------------------
        self.n_samples.add_(global_batch_size)

    # =========================================================================
    # Main Forward Method
    # =========================================================================

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update eigenvector estimates with a new batch of data.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (batch_size, dim).

        Returns:
        -------
        eigenvalues : torch.Tensor
            Current eigenvalue estimates, shape (k,).
        eigenvectors : torch.Tensor
            Current eigenvector estimates, shape (dim, k).
        """
        # Input validation
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D input (batch_size, dim), got {x.dim()}D tensor"
            )
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected dim={self.dim}, got {x.shape[1]}. Input shape: {x.shape}"
            )

        # Dispatch to initialization or update
        if not self.initialized:
            self._init_from_batch(x)
        else:
            x_centered, global_batch_size = self._update_mean(x)
            self._sanger_update(x_centered, global_batch_size)

        return self.eigenvalues.clone(), self.V.clone()

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project data onto the estimated principal subspace."""
        return (x - self.mean) @ self.V

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Project and reconstruct data (PCA denoising/compression)."""
        z = self.project(x)
        return z @ self.V.T + self.mean

    @property
    def explained_variance_ratio(self) -> torch.Tensor:
        """Fraction of total variance explained by each component."""
        return self.eigenvalues / (self.total_variance + 1e-8)

    @property
    def cumulative_explained_variance_ratio(self) -> torch.Tensor:
        """Cumulative explained variance ratio."""
        return self.explained_variance_ratio.cumsum(dim=0)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"dim={self.dim}, k={self.k}, "
            f"n_samples={int(self.n_samples.item())}, "
            f"initialized={self.initialized.item()}"
            f")"
        )
