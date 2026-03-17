"""Test suite for StreamingTopKEigen.

Run with:
    pytest test_streaming_eigen.py -v -m unit
    pytest test_streaming_eigen.py -v --tb=short
"""

import pytest
import torch
from typing import Tuple

# =============================================================================
# Import the module under test (assuming it's in streaming_eigen.py)
# If running standalone, paste the StreamingTopKEigen class above this line
# =============================================================================

# For standalone testing, we include a minimal version here:
# (In production, replace with: from streaming_eigen import StreamingTopKEigen)

from stable_pretraining.utils import StreamingTopKEigen

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def seed():
    """Set deterministic seed for reproducibility."""
    torch.manual_seed(42)
    return 42


@pytest.fixture
def synthetic_data_simple(seed) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create synthetic data with known covariance structure.

    Returns:
        data: (n_samples, dim) data matrix
        true_eigenvalues: (dim,) true eigenvalues sorted descending
        true_eigenvectors: (dim, dim) true eigenvectors as columns
    """
    dim = 64
    n_samples = 2000

    # Create covariance with known eigenstructure
    # Eigenvalues: 10, 5, 2, 1, 0.5, 0.1, ...
    true_eigenvalues = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5] + [0.1] * (dim - 5))

    # Random orthogonal eigenvectors
    Q, _ = torch.linalg.qr(torch.randn(dim, dim))
    true_eigenvectors = Q

    # Construct covariance and sample
    cov = Q @ torch.diag(true_eigenvalues) @ Q.T
    L = torch.linalg.cholesky(cov)
    data = torch.randn(n_samples, dim) @ L.T

    return data, true_eigenvalues, true_eigenvectors


@pytest.fixture
def synthetic_data_with_mean(
    seed,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create synthetic data with known mean and covariance.

    Returns:
        data: (n_samples, dim) data matrix
        true_mean: (dim,) true mean
        true_eigenvalues: (dim,) true eigenvalues
        true_eigenvectors: (dim, dim) true eigenvectors
    """
    dim = 32
    n_samples = 1500

    true_mean = torch.randn(dim) * 5  # Non-zero mean
    true_eigenvalues = torch.logspace(1, -1, dim)  # Log-spaced from 10 to 0.1

    Q, _ = torch.linalg.qr(torch.randn(dim, dim))
    true_eigenvectors = Q

    cov = Q @ torch.diag(true_eigenvalues) @ Q.T
    L = torch.linalg.cholesky(cov)
    data = torch.randn(n_samples, dim) @ L.T + true_mean

    return data, true_mean, true_eigenvalues, true_eigenvectors


# =============================================================================
# Helper Functions
# =============================================================================


def subspace_alignment(V_est: torch.Tensor, V_true: torch.Tensor) -> float:
    """Compute alignment between estimated and true subspaces.

    Uses average of top singular values of V_est^T @ V_true.
    Returns 1.0 for perfect alignment, 0.0 for orthogonal subspaces.
    """
    k = V_est.shape[1]
    V_true_k = V_true[:, :k]

    # Singular values of projection
    S = torch.linalg.svdvals(V_est.T @ V_true_k)

    # Average alignment (each singular value is between 0 and 1)
    return S.mean().item()


def relative_error(estimated: torch.Tensor, true: torch.Tensor) -> float:
    """Compute relative error: ||est - true|| / ||true||."""
    return (estimated - true).norm().item() / (true.norm().item() + 1e-8)


# =============================================================================
# Unit Tests: Initialization
# =============================================================================


@pytest.mark.unit
class TestInitialization:
    """Tests for module initialization."""

    def test_basic_init(self):
        """Test basic initialization with valid parameters."""
        estimator = StreamingTopKEigen(dim=128, k=16)

        assert estimator.dim == 128
        assert estimator.k == 16
        assert estimator.V.shape == (128, 16)
        assert estimator.mean.shape == (128,)
        assert estimator.eigenvalues.shape == (16,)
        assert not estimator.initialized.item()
        assert estimator.n_samples.item() == 0.0

    def test_init_with_dtype(self):
        """Test initialization with different dtypes."""
        estimator_f32 = StreamingTopKEigen(dim=64, k=8, dtype=torch.float32)
        estimator_f64 = StreamingTopKEigen(dim=64, k=8, dtype=torch.float64)

        assert estimator_f32.V.dtype == torch.float32
        assert estimator_f64.V.dtype == torch.float64

    def test_invalid_k_greater_than_dim(self):
        """Test that k > dim raises ValueError."""
        with pytest.raises(ValueError, match="k .* cannot exceed dim"):
            StreamingTopKEigen(dim=10, k=20)

    def test_invalid_k_zero(self):
        """Test that k=0 raises ValueError."""
        with pytest.raises(ValueError, match="k must be positive"):
            StreamingTopKEigen(dim=10, k=0)

    def test_invalid_k_negative(self):
        """Test that negative k raises ValueError."""
        with pytest.raises(ValueError, match="k must be positive"):
            StreamingTopKEigen(dim=10, k=-5)

    def test_repr(self):
        """Test string representation."""
        estimator = StreamingTopKEigen(dim=100, k=10)
        repr_str = repr(estimator)

        assert "StreamingTopKEigen" in repr_str
        assert "dim=100" in repr_str
        assert "k=10" in repr_str
        assert "initialized=False" in repr_str


# =============================================================================
# Unit Tests: Forward Pass
# =============================================================================


@pytest.mark.unit
class TestForwardPass:
    """Tests for forward pass behavior."""

    def test_first_forward_initializes(self, seed):
        """Test that first forward call initializes the estimator."""
        estimator = StreamingTopKEigen(dim=32, k=4)
        x = torch.randn(100, 32)

        assert not estimator.initialized.item()
        eigenvalues, eigenvectors = estimator(x)
        assert estimator.initialized.item()
        assert estimator.n_samples.item() == 100.0

    def test_forward_output_shapes(self, seed):
        """Test output shapes of forward pass."""
        dim, k, batch_size = 64, 8, 50
        estimator = StreamingTopKEigen(dim=dim, k=k)
        x = torch.randn(batch_size, dim)

        eigenvalues, eigenvectors = estimator(x)

        assert eigenvalues.shape == (k,)
        assert eigenvectors.shape == (dim, k)

    def test_forward_multiple_batches(self, seed):
        """Test that multiple forward calls accumulate samples."""
        estimator = StreamingTopKEigen(dim=32, k=4)
        batch_size = 64

        for i in range(5):
            x = torch.randn(batch_size, 32)
            estimator(x)

        assert estimator.n_samples.item() == 5 * batch_size

    def test_forward_wrong_dim(self, seed):
        """Test that wrong input dimension raises error."""
        estimator = StreamingTopKEigen(dim=32, k=4)
        x = torch.randn(100, 64)  # Wrong dim

        with pytest.raises(ValueError, match="Expected dim=32"):
            estimator(x)

    def test_forward_1d_input(self, seed):
        """Test that 1D input raises error."""
        estimator = StreamingTopKEigen(dim=32, k=4)
        x = torch.randn(32)

        with pytest.raises(ValueError, match="Expected 2D input"):
            estimator(x)

    def test_forward_3d_input(self, seed):
        """Test that 3D input raises error."""
        estimator = StreamingTopKEigen(dim=32, k=4)
        x = torch.randn(10, 5, 32)

        with pytest.raises(ValueError, match="Expected 2D input"):
            estimator(x)

    def test_eigenvectors_orthonormal(self, seed):
        """Test that output eigenvectors are orthonormal."""
        estimator = StreamingTopKEigen(dim=64, k=8)
        x = torch.randn(500, 64)

        _, V = estimator(x)

        # V^T V should be identity
        VtV = V.T @ V
        identity = torch.eye(8)

        assert torch.allclose(VtV, identity, atol=1e-5)

    def test_eigenvalues_non_negative(self, seed):
        """Test that eigenvalues are non-negative."""
        estimator = StreamingTopKEigen(dim=64, k=8)

        for _ in range(10):
            eigenvalues, _ = estimator(torch.randn(100, 64))

        assert (eigenvalues >= 0).all()

    def test_eigenvalues_sorted_descending(self, synthetic_data_simple):
        """Test that eigenvalues are approximately sorted (descending)."""
        data, _, _ = synthetic_data_simple
        estimator = StreamingTopKEigen(dim=64, k=8)

        # Process all data in batches
        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            eigenvalues, _ = estimator(batch)

        # After convergence, should be approximately sorted
        # Allow some tolerance due to streaming estimation
        for i in range(len(eigenvalues) - 1):
            # Each eigenvalue should be >= 0.5x the next (loose bound)
            assert eigenvalues[i] >= eigenvalues[i + 1] * 0.5


# =============================================================================
# Unit Tests: Convergence
# =============================================================================


@pytest.mark.unit
class TestConvergence:
    """Tests for convergence to true eigenvectors."""

    def test_converges_to_true_eigenvectors(self, synthetic_data_simple):
        """Test convergence to true principal subspace."""
        data, true_eigenvalues, true_eigenvectors = synthetic_data_simple
        k = 5
        estimator = StreamingTopKEigen(dim=64, k=k)

        # Process all data
        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            eigenvalues, V = estimator(batch)

        # Check subspace alignment
        alignment = subspace_alignment(V, true_eigenvectors)

        # Should achieve good alignment (> 0.9)
        assert alignment > 0.9, f"Subspace alignment {alignment:.4f} < 0.9"

    def test_eigenvalue_estimates_converge(self, synthetic_data_simple):
        """Test that eigenvalue estimates converge to true values."""
        data, true_eigenvalues, _ = synthetic_data_simple
        k = 5
        estimator = StreamingTopKEigen(dim=64, k=k)

        # Process all data
        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            eigenvalues, _ = estimator(batch)

        # Check relative error of eigenvalue estimates
        true_k = true_eigenvalues[:k]
        rel_error = relative_error(eigenvalues, true_k)

        # Should achieve < 20% relative error
        assert rel_error < 0.2, f"Eigenvalue relative error {rel_error:.4f} >= 0.2"

    def test_mean_estimate_converges(self, synthetic_data_with_mean):
        """Test that mean estimate converges to true mean."""
        data, true_mean, _, _ = synthetic_data_with_mean
        estimator = StreamingTopKEigen(dim=32, k=4)

        batch_size = 100
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            estimator(batch)

        mean_error = relative_error(estimator.mean, true_mean)

        # Should achieve < 5% relative error on mean
        assert mean_error < 0.05, f"Mean relative error {mean_error:.4f} >= 0.05"

    def test_convergence_improves_with_samples(self, seed):
        """Test that alignment improves as more samples are processed."""
        dim, k = 32, 4

        # Create simple data
        true_eigenvalues = torch.tensor([5.0, 2.0, 1.0, 0.5] + [0.1] * (dim - 4))
        Q, _ = torch.linalg.qr(torch.randn(dim, dim))
        cov = Q @ torch.diag(true_eigenvalues) @ Q.T
        L = torch.linalg.cholesky(cov)

        estimator = StreamingTopKEigen(dim=dim, k=k)

        alignments = []
        for _ in range(20):
            x = torch.randn(100, dim) @ L.T
            _, V = estimator(x)
            alignments.append(subspace_alignment(V, Q))

        # Alignment should generally improve (compare first half avg vs second half avg)
        first_half = sum(alignments[:10]) / 10
        second_half = sum(alignments[10:]) / 10

        assert second_half >= first_half - 0.05, (
            f"Alignment did not improve: {first_half:.4f} -> {second_half:.4f}"
        )


# =============================================================================
# Unit Tests: Projection and Reconstruction
# =============================================================================


@pytest.mark.unit
class TestProjectionReconstruction:
    """Tests for project and reconstruct methods."""

    def test_project_shape(self, seed):
        """Test projection output shape."""
        estimator = StreamingTopKEigen(dim=64, k=8)
        x = torch.randn(100, 64)
        estimator(x)

        z = estimator.project(x)
        assert z.shape == (100, 8)

    def test_project_single_sample(self, seed):
        """Test projection of single sample."""
        estimator = StreamingTopKEigen(dim=64, k=8)
        x = torch.randn(100, 64)
        estimator(x)

        single = x[0:1]
        z = estimator.project(single)
        assert z.shape == (1, 8)

    def test_reconstruct_shape(self, seed):
        """Test reconstruction output shape."""
        estimator = StreamingTopKEigen(dim=64, k=8)
        x = torch.randn(100, 64)
        estimator(x)

        x_recon = estimator.reconstruct(x)
        assert x_recon.shape == x.shape

    def test_reconstruction_error_bounded(self, synthetic_data_simple):
        """Test that reconstruction captures most variance."""
        data, true_eigenvalues, _ = synthetic_data_simple
        k = 5
        estimator = StreamingTopKEigen(dim=64, k=k)

        # Train on all data
        batch_size = 100
        for i in range(0, len(data), batch_size):
            estimator(data[i : i + batch_size])

        # Compute reconstruction error on held-out data
        test_data = torch.randn(200, 64)
        cov = data.T @ data / len(data)
        L = torch.linalg.cholesky(cov + torch.eye(64) * 1e-6)
        test_data = test_data @ L.T

        x_recon = estimator.reconstruct(test_data)
        recon_error = ((test_data - x_recon) ** 2).mean()
        total_var = ((test_data - test_data.mean(dim=0)) ** 2).mean()

        # Reconstruction should capture significant variance
        explained = 1 - recon_error / total_var
        assert explained > 0.7, f"Only {explained:.1%} variance explained"

    def test_double_projection_idempotent(self, seed):
        """Test that projecting reconstructed data gives same projection."""
        estimator = StreamingTopKEigen(dim=64, k=8)
        x = torch.randn(100, 64)
        estimator(x)

        z1 = estimator.project(x)
        x_recon = estimator.reconstruct(x)
        z2 = estimator.project(x_recon)

        # z1 and z2 should be nearly identical
        assert torch.allclose(z1, z2, atol=1e-5)


# =============================================================================
# Unit Tests: Properties
# =============================================================================


@pytest.mark.unit
class TestProperties:
    """Tests for computed properties."""

    def test_explained_variance_ratio_range(self, seed):
        """Test that explained variance ratios are in [0, 1]."""
        estimator = StreamingTopKEigen(dim=64, k=8)
        x = torch.randn(500, 64)
        estimator(x)

        evr = estimator.explained_variance_ratio

        assert (evr >= 0).all()
        assert (evr <= 1).all()

    def test_explained_variance_ratio_sums_less_than_one(self, seed):
        """Test that explained variance ratios sum to <= 1."""
        estimator = StreamingTopKEigen(dim=64, k=8)
        x = torch.randn(500, 64)
        estimator(x)

        total = estimator.explained_variance_ratio.sum()

        assert total <= 1.0 + 1e-5, f"Sum of EVR = {total:.4f} > 1"

    def test_cumulative_explained_variance_monotonic(self, seed):
        """Test that cumulative explained variance is monotonically increasing."""
        estimator = StreamingTopKEigen(dim=64, k=8)
        x = torch.randn(500, 64)
        estimator(x)

        cumvar = estimator.cumulative_explained_variance_ratio

        for i in range(len(cumvar) - 1):
            assert cumvar[i] <= cumvar[i + 1] + 1e-6

    def test_total_variance_positive(self, seed):
        """Test that total variance estimate is positive."""
        estimator = StreamingTopKEigen(dim=64, k=8)

        for _ in range(10):
            x = torch.randn(100, 64) * 5  # Scale up
            estimator(x)

        assert estimator.total_variance.item() > 0


# =============================================================================
# Unit Tests: Edge Cases
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_batch_initialization(self, seed):
        """Test initialization with batch smaller than k."""
        estimator = StreamingTopKEigen(dim=32, k=16)
        x = torch.randn(8, 32)  # batch_size < k

        eigenvalues, V = estimator(x)

        # Should still work and produce orthonormal vectors
        assert estimator.initialized.item()
        VtV = V.T @ V
        assert torch.allclose(VtV, torch.eye(16), atol=1e-5)

    def test_single_sample_batch(self, seed):
        """Test with single sample batches."""
        estimator = StreamingTopKEigen(dim=16, k=4)

        # Initialize with reasonable batch
        x_init = torch.randn(50, 16)
        estimator(x_init)

        # Then process single samples
        for _ in range(20):
            x = torch.randn(1, 16)
            eigenvalues, V = estimator(x)

        # Should still be orthonormal
        VtV = V.T @ V
        assert torch.allclose(VtV, torch.eye(4), atol=1e-4)

    def test_k_equals_dim(self, seed):
        """Test when k equals dim (full eigendecomposition)."""
        dim = 16
        estimator = StreamingTopKEigen(dim=dim, k=dim)
        x = torch.randn(100, dim)

        eigenvalues, V = estimator(x)

        assert V.shape == (dim, dim)
        VtV = V.T @ V
        assert torch.allclose(VtV, torch.eye(dim), atol=1e-5)

    def test_k_equals_one(self, seed):
        """Test when k=1 (just first principal component)."""
        estimator = StreamingTopKEigen(dim=32, k=1)
        x = torch.randn(200, 32)

        eigenvalues, V = estimator(x)

        assert eigenvalues.shape == (1,)
        assert V.shape == (32, 1)
        assert torch.allclose(V.T @ V, torch.ones(1, 1), atol=1e-5)

    def test_large_batch_size(self, seed):
        """Test with batch larger than dim."""
        estimator = StreamingTopKEigen(dim=16, k=4)
        x = torch.randn(1000, 16)  # batch >> dim

        eigenvalues, V = estimator(x)

        assert estimator.initialized.item()
        assert torch.allclose(V.T @ V, torch.eye(4), atol=1e-5)

    def test_zero_variance_direction(self, seed):
        """Test with data that has zero variance in some directions."""
        dim, k = 32, 8
        estimator = StreamingTopKEigen(dim=dim, k=k)

        # Create data with variance only in first 10 dimensions
        x = torch.zeros(200, dim)
        x[:, :10] = torch.randn(200, 10)

        for _ in range(5):
            eigenvalues, V = estimator(x + torch.randn_like(x) * 0.01)

        # Should not crash and should maintain orthonormality
        VtV = V.T @ V
        assert torch.allclose(VtV, torch.eye(k), atol=1e-4)

    def test_scaled_data(self, seed):
        """Test with differently scaled data."""
        estimator = StreamingTopKEigen(dim=32, k=4)

        # Very large values
        x_large = torch.randn(100, 32) * 1000
        estimator(x_large)

        # Very small values
        x_small = torch.randn(100, 32) * 0.001
        eigenvalues, V = estimator(x_small)

        # Should still be orthonormal
        VtV = V.T @ V
        assert torch.allclose(VtV, torch.eye(4), atol=1e-4)

    def test_constant_data(self, seed):
        """Test with constant data (edge case)."""
        estimator = StreamingTopKEigen(dim=16, k=4)

        # Initialize normally
        x_init = torch.randn(50, 16)
        estimator(x_init)

        # Then constant data
        x_const = torch.ones(20, 16) * 5.0
        eigenvalues, V = estimator(x_const)

        # Should handle gracefully
        assert not torch.isnan(eigenvalues).any()
        assert not torch.isnan(V).any()


# =============================================================================
# Unit Tests: Numerical Stability
# =============================================================================


@pytest.mark.unit
class TestNumericalStability:
    """Tests for numerical stability."""

    def test_no_nan_after_many_updates(self, seed):
        """Test that no NaN values appear after many updates."""
        estimator = StreamingTopKEigen(dim=64, k=8)

        for _ in range(100):
            x = torch.randn(50, 64)
            eigenvalues, V = estimator(x)

        assert not torch.isnan(eigenvalues).any()
        assert not torch.isnan(V).any()
        assert not torch.isnan(estimator.mean).any()

    def test_no_inf_values(self, seed):
        """Test that no Inf values appear."""
        estimator = StreamingTopKEigen(dim=64, k=8)

        for _ in range(50):
            x = torch.randn(100, 64) * 10
            eigenvalues, V = estimator(x)

        assert not torch.isinf(eigenvalues).any()
        assert not torch.isinf(V).any()

    def test_orthonormality_preserved(self, seed):
        """Test that eigenvectors stay orthonormal over many updates."""
        estimator = StreamingTopKEigen(dim=64, k=8)

        for i in range(100):
            x = torch.randn(50, 64)
            eigenvalues, V = estimator(x)

            # Check orthonormality every 10 steps
            if (i + 1) % 10 == 0:
                VtV = V.T @ V
                error = (VtV - torch.eye(8)).abs().max()
                assert error < 1e-4, (
                    f"Orthonormality error at step {i + 1}: {error:.6f}"
                )

    def test_float64_precision(self, seed):
        """Test with float64 for higher precision."""
        estimator = StreamingTopKEigen(dim=32, k=4, dtype=torch.float64)

        for _ in range(50):
            x = torch.randn(100, 32, dtype=torch.float64)
            eigenvalues, V = estimator(x)

        VtV = V.T @ V
        error = (VtV - torch.eye(4, dtype=torch.float64)).abs().max()

        # Should have better precision than float32
        assert error < 1e-10


# =============================================================================
# Unit Tests: State Management
# =============================================================================


@pytest.mark.unit
class TestStateManagement:
    """Tests for state saving/loading."""

    def test_state_dict_complete(self, seed):
        """Test that state_dict contains all buffers."""
        estimator = StreamingTopKEigen(dim=32, k=4)
        x = torch.randn(100, 32)
        estimator(x)

        state = estimator.state_dict()

        expected_keys = {
            "V",
            "mean",
            "eigenvalues",
            "n_samples",
            "initialized",
            "total_variance",
        }
        assert expected_keys == set(state.keys())

    def test_load_state_dict(self, seed):
        """Test loading state dict restores state."""
        # Create and train estimator
        estimator1 = StreamingTopKEigen(dim=32, k=4)
        x = torch.randn(200, 32)
        estimator1(x)

        # Save state
        state = estimator1.state_dict()

        # Create new estimator and load state
        estimator2 = StreamingTopKEigen(dim=32, k=4)
        estimator2.load_state_dict(state)

        # Compare
        assert torch.allclose(estimator1.V, estimator2.V)
        assert torch.allclose(estimator1.eigenvalues, estimator2.eigenvalues)
        assert torch.allclose(estimator1.mean, estimator2.mean)
        assert estimator1.n_samples.item() == estimator2.n_samples.item()

    def test_loaded_state_continues_training(self, seed):
        """Test that loaded model can continue training."""
        # Train first model
        estimator1 = StreamingTopKEigen(dim=32, k=4)
        for _ in range(10):
            estimator1(torch.randn(50, 32))

        # Load into second model
        estimator2 = StreamingTopKEigen(dim=32, k=4)
        estimator2.load_state_dict(estimator1.state_dict())

        # Continue training second model
        for _ in range(10):
            eigenvalues, V = estimator2(torch.randn(50, 32))

        # Should have more samples
        assert estimator2.n_samples.item() > estimator1.n_samples.item()

        # Should still be valid
        VtV = V.T @ V
        assert torch.allclose(VtV, torch.eye(4), atol=1e-4)


# =============================================================================
# Unit Tests: Determinism
# =============================================================================


@pytest.mark.unit
class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_seed_same_result(self):
        """Test that same seed produces same results."""
        results = []

        for _ in range(2):
            torch.manual_seed(12345)
            estimator = StreamingTopKEigen(dim=32, k=4)
            x = torch.randn(100, 32)
            eigenvalues, V = estimator(x)
            results.append((eigenvalues.clone(), V.clone()))

        assert torch.allclose(results[0][0], results[1][0])
        assert torch.allclose(results[0][1], results[1][1])

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        results = []

        for seed_val in [111, 222]:
            torch.manual_seed(seed_val)
            estimator = StreamingTopKEigen(dim=32, k=4)
            x = torch.randn(100, 32)
            eigenvalues, V = estimator(x)
            results.append(V.clone())

        # Should be different (not identical)
        assert not torch.allclose(results[0], results[1])


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit", "--tb=short"])
