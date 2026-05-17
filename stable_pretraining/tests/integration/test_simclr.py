"""Integration tests for SimCLR functionality."""

import pytest
import torch

import stable_pretraining as spt


@pytest.mark.integration
class TestSimCLRIntegration:
    """Integration tests for SimCLR with actual training."""

    def test_multi_view_data_loading(self):
        """Test multi-view data loading for SimCLR."""
        # Create dummy dataset
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 3, 224, 224), torch.randint(0, 10, (100,))
        )

        # Create multi-view sampler
        sampler = spt.data.sampler.RepeatedRandomSampler(dataset, n_views=2)

        # Create dataloader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            sampler=sampler,
        )

        # Get a batch
        images, labels = next(iter(loader))

        # Verify multi-view structure
        assert images.shape[0] == 8  # batch_size
        # Sample indices should have repeats for multi-view

    def test_fold_views_operation(self):
        """Test fold_views operation for multi-view data."""
        batch_size = 16
        n_views = 2
        feature_dim = 128

        # Create multi-view features
        features = torch.randn(batch_size, feature_dim)
        sample_idx = torch.repeat_interleave(
            torch.arange(batch_size // n_views), n_views
        )

        # Fold views
        views = spt.data.fold_views(features, sample_idx)

        # Verify views
        assert len(views) == n_views
        assert all(v.shape == (batch_size // n_views, feature_dim) for v in views)

    @pytest.mark.gpu
    def test_projector_architecture(self):
        """Test different projector architectures for SimCLR."""
        input_dim = 512
        hidden_dim = 256
        output_dim = 128

        # Linear projector
        linear_proj = torch.nn.Linear(input_dim, output_dim)

        # MLP projector (standard for SimCLR)
        mlp_proj = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

        # Test forward pass
        x = torch.randn(32, input_dim)

        linear_out = linear_proj(x)
        mlp_out = mlp_proj(x)

        assert linear_out.shape == (32, output_dim)
        assert mlp_out.shape == (32, output_dim)

    def test_temperature_scaling(self):
        """Test temperature scaling in SimCLR loss."""
        feature_dim = 128

        # Create normalized features
        z1 = torch.nn.functional.normalize(torch.randn(8, feature_dim), dim=1)
        z2 = torch.nn.functional.normalize(torch.randn(8, feature_dim), dim=1)

        # Test different temperatures
        for temp in [0.05, 0.1, 0.5, 1.0]:
            loss_fn = spt.losses.NTXEntLoss(temperature=temp)
            loss = loss_fn(z1, z2)
            assert loss.item() > 0
