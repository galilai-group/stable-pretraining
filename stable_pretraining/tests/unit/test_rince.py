"""Unit tests for RINCE functionality."""

import pytest
import torch
import torch.nn as nn

from stable_pretraining.losses import RINCELoss
from stable_pretraining.methods import RINCE


@pytest.fixture
def encoder():
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, 16),
    )


@pytest.mark.unit
class TestRINCEUnit:
    """Unit tests for RINCE components without actual training."""

    def test_rince_initialization(self, encoder):
        """Test RINCE initialization and loss configuration."""
        model = RINCE(
            encoder_name=encoder,
            projector_dims=(32, 16),
            q=0.5,
            lam=0.5,
        )

        assert model.backbone is encoder
        assert isinstance(model.simclr_loss, RINCELoss)
        assert model.simclr_loss.q == 0.5
        assert model.simclr_loss.lam == 0.5

    def test_rince_forward(self, encoder):
        """Test forward pass with two augmented views."""
        model = RINCE(encoder_name=encoder, projector_dims=(32, 16))

        v1 = torch.randn(4, 3, 32, 32)
        v2 = torch.randn(4, 3, 32, 32)

        output = model(v1, v2)

        assert output.loss.ndim == 0
        assert torch.isfinite(output.loss)
        assert output.embedding.shape == (8, 16)
        assert output.projection.shape == (8, 16)

    def test_rince_gradients(self, encoder):
        """Test gradient propagation through the complete method."""
        model = RINCE(encoder_name=encoder, projector_dims=(32, 16))

        v1 = torch.randn(4, 3, 32, 32)
        v2 = torch.randn(4, 3, 32, 32)

        output = model(v1, v2)
        output.loss.backward()

        assert encoder[2].weight.grad is not None
        assert torch.isfinite(encoder[2].weight.grad).all()

    def test_rince_eval_mode(self, encoder):
        """Test single-view inference without contrastive loss."""
        model = RINCE(encoder_name=encoder, projector_dims=(32, 16))
        model.eval()

        image = torch.randn(4, 3, 32, 32)
        output = model(image)

        assert output.loss.item() == 0.0
        assert output.embedding.shape == (4, 16)
        assert output.projection is None

    @pytest.mark.parametrize("q", [-0.1, 0.0, 1.5])
    def test_invalid_q(self, encoder, q):
        """Test rejection of invalid q values."""
        with pytest.raises(ValueError):
            RINCE(encoder_name=encoder, projector_dims=(32, 16), q=q)
