"""Unit tests for RINCELoss."""

import pytest
import torch
import torch.nn.functional as F

from stable_pretraining.losses.joint_embedding import RINCELoss


@pytest.mark.unit
class TestRINCELoss:
    """Unit tests for the RINCE loss (single process)."""

    @pytest.mark.parametrize("q", [0.2, 0.5, 1.0])
    def test_loss_matches_reference(self, q):
        """Test RINCE against its analytical equation."""
        torch.manual_seed(42)

        z_i = torch.randn(4, 128)
        z_j = torch.randn(4, 128)

        loss_fn = RINCELoss(temperature=0.5, q=q, lam=0.5)
        actual = loss_fn(z_i, z_j)

        z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=-1)
        logits = (z @ z.T) / 0.5
        logits.fill_diagonal_(-torch.inf)

        idx = torch.arange(8)
        pos = logits[idx, (idx + 4) % 8]

        expected = (
            ((0.5 * logits.exp().sum(dim=1)).pow(q) - pos.exp().pow(q)).div(q).mean()
        )

        assert actual.ndim == 0
        torch.testing.assert_close(actual, expected)

    def test_loss_gradients(self):
        """Test that RINCE produces finite loss and gradients."""
        torch.manual_seed(42)

        z_i = torch.randn(4, 128, requires_grad=True)
        z_j = torch.randn(4, 128, requires_grad=True)

        loss = RINCELoss()(z_i, z_j)
        loss.backward()

        assert torch.isfinite(loss).item()
        assert torch.isfinite(z_i.grad).all()
        assert torch.isfinite(z_j.grad).all()

    @pytest.mark.parametrize("lam", [0.01, 1.0, 2.0])
    def test_valid_lam(self, lam):
        loss_fn = RINCELoss(lam=lam)
        assert loss_fn.lam == lam
