import math

import pytest
import torch
import torch.nn.functional as F

from stable_pretraining.losses import SigLIPLoss


@pytest.mark.unit
class TestSigLIPLoss:
    """Unit tests for the SigLIPLoss function."""

    def test_loss_is_lower_for_matched_pairs_than_mismatched_pairs(self):
        """Loss should be lower when positive pairs are aligned on the diagonal."""
        torch.manual_seed(0)
        batch_size, dim = 4, 8
        feats_i = F.normalize(torch.eye(dim)[:batch_size], dim=-1)
        feats_j = feats_i.clone()
        loss_fn = SigLIPLoss()

        matched_loss = loss_fn(image_features=feats_i, text_features=feats_j)
        mismatched_loss = loss_fn(
            image_features=feats_i,
            text_features=torch.flip(feats_j, dims=[0]),
        )

        assert matched_loss.ndim == 0
        assert matched_loss < mismatched_loss

    def test_matches_reference_softplus_formula(self):
        """Loss should match the positive/negative softplus decomposition."""
        feats_i = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        feats_j = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        loss_fn = SigLIPLoss(
            init_logit_scale=math.log(2.0),
            init_logit_bias=-0.5,
            normalize=False,
        )

        loss = loss_fn(image_features=feats_i, text_features=feats_j)

        logits = 2.0 * (feats_i @ feats_j.T) - 0.5
        positive_logits = logits.diagonal()
        negative_mask = ~torch.eye(logits.size(0), dtype=torch.bool)
        negative_logits = logits[negative_mask]
        expected = (
            F.softplus(-positive_logits).sum() + F.softplus(negative_logits).sum()
        ) / feats_i.size(0)
        assert torch.allclose(loss, expected, atol=1e-7, rtol=0)

    def test_invariance_to_feature_magnitude(self):
        """Loss should be identical regardless of input vector magnitude."""
        torch.manual_seed(123)
        batch_size, dim = 8, 256
        feats_i = torch.randn(batch_size, dim)
        feats_j = torch.randn(batch_size, dim)
        loss_fn = SigLIPLoss()

        loss1 = loss_fn(image_features=feats_i, text_features=feats_j)
        loss2 = loss_fn(
            image_features=feats_i * 100.0,
            text_features=feats_j * 0.01,
        )

        assert torch.allclose(loss1, loss2, atol=1e-7, rtol=1e-6)

    def test_logit_scale_and_bias_receive_gradients(self):
        """Learnable logit scale and bias should receive gradients."""
        torch.manual_seed(42)
        batch_size, dim = 4, 128
        feats_i = torch.randn(batch_size, dim, requires_grad=True)
        feats_j = torch.randn(batch_size, dim, requires_grad=True)
        loss_fn = SigLIPLoss()

        loss = loss_fn(image_features=feats_i, text_features=feats_j)
        loss.backward()

        assert loss_fn.logit_scale.grad is not None
        assert loss_fn.logit_bias.grad is not None
        assert loss_fn.logit_scale.grad.abs().item() > 0
        assert loss_fn.logit_bias.grad.abs().item() > 0

    def test_raises_for_unpaired_batches(self):
        """Loss should reject unequal image/text batch sizes."""
        loss_fn = SigLIPLoss()
        feats_i = torch.randn(2, 8)
        feats_j = torch.randn(3, 8)

        with pytest.raises(ValueError, match="same batch size"):
            loss_fn(image_features=feats_i, text_features=feats_j)
