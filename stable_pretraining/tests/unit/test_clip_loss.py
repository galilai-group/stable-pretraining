"""Single-process unit tests for :class:`CLIPLoss`.

These cover the loss math (perfect match, mismatch, magnitude invariance,
logit-scale override, symmetry) in a single process, where the cross-GPU gather
is a no-op so the result is the plain diagonal-target InfoNCE. The multi-process
correctness of the gathered-negatives path (each rank's anchors scored against
candidates from all ranks, with rank-offset targets) is verified in
``tests/distributed/test_collectives_dist.py``.
"""

import pytest
import torch
import torch.nn.functional as F

from stable_pretraining.losses import CLIPLoss


@pytest.mark.unit
class TestCLIPLoss:
    """Unit tests for the CLIPLoss function (single process)."""

    def test_loss_is_low_for_perfect_match(self):
        """Loss should be near-zero when feats_i == feats_j and features are orthonormal."""
        torch.manual_seed(0)
        batch_size, dim = 4, 8
        # orthonormal rows so off-diagonal similarities are exactly 0
        feats = F.normalize(torch.eye(dim)[:batch_size], dim=-1)

        loss = CLIPLoss()(feats_i=feats, feats_j=feats)

        assert loss.ndim == 0
        assert loss.item() < 1e-5

    def test_loss_is_high_for_mismatch(self):
        """Loss should be high when positive pairs are swapped/orthogonal."""
        feats_i = torch.eye(2)
        feats_j = torch.flip(torch.eye(2), dims=[1])

        loss = CLIPLoss()(feats_i=feats_i, feats_j=feats_j)

        s = 1.0 / 0.07
        expected_loss = -F.log_softmax(torch.tensor([0.0, s]), dim=0)[0]
        assert torch.allclose(loss, expected_loss, atol=1e-7, rtol=0)

    def test_invariance_to_feature_magnitude(self):
        """Loss should be identical regardless of input vector magnitude."""
        torch.manual_seed(123)
        feats_i = torch.randn(8, 256)
        feats_j = torch.randn(8, 256)

        loss1 = CLIPLoss()(feats_i=feats_i, feats_j=feats_j)
        loss2 = CLIPLoss()(feats_i=feats_i * 100.0, feats_j=feats_j * 100.0)

        # normalization cancels magnitude
        assert torch.allclose(loss1, loss2, atol=1e-7, rtol=1e-6)

    def test_logit_scale_overrides_temperature(self):
        """A provided logit_scale should be used instead of temperature."""
        torch.manual_seed(42)
        feats_i = F.normalize(torch.randn(4, 128), dim=-1)
        feats_j = F.normalize(torch.randn(4, 128), dim=-1)
        loss_fn = CLIPLoss(temperature=0.01)

        loss_temp = loss_fn(feats_i=feats_i, feats_j=feats_j, logit_scale=None)
        loss_float = loss_fn(feats_i=feats_i, feats_j=feats_j, logit_scale=20.0)
        loss_tensor = loss_fn(
            feats_i=feats_i,
            feats_j=feats_j,
            logit_scale=torch.tensor(20.0, requires_grad=True),
        )

        assert not torch.allclose(loss_temp, loss_float)
        assert torch.allclose(loss_float, loss_tensor, atol=1e-7, rtol=0)

    def test_symmetry_image_text(self):
        """Loss(img, txt) ~= Loss(txt, img) within numerical tolerance."""
        torch.manual_seed(7)
        x = F.normalize(torch.randn(5, 64), dim=-1)
        y = F.normalize(torch.randn(5, 64), dim=-1)
        loss_fn = CLIPLoss()

        a = loss_fn(x, y)
        b = loss_fn(y, x)

        assert torch.allclose(a, b, atol=1e-5, rtol=1e-6), f"{a=} {b=}"
