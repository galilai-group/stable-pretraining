"""Unit tests for MultiBlockMasking strategy."""

import pytest
import torch

from stable_pretraining.backbone import MultiBlockMasking, MaskedEncoder
from stable_pretraining.backbone.patch_masking import MaskingOutput


pytestmark = [pytest.mark.unit, pytest.mark.cpu]


class TestInit:
    """Tests for MultiBlockMasking initialization."""

    def test_default_init(self):
        masking = MultiBlockMasking()
        assert masking.num_targets == 4
        assert masking.context_scale == (0.85, 1.0)
        assert masking.target_scale == (0.15, 0.2)
        assert masking.context_aspect_ratio == (1.0, 1.0)
        assert masking.target_aspect_ratio == (0.75, 1.5)

    def test_custom_params(self):
        masking = MultiBlockMasking(
            num_targets=2,
            context_scale=(0.7, 0.9),
            target_scale=(0.1, 0.15),
            context_aspect_ratio=(0.8, 1.2),
            target_aspect_ratio=(0.5, 2.0),
        )
        assert masking.num_targets == 2
        assert masking.context_scale == (0.7, 0.9)

    def test_invalid_num_targets(self):
        with pytest.raises(ValueError, match="num_targets"):
            MultiBlockMasking(num_targets=0)


class TestOutputShapes:
    """Test output shapes for various grid sizes."""

    @pytest.fixture
    def masking(self):
        return MultiBlockMasking(num_targets=4)

    @pytest.mark.parametrize(
        "grid_h,grid_w",
        [(14, 14), (7, 7), (8, 16)],
    )
    def test_output_shapes(self, masking, grid_h, grid_w):
        B, N, D = 4, grid_h * grid_w, 192
        x = torch.randn(B, N, D)
        masking.train()
        output = masking(x, grid_h, grid_w)

        assert isinstance(output, MaskingOutput)
        assert output.mask.shape == (B, N)
        assert output.ids_restore.shape == (B, N)
        # visible and ids_keep should match
        N_keep = output.ids_keep.shape[1]
        assert output.visible.shape == (B, N_keep, D)
        assert N_keep > 0
        assert N_keep < N

    def test_output_type(self, masking):
        x = torch.randn(4, 196, 192)
        masking.train()
        output = masking(x, 14, 14)
        assert isinstance(output, MaskingOutput)


class TestOutputCorrectness:
    """Test correctness of mask values and indices."""

    @pytest.fixture
    def masking(self):
        return MultiBlockMasking(num_targets=4)

    def test_binary_mask_values(self, masking):
        """Mask should be binary (0 or 1)."""
        x = torch.randn(4, 196, 192)
        masking.train()
        output = masking(x, 14, 14)
        assert ((output.mask == 0) | (output.mask == 1)).all()

    def test_context_target_disjoint(self, masking):
        """Context (visible) patches should be disjoint from target patches."""
        x = torch.randn(4, 196, 192)
        masking.train()

        for _ in range(10):
            output = masking(x, 14, 14)
            # Visible patches = where mask is 0
            visible_mask = output.mask[0] == 0
            visible_indices = set(visible_mask.nonzero(as_tuple=True)[0].tolist())
            keep_indices = set(output.ids_keep[0].tolist())
            assert visible_indices == keep_indices

    def test_ids_keep_consistency(self, masking):
        """ids_keep should index into the correct visible patches."""
        x = torch.randn(2, 196, 64)
        masking.train()
        output = masking(x, 14, 14)

        for b in range(2):
            gathered = torch.gather(
                x[b : b + 1],
                1,
                output.ids_keep[b : b + 1].unsqueeze(-1).expand(-1, -1, 64),
            )
            assert torch.allclose(gathered, output.visible[b : b + 1])

    def test_visible_patches_match(self, masking):
        """Visible patches should match gathered input."""
        x = torch.randn(4, 196, 192)
        masking.train()
        output = masking(x, 14, 14)

        gathered = torch.gather(
            x, dim=1, index=output.ids_keep.unsqueeze(-1).expand(-1, -1, 192)
        )
        assert torch.allclose(gathered, output.visible)


class TestEvalMode:
    """Test eval mode behavior."""

    def test_eval_no_masking(self):
        masking = MultiBlockMasking(num_targets=4)
        masking.eval()

        B, N, D = 4, 196, 192
        x = torch.randn(B, N, D)
        output = masking(x, 14, 14)

        assert torch.allclose(output.visible, x)
        assert (output.mask == 0).all()
        assert output.ids_keep.shape == (B, N)
        assert output.ids_restore.shape == (B, N)

    def test_train_has_masking(self):
        masking = MultiBlockMasking(num_targets=4)
        masking.train()

        x = torch.randn(4, 196, 192)
        output = masking(x, 14, 14)

        assert output.mask.sum() > 0
        assert output.ids_keep.shape[1] < 196


class TestGradients:
    """Test gradient flow through visible patches."""

    def test_gradient_flow(self):
        masking = MultiBlockMasking(num_targets=4)
        masking.train()

        x = torch.randn(2, 196, 64, requires_grad=True)
        output = masking(x, 14, 14)
        loss = output.visible.sum()
        loss.backward()

        assert x.grad is not None
        # Gradients should only flow through visible patches
        visible_mask = output.mask[0] == 0
        assert (x.grad[0][visible_mask] != 0).any()


class TestMaskedEncoderIntegration:
    """Test integration with MaskedEncoder."""

    def test_masked_encoder_works(self):
        masking = MultiBlockMasking(num_targets=4)
        encoder = MaskedEncoder("vit_tiny_patch16_224", masking=masking)
        encoder.train()

        images = torch.randn(2, 3, 224, 224)
        output = encoder(images)

        # Should have fewer patches than full
        num_prefix = encoder.num_prefix_tokens
        num_patches = output.encoded.shape[1] - num_prefix
        assert num_patches < 196
        assert output.mask.shape == (2, 196)

    def test_masked_encoder_eval(self):
        masking = MultiBlockMasking(num_targets=4)
        encoder = MaskedEncoder("vit_tiny_patch16_224", masking=masking)
        encoder.eval()

        images = torch.randn(2, 3, 224, 224)
        output = encoder(images)

        # In eval, all patches should be encoded
        num_prefix = encoder.num_prefix_tokens
        num_patches = output.encoded.shape[1] - num_prefix
        assert num_patches == 196
        assert (output.mask == 0).all()


class TestInputValidation:
    """Test input validation."""

    def test_wrong_dims(self):
        masking = MultiBlockMasking()
        with pytest.raises(ValueError, match="Expected 3D"):
            masking(torch.randn(4, 196), 14, 14)

    def test_mismatched_grid(self):
        masking = MultiBlockMasking()
        with pytest.raises(ValueError, match="doesn't match grid"):
            masking(torch.randn(4, 196, 64), 10, 10)
