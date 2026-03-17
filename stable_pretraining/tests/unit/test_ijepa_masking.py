"""Unit tests for I-JEPA masking strategy."""

import pytest
import torch

from stable_pretraining.backbone import IJEPAMasking, IJEPAMaskOutput


# Mark all tests in this module as unit tests
pytestmark = [pytest.mark.unit, pytest.mark.cpu]


class TestIJEPAMaskingOutputShape:
    """Test output shapes and types."""

    @pytest.fixture
    def masking(self):
        return IJEPAMasking(
            num_targets=4,
            target_scale=(0.15, 0.2),
            target_aspect_ratio=(0.75, 1.5),
            context_scale=(0.85, 1.0),
        )

    @pytest.fixture
    def default_input(self):
        B, H, W, D = 4, 14, 14, 768
        x = torch.randn(B, H * W, D)
        return x, H, W

    def test_output_type(self, masking, default_input):
        """Output should be IJEPAMaskOutput dataclass."""
        x, H, W = default_input
        masking.train()
        output = masking(x, H, W)
        assert isinstance(output, IJEPAMaskOutput)

    def test_context_idx_shape(self, masking, default_input):
        """context_idx should be [B, N_ctx]."""
        x, H, W = default_input
        B, N, D = x.shape
        masking.train()
        output = masking(x, H, W)

        assert output.context_idx.dim() == 2
        assert output.context_idx.shape[0] == B
        assert output.context_idx.shape[1] > 0
        assert output.context_idx.shape[1] < N

    def test_target_idx_shape(self, masking, default_input):
        """target_idx should be [B, N_tgt]."""
        x, H, W = default_input
        B, N, D = x.shape
        masking.train()
        output = masking(x, H, W)

        assert output.target_idx.dim() == 2
        assert output.target_idx.shape[0] == B
        assert output.target_idx.shape[1] > 0
        assert output.target_idx.shape[1] < N

    def test_mask_shape(self, masking, default_input):
        """Mask should be [B, N]."""
        x, H, W = default_input
        B, N, D = x.shape
        masking.train()
        output = masking(x, H, W)

        assert output.mask.shape == (B, N)

    def test_target_block_masks_count(self, masking, default_input):
        """Should always have exactly num_targets block masks."""
        x, H, W = default_input
        masking.train()
        output = masking(x, H, W)

        assert len(output.target_block_masks) == masking.num_targets

    def test_some_block_masks_may_be_empty(self):
        """With large blocks, some masks may be empty (all False) due to overlap prevention."""
        masking = IJEPAMasking(
            num_targets=6,
            target_scale=(0.20, 0.25),  # Large blocks, likely won't all fit
            allow_target_overlap=False,
        )
        masking.train()

        x = torch.randn(2, 196, 768)
        output = masking(x, 14, 14)

        # Always get 6 masks
        assert len(output.target_block_masks) == 6

        # But some may be empty
        non_empty = sum(1 for m in output.target_block_masks if m.any())
        assert non_empty >= 1  # At least one block
        assert non_empty <= 6  # At most all blocks


class TestIJEPAMaskingConstraints:
    """Test that masking satisfies I-JEPA constraints."""

    @pytest.fixture
    def masking(self):
        return IJEPAMasking(
            num_targets=4,
            target_scale=(0.15, 0.2),
            target_aspect_ratio=(0.75, 1.5),
            context_scale=(0.85, 1.0),
            allow_target_overlap=False,
        )

    def test_context_target_no_overlap(self, masking):
        """Context and target indices should not overlap."""
        B, H, W, D = 4, 14, 14, 768
        x = torch.randn(B, H * W, D)
        masking.train()
        output = masking(x, H, W)

        for b in range(B):
            context_set = set(output.context_idx[b].tolist())
            target_set = set(output.target_idx[b].tolist())
            overlap = context_set & target_set
            assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_context_target_cover_subset(self, masking):
        """Context + target should be a subset of all patches."""
        B, H, W, D = 4, 14, 14, 768
        N = H * W
        x = torch.randn(B, N, D)
        masking.train()
        output = masking(x, H, W)

        for b in range(B):
            context_set = set(output.context_idx[b].tolist())
            target_set = set(output.target_idx[b].tolist())
            all_indices = context_set | target_set
            assert all_indices.issubset(set(range(N)))

    def test_mask_matches_target_idx(self, masking):
        """Mask should be 1 exactly where target_idx points."""
        B, H, W, D = 4, 14, 14, 768
        x = torch.randn(B, H * W, D)
        masking.train()
        output = masking(x, H, W)

        # Positions in target_idx should have mask == 1
        for b in range(B):
            for idx in output.target_idx[b]:
                assert output.mask[b, idx] == 1.0

    def test_block_masks_union_equals_target(self, masking):
        """Union of all block masks should equal combined target mask."""
        B, H, W, D = 4, 14, 14, 768
        x = torch.randn(B, H * W, D)
        masking.train()
        output = masking(x, H, W)

        # Combine all block masks with OR
        combined = torch.zeros_like(output.target_block_masks[0])
        for block_mask in output.target_block_masks:
            combined = combined | block_mask

        # Should match the overall mask
        assert torch.equal(combined.float(), output.mask)

    def test_indices_are_valid(self, masking):
        """All indices should be valid patch positions."""
        B, H, W, D = 4, 14, 14, 768
        N = H * W
        x = torch.randn(B, N, D)
        masking.train()
        output = masking(x, H, W)

        assert (output.context_idx >= 0).all()
        assert (output.context_idx < N).all()
        assert (output.target_idx >= 0).all()
        assert (output.target_idx < N).all()

    def test_indices_are_sorted(self, masking):
        """Indices should be sorted for consistency."""
        B, H, W, D = 4, 14, 14, 768
        x = torch.randn(B, H * W, D)
        masking.train()
        output = masking(x, H, W)

        # Target idx should be sorted
        for b in range(B):
            target = output.target_idx[b]
            assert torch.equal(target, target.sort().values)


class TestIJEPAMaskingScale:
    """Test that scale constraints are respected."""

    def test_target_scale_bounds(self):
        """Non-empty target blocks should respect scale bounds."""
        target_scale = (0.10, 0.15)
        masking = IJEPAMasking(
            num_targets=4,
            target_scale=target_scale,
            target_aspect_ratio=(1.0, 1.0),  # Square for easier calculation
            context_scale=(1.0, 1.0),
        )

        B, H, W, D = 8, 14, 14, 768
        N = H * W
        x = torch.randn(B, N, D)
        masking.train()

        # Run multiple times to check bounds statistically
        min_scale_seen = 1.0
        max_scale_seen = 0.0
        non_empty_count = 0

        for _ in range(50):
            output = masking(x, H, W)
            for block_mask in output.target_block_masks:
                # Only check non-empty masks
                block_size = block_mask[0].sum().item()
                if block_size > 0:  # Skip empty masks
                    scale = block_size / N
                    min_scale_seen = min(min_scale_seen, scale)
                    max_scale_seen = max(max_scale_seen, scale)
                    non_empty_count += 1

        # Should have seen some non-empty blocks
        assert non_empty_count > 0, "No non-empty blocks were sampled"

        # Allow some tolerance for discretization
        assert min_scale_seen >= target_scale[0] * 0.5, (
            f"Min scale {min_scale_seen} too small"
        )
        assert max_scale_seen <= target_scale[1] * 2.0, (
            f"Max scale {max_scale_seen} too large"
        )

    def test_context_scale_bounds(self):
        """Context should respect scale bounds relative to available patches."""
        context_scale = (0.8, 0.9)
        masking = IJEPAMasking(
            num_targets=2,
            target_scale=(0.05, 0.08),  # Small blocks to ensure they fit
            target_aspect_ratio=(1.0, 1.0),
            context_scale=context_scale,
        )

        B, H, W, D = 8, 14, 14, 768
        N = H * W
        x = torch.randn(B, N, D)
        masking.train()

        for _ in range(20):
            output = masking(x, H, W)
            n_target = output.target_idx.shape[1]
            n_context = output.context_idx.shape[1]
            n_available = N - n_target

            if n_available > 0:
                actual_ratio = n_context / n_available
                # Allow tolerance for rounding
                assert actual_ratio >= context_scale[0] * 0.8
                assert actual_ratio <= min(context_scale[1] * 1.2, 1.0)


class TestIJEPAMaskingEvalMode:
    """Test behavior in eval mode."""

    @pytest.fixture
    def masking(self):
        return IJEPAMasking(num_targets=4)

    def test_eval_no_masking(self, masking):
        """In eval mode, everything should be context, block masks are empty."""
        B, H, W, D = 4, 14, 14, 768
        N = H * W
        x = torch.randn(B, N, D)

        masking.eval()
        output = masking(x, H, W)

        # All patches should be context
        assert output.context_idx.shape == (B, N)
        # No targets
        assert output.target_idx.shape[1] == 0
        # Still returns num_targets masks, but all empty
        assert len(output.target_block_masks) == masking.num_targets
        for block_mask in output.target_block_masks:
            assert not block_mask.any(), "Block masks should be empty in eval mode"
        # Mask should be all zeros
        assert (output.mask == 0).all()

    def test_train_has_masking(self, masking):
        """In train mode, should have targets."""
        B, H, W, D = 4, 14, 14, 768
        N = H * W
        x = torch.randn(B, N, D)

        masking.train()
        output = masking(x, H, W)

        assert output.context_idx.shape[1] < N
        assert output.target_idx.shape[1] > 0
        # At least one block mask should be non-empty
        assert any(m.any() for m in output.target_block_masks)


class TestIJEPAMaskingEdgeCases:
    """Test edge cases and error handling."""

    def test_small_grid(self):
        """Should work with small grids."""
        masking = IJEPAMasking(
            num_targets=2,
            target_scale=(0.2, 0.3),
            context_scale=(0.5, 1.0),
        )

        B, H, W, D = 2, 4, 4, 64
        x = torch.randn(B, H * W, D)
        masking.train()

        output = masking(x, H, W)
        assert output.context_idx.shape[1] > 0
        assert output.target_idx.shape[1] > 0

    def test_single_target(self):
        """Should work with single target block."""
        masking = IJEPAMasking(num_targets=1)

        B, H, W, D = 2, 14, 14, 768
        x = torch.randn(B, H * W, D)
        masking.train()

        output = masking(x, H, W)
        assert len(output.target_block_masks) == 1

    def test_many_targets(self):
        """Should work with many target blocks."""
        masking = IJEPAMasking(
            num_targets=8,
            target_scale=(0.05, 0.08),  # Smaller to fit more
        )

        B, H, W, D = 2, 14, 14, 768
        x = torch.randn(B, H * W, D)
        masking.train()

        output = masking(x, H, W)
        assert len(output.target_block_masks) == 8

    def test_non_square_grid(self):
        """Should work with non-square grids."""
        masking = IJEPAMasking(num_targets=4)

        B, H, W, D = 2, 8, 16, 768
        x = torch.randn(B, H * W, D)
        masking.train()

        output = masking(x, H, W)
        assert output.context_idx.shape[0] == B
        assert output.target_idx.shape[0] == B

    def test_wrong_input_dim_raises(self):
        """Should raise error for wrong input dimensions."""
        masking = IJEPAMasking()

        x_2d = torch.randn(4, 196)
        with pytest.raises(ValueError, match="Expected 3D input"):
            masking(x_2d, 14, 14)

        x_4d = torch.randn(4, 196, 768, 1)
        with pytest.raises(ValueError, match="Expected 3D input"):
            masking(x_4d, 14, 14)

    def test_mismatched_grid_raises(self):
        """Should raise error when N != grid_h * grid_w."""
        masking = IJEPAMasking()

        x = torch.randn(4, 196, 768)
        with pytest.raises(ValueError, match="doesn't match grid"):
            masking(x, 10, 10)  # 10*10=100 != 196


class TestIJEPAMaskingValidation:
    """Test input validation."""

    def test_invalid_num_targets(self):
        with pytest.raises(ValueError, match="num_targets"):
            IJEPAMasking(num_targets=0)

    def test_invalid_target_scale(self):
        with pytest.raises(ValueError, match="target_scale"):
            IJEPAMasking(target_scale=(0.5, 0.3))  # min > max

        with pytest.raises(ValueError, match="target_scale"):
            IJEPAMasking(target_scale=(-0.1, 0.3))  # negative

        with pytest.raises(ValueError, match="target_scale"):
            IJEPAMasking(target_scale=(0.5, 1.5))  # > 1

    def test_invalid_aspect_ratio(self):
        with pytest.raises(ValueError, match="target_aspect_ratio"):
            IJEPAMasking(target_aspect_ratio=(2.0, 1.0))  # min > max

        with pytest.raises(ValueError, match="target_aspect_ratio"):
            IJEPAMasking(target_aspect_ratio=(-1.0, 1.0))  # negative

    def test_invalid_context_scale(self):
        with pytest.raises(ValueError, match="context_scale"):
            IJEPAMasking(context_scale=(0.9, 0.5))  # min > max

        with pytest.raises(ValueError, match="context_scale"):
            IJEPAMasking(context_scale=(0.0, 0.5))  # zero


class TestIJEPAMaskingDeterminism:
    """Test reproducibility with seeds."""

    def test_different_without_seed(self):
        """Without seed, results should vary."""
        masking = IJEPAMasking(num_targets=4)
        masking.train()

        x = torch.randn(2, 196, 768)

        output1 = masking(x, 14, 14)
        output2 = masking(x, 14, 14)

        # Very unlikely to be identical
        # (could theoretically fail, but extremely rare)
        assert not torch.equal(output1.context_idx, output2.context_idx)

    def test_reproducible_with_seed(self):
        """With same seed, results should be identical."""
        masking = IJEPAMasking(num_targets=4)
        masking.train()

        x = torch.randn(2, 196, 768)

        torch.manual_seed(42)
        output1 = masking(x, 14, 14)

        torch.manual_seed(42)
        output2 = masking(x, 14, 14)

        assert torch.equal(output1.context_idx, output2.context_idx)
        assert torch.equal(output1.target_idx, output2.target_idx)
        assert torch.equal(output1.mask, output2.mask)


class TestIJEPAMaskingOverlap:
    """Test target block overlap behavior."""

    def test_no_overlap_by_default(self):
        """By default, target blocks should not overlap."""
        masking = IJEPAMasking(
            num_targets=4,
            target_scale=(0.15, 0.2),
            allow_target_overlap=False,
        )
        masking.train()

        x = torch.randn(4, 196, 768)

        for _ in range(10):
            output = masking(x, 14, 14)

            # Check pairwise: each patch in at most one block
            total_per_patch = torch.zeros(196)
            for block_mask in output.target_block_masks:
                total_per_patch += block_mask[0].float()

            # No patch should be in more than one block
            assert (total_per_patch <= 1).all()

    def test_overlap_allowed(self):
        """When allow_target_overlap=True, overlap is permitted."""
        masking = IJEPAMasking(
            num_targets=4,
            target_scale=(0.25, 0.3),  # Large blocks to encourage overlap
            allow_target_overlap=True,
        )
        masking.train()

        x = torch.randn(4, 196, 768)

        # Run many times - with large blocks, overlap should occur sometimes
        overlap_seen = False
        for _ in range(50):
            output = masking(x, 14, 14)

            total_per_patch = torch.zeros(196)
            for block_mask in output.target_block_masks:
                total_per_patch += block_mask[0].float()

            if (total_per_patch > 1).any():
                overlap_seen = True
                break

        # With large blocks, we should see overlap
        assert overlap_seen, "Expected overlap with large blocks"
