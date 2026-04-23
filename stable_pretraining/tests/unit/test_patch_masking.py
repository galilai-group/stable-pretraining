import pytest
import torch
from PIL import Image
import numpy as np

# Assume TPatchMasking is defined in patch_masking.py
from stable_pretraining.data.transforms import PatchMasking as TPatchMasking
from stable_pretraining.backbone import PatchMasking
from stable_pretraining.backbone.patch_masking import MaskingOutput

pytestmark = pytest.mark.unit

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_masking():
    """Default masking module with 75% mask ratio."""
    return PatchMasking(mask_ratio=0.75)


@pytest.fixture
def block_masking():
    """Block masking with 4x4 blocks."""
    return PatchMasking(mask_ratio=0.75, block_size=4)


@pytest.fixture
def crop_masking():
    """Crop masking with 100% crop probability."""
    return PatchMasking(mask_ratio=0.75, crop_ratio=1.0)


@pytest.fixture
def sample_input():
    """Sample input tensor (B=4, N=196, D=768) for 14x14 grid."""
    torch.manual_seed(42)
    return torch.randn(4, 196, 768)


@pytest.fixture
def small_input():
    """Small input tensor (B=2, N=16, D=64) for 4x4 grid."""
    torch.manual_seed(42)
    return torch.randn(2, 16, 64)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInit:
    """Tests for PatchMasking initialization and validation."""

    def test_default_init(self):
        """Test default initialization."""
        masking = PatchMasking()
        assert masking.mask_ratio == 0.75
        assert masking.block_size == 1
        assert masking.crop_ratio == 0.0
        assert masking.crop_aspect_ratio == (0.75, 1.33)

    def test_custom_init(self):
        """Test custom initialization."""
        masking = PatchMasking(
            mask_ratio=0.5,
            block_size=2,
            crop_ratio=0.3,
            crop_aspect_ratio=(0.5, 2.0),
        )
        assert masking.mask_ratio == 0.5
        assert masking.block_size == 2
        assert masking.crop_ratio == 0.3
        assert masking.crop_aspect_ratio == (0.5, 2.0)

    @pytest.mark.parametrize("mask_ratio", [-0.1, 1.0, 1.5])
    def test_invalid_mask_ratio(self, mask_ratio):
        """Test that invalid mask_ratio raises ValueError."""
        with pytest.raises(ValueError, match="mask_ratio must be in"):
            PatchMasking(mask_ratio=mask_ratio)

    @pytest.mark.parametrize("block_size", [0, -1, -10])
    def test_invalid_block_size(self, block_size):
        """Test that invalid block_size raises ValueError."""
        with pytest.raises(ValueError, match="block_size must be >= 1"):
            PatchMasking(block_size=block_size)

    @pytest.mark.parametrize("crop_ratio", [-0.1, 1.1, 2.0])
    def test_invalid_crop_ratio(self, crop_ratio):
        """Test that invalid crop_ratio raises ValueError."""
        with pytest.raises(ValueError, match="crop_ratio must be in"):
            PatchMasking(crop_ratio=crop_ratio)

    def test_invalid_crop_aspect_ratio_order(self):
        """Test that reversed crop_aspect_ratio raises ValueError."""
        with pytest.raises(ValueError, match="crop_aspect_ratio\\[0\\] must be <="):
            PatchMasking(crop_aspect_ratio=(2.0, 0.5))

    def test_invalid_crop_aspect_ratio_negative(self):
        """Test that negative crop_aspect_ratio raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            PatchMasking(crop_aspect_ratio=(-1.0, 1.0))

    def test_invalid_crop_aspect_ratio_length(self):
        """Test that wrong length crop_aspect_ratio raises ValueError."""
        with pytest.raises(ValueError, match="must be a tuple of 2"):
            PatchMasking(crop_aspect_ratio=(0.5, 1.0, 1.5))

    def test_extra_repr(self, default_masking):
        """Test extra_repr for debugging."""
        repr_str = default_masking.extra_repr()
        assert "mask_ratio=0.75" in repr_str
        assert "block_size=1" in repr_str


# =============================================================================
# Forward Validation Tests
# =============================================================================


class TestForwardValidation:
    """Tests for forward pass input validation."""

    def test_wrong_input_dim_2d(self, default_masking):
        """Test that 2D input raises ValueError."""
        x = torch.randn(16, 768)
        with pytest.raises(ValueError, match="Expected 3D input"):
            default_masking(x, grid_h=4, grid_w=4)

    def test_wrong_input_dim_4d(self, default_masking):
        """Test that 4D input raises ValueError."""
        x = torch.randn(2, 4, 4, 768)
        with pytest.raises(ValueError, match="Expected 3D input"):
            default_masking(x, grid_h=4, grid_w=4)

    def test_mismatched_grid_size(self, default_masking):
        """Test that mismatched N and grid size raises ValueError."""
        x = torch.randn(2, 16, 64)  # N=16
        with pytest.raises(ValueError, match="doesn't match grid size"):
            default_masking(x, grid_h=5, grid_w=5)  # 5*5=25 != 16


# =============================================================================
# Output Shape Tests
# =============================================================================


class TestOutputShapes:
    """Tests for correct output shapes."""

    def test_output_type(self, default_masking, small_input):
        """Test that output is MaskingOutput dataclass."""
        output = default_masking(small_input, grid_h=4, grid_w=4)
        assert isinstance(output, MaskingOutput)

    def test_visible_shape(self, default_masking, small_input):
        """Test visible patches shape."""
        output = default_masking(small_input, grid_h=4, grid_w=4)
        B, N, D = small_input.shape
        num_keep = N - int(N * default_masking.mask_ratio)
        assert output.visible.shape == (B, num_keep, D)

    def test_mask_shape(self, default_masking, small_input):
        """Test mask shape."""
        output = default_masking(small_input, grid_h=4, grid_w=4)
        B, N, _ = small_input.shape
        assert output.mask.shape == (B, N)

    def test_ids_restore_shape(self, default_masking, small_input):
        """Test ids_restore shape."""
        output = default_masking(small_input, grid_h=4, grid_w=4)
        B, N, _ = small_input.shape
        assert output.ids_restore.shape == (B, N)

    def test_ids_keep_shape(self, default_masking, small_input):
        """Test ids_keep shape."""
        output = default_masking(small_input, grid_h=4, grid_w=4)
        B, N, _ = small_input.shape
        num_keep = N - int(N * default_masking.mask_ratio)
        assert output.ids_keep.shape == (B, num_keep)

    @pytest.mark.parametrize("grid_h,grid_w", [(4, 4), (7, 7), (4, 8), (14, 14)])
    def test_various_grid_sizes(self, grid_h, grid_w):
        """Test with various grid sizes."""
        masking = PatchMasking(mask_ratio=0.5)
        N = grid_h * grid_w
        x = torch.randn(2, N, 64)
        output = masking(x, grid_h=grid_h, grid_w=grid_w)

        num_keep = N - int(N * 0.5)
        assert output.visible.shape == (2, num_keep, 64)
        assert output.mask.shape == (2, N)


# =============================================================================
# Output Correctness Tests
# =============================================================================


class TestOutputCorrectness:
    """Tests for correct masking behavior."""

    def test_mask_values_binary(self, default_masking, small_input):
        """Test that mask contains only 0s and 1s."""
        output = default_masking(small_input, grid_h=4, grid_w=4)
        unique_values = torch.unique(output.mask)
        assert torch.allclose(unique_values, torch.tensor([0.0, 1.0]))

    def test_mask_ratio_per_sample(self, default_masking, sample_input):
        """Test that each sample has correct mask ratio."""
        output = default_masking(sample_input, grid_h=14, grid_w=14)
        B, N, _ = sample_input.shape
        expected_masked = int(N * default_masking.mask_ratio)

        for i in range(B):
            actual_masked = output.mask[i].sum().item()
            assert actual_masked == expected_masked

    def test_visible_matches_gathered(self, default_masking, small_input):
        """Test that visible patches match gathering by ids_keep."""
        output = default_masking(small_input, grid_h=4, grid_w=4)

        # Manually gather using ids_keep
        expected_visible = torch.gather(
            small_input,
            dim=1,
            index=output.ids_keep.unsqueeze(-1).expand(-1, -1, small_input.shape[-1]),
        )
        assert torch.allclose(output.visible, expected_visible)

    def test_ids_restore_inverts_shuffle(self, default_masking, small_input):
        """Test that ids_restore can restore original order."""
        output = default_masking(small_input, grid_h=4, grid_w=4)
        B, N, _ = small_input.shape

        for i in range(B):
            shuffled_to_original = torch.argsort(output.ids_restore[i])
            restored = shuffled_to_original[output.ids_restore[i]]
            assert torch.equal(restored, torch.arange(N))

    def test_mask_matches_ids_keep(self, default_masking, small_input):
        """Test that mask=0 positions match ids_keep."""
        output = default_masking(small_input, grid_h=4, grid_w=4)
        B = small_input.shape[0]

        for i in range(B):
            visible_positions = (output.mask[i] == 0).nonzero(as_tuple=True)[0]
            sorted_ids_keep = output.ids_keep[i].sort()[0]
            sorted_visible = visible_positions.sort()[0]
            assert torch.equal(sorted_ids_keep, sorted_visible)

    def test_ids_keep_unique(self, default_masking, sample_input):
        """Test that ids_keep contains unique indices."""
        output = default_masking(sample_input, grid_h=14, grid_w=14)
        B = sample_input.shape[0]

        for i in range(B):
            unique_ids = torch.unique(output.ids_keep[i])
            assert len(unique_ids) == len(output.ids_keep[i])


# =============================================================================
# Zero Mask Ratio Tests
# =============================================================================


class TestZeroMaskRatio:
    """Tests for mask_ratio=0 edge case."""

    def test_zero_mask_ratio_all_visible(self):
        """Test that mask_ratio=0 keeps all patches visible."""
        masking = PatchMasking(mask_ratio=0.0)
        x = torch.randn(2, 16, 64)
        output = masking(x, grid_h=4, grid_w=4)

        assert output.visible.shape == x.shape
        assert torch.allclose(output.visible, x)

    def test_zero_mask_ratio_mask_all_zeros(self):
        """Test that mask_ratio=0 produces all-zero mask."""
        masking = PatchMasking(mask_ratio=0.0)
        x = torch.randn(2, 16, 64)
        output = masking(x, grid_h=4, grid_w=4)

        assert torch.all(output.mask == 0)

    def test_zero_mask_ratio_ids_sequential(self):
        """Test that mask_ratio=0 produces sequential ids."""
        masking = PatchMasking(mask_ratio=0.0)
        x = torch.randn(2, 16, 64)
        output = masking(x, grid_h=4, grid_w=4)

        expected_ids = torch.arange(16).unsqueeze(0).expand(2, -1)
        assert torch.equal(output.ids_keep, expected_ids)
        assert torch.equal(output.ids_restore, expected_ids)


# =============================================================================
# Strategy-Specific Tests
# =============================================================================


class TestRandomMasking:
    """Tests specific to random masking strategy."""

    def test_random_different_samples(self):
        """Test that different samples get different masks (probabilistic)."""
        masking = PatchMasking(mask_ratio=0.5, block_size=1, crop_ratio=0.0)
        x = torch.randn(10, 64, 32)
        output = masking(x, grid_h=8, grid_w=8)

        # Check that not all masks are identical (very unlikely with random)
        masks_equal = [
            torch.equal(output.mask[0], output.mask[i]) for i in range(1, 10)
        ]
        assert not all(masks_equal), "All masks are identical, which is very unlikely"


class TestBlockMasking:
    """Tests specific to block masking strategy."""

    def test_block_masking_exact_count(self, block_masking):
        """Test that block masking produces exact mask count."""
        x = torch.randn(4, 196, 64)
        output = block_masking(x, grid_h=14, grid_w=14)

        expected_masked = int(196 * block_masking.mask_ratio)
        for i in range(4):
            actual_masked = output.mask[i].sum().item()
            assert actual_masked == expected_masked

    def test_block_masking_different_block_sizes(self):
        """Test various block sizes."""
        for block_size in [2, 3, 4, 7]:
            masking = PatchMasking(mask_ratio=0.5, block_size=block_size)
            x = torch.randn(2, 196, 64)
            output = masking(x, grid_h=14, grid_w=14)

            expected_masked = int(196 * 0.5)
            assert output.mask[0].sum().item() == expected_masked


class TestCropMasking:
    """Tests specific to crop masking strategy."""

    def test_crop_masking_exact_count(self, crop_masking):
        """Test that crop masking produces exact visible count."""
        x = torch.randn(4, 196, 64)
        output = crop_masking(x, grid_h=14, grid_w=14)

        expected_visible = 196 - int(196 * crop_masking.mask_ratio)
        for i in range(4):
            actual_visible = (output.mask[i] == 0).sum().item()
            assert actual_visible == expected_visible

    def test_crop_masking_contiguous_region(self, crop_masking):
        """Test that crop produces somewhat contiguous visible region."""
        x = torch.randn(2, 64, 32)
        output = crop_masking(x, grid_h=8, grid_w=8)

        for i in range(2):
            mask_2d = output.mask[i].view(8, 8)
            visible_2d = (mask_2d == 0).float()

            if visible_2d.sum() > 1:
                padded = torch.nn.functional.pad(
                    visible_2d.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1)
                )
                neighbor_sum = torch.nn.functional.avg_pool2d(
                    padded, 3, stride=1, padding=0
                )
                neighbor_sum = neighbor_sum.squeeze()

                visible_neighbor_avg = (
                    neighbor_sum * visible_2d
                ).sum() / visible_2d.sum()
                assert visible_neighbor_avg > 0.3, "Visible region seems too scattered"


class TestMixedStrategies:
    """Tests for mixed masking strategies."""

    def test_crop_ratio_probabilistic(self):
        """Test that crop_ratio controls strategy selection."""
        masking = PatchMasking(mask_ratio=0.5, block_size=1, crop_ratio=0.5)
        x = torch.randn(100, 64, 32)

        output = masking(x, grid_h=8, grid_w=8)
        assert output.visible.shape[0] == 100
        assert output.mask.shape[0] == 100


# =============================================================================
# Determinism Tests
# =============================================================================


class TestDeterminism:
    """Tests for reproducibility with manual seed."""

    def test_reproducible_with_seed(self, default_masking, small_input):
        """Test that same seed produces same output."""
        torch.manual_seed(123)
        output1 = default_masking(small_input, grid_h=4, grid_w=4)

        torch.manual_seed(123)
        output2 = default_masking(small_input, grid_h=4, grid_w=4)

        assert torch.equal(output1.mask, output2.mask)
        assert torch.equal(output1.ids_keep, output2.ids_keep)
        assert torch.equal(output1.visible, output2.visible)


# =============================================================================
# Gradient Tests
# =============================================================================


class TestGradients:
    """Tests for gradient flow."""

    def test_gradients_flow_through_visible(self, default_masking):
        """Test that gradients flow through visible patches."""
        x = torch.randn(2, 16, 64, requires_grad=True)
        output = default_masking(x, grid_h=4, grid_w=4)

        loss = output.visible.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


# =============================================================================
# Module State Tests
# =============================================================================


class TestModuleState:
    """Tests for nn.Module functionality."""

    def test_is_nn_module(self, default_masking):
        """Test that PatchMasking is an nn.Module."""
        assert isinstance(default_masking, torch.nn.Module)

    def test_no_learnable_parameters(self, default_masking):
        """Test that there are no learnable parameters."""
        params = list(default_masking.parameters())
        assert len(params) == 0

    def test_eval_mode(self, default_masking, small_input):
        """Test that module works in eval mode."""
        default_masking.eval()
        output = default_masking(small_input, grid_h=4, grid_w=4)
        assert output.visible is not None

    def test_train_mode(self, default_masking, small_input):
        """Test that module works in train mode."""
        default_masking.train()
        output = default_masking(small_input, grid_h=4, grid_w=4)
        assert output.visible is not None


@pytest.mark.unit
@pytest.mark.parametrize("input_type", ["pil", "tensor_float", "tensor_uint8"])
@pytest.mark.parametrize("fill_value", [None, 0.5, 0.0, 1.0])
def test_patch_masking_transform(input_type, fill_value):
    # Create a dummy image (3x32x32)
    np_img = np.ones((32, 32, 3), dtype=np.uint8) * 255
    if input_type == "pil":
        img = Image.fromarray(np_img)
    elif input_type == "tensor_float":
        img = (
            torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
        )  # C, H, W, float
    elif input_type == "tensor_uint8":
        img = torch.from_numpy(np_img).permute(2, 0, 1)  # C, H, W, uint8
    sample = {"image": img}
    patch_size = 8
    drop_ratio = 0.5
    transform = TPatchMasking(
        patch_size=patch_size,
        drop_ratio=drop_ratio,
        source="image",
        target="masked_image",
        fill_value=fill_value,
    )
    out = transform(sample)
    # Check output keys
    assert "masked_image" in out
    assert "patch_mask" in out
    # Check mask shape and dtype
    n_patches_h = 32 // patch_size
    n_patches_w = 32 // patch_size
    mask = out["patch_mask"]
    assert mask.shape == (n_patches_h, n_patches_w)
    assert mask.dtype == torch.bool
    # Check that masked_image is still an image of the same size and type
    masked_img = out["masked_image"]
    assert isinstance(masked_img, torch.Tensor)
    assert masked_img.shape == (3, 32, 32)
    masked_img_tensor = masked_img

    # Determine expected mask value
    if fill_value is not None:
        expected_fill_value = fill_value
    else:
        expected_fill_value = 0.0
    # Check that at least one patch is masked and that masked patches have the correct value
    found_masked = False
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            h_start = i * patch_size
            w_start = j * patch_size
            patch = masked_img_tensor[
                :, h_start : h_start + patch_size, w_start : w_start + patch_size
            ]
            if not mask[i, j]:
                found_masked = True
                # All values in the patch should be close to the mask value
                assert torch.allclose(
                    patch, torch.full_like(patch, expected_fill_value), atol=1e-2
                )
            else:
                # Only check if the original image is not all fill_value
                if not np.isclose(expected_fill_value, 1.0, atol=1e-2):
                    assert not torch.allclose(
                        patch, torch.full_like(patch, expected_fill_value), atol=1e-2
                    )
    assert found_masked, "At least one patch should be masked"


def test_patch_masking_fill_value_mean(monkeypatch):
    # Test using the mean of the image as fill_value
    np_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
    sample = {"image": img}
    patch_size = 8
    drop_ratio = 1.0  # Mask all patches

    # Patch the transform to use the mean as fill_value
    class TPatchMaskingMean(TPatchMasking):
        def __call__(self, x):
            img = self.nested_get(x, self.source)
            img_tensor = self._to_tensor(img)
            mean_val = img_tensor.mean().item()
            self.fill_value = mean_val
            return super().__call__(x)

    transform = TPatchMaskingMean(
        patch_size=patch_size,
        drop_ratio=drop_ratio,
        source="image",
        target="masked_image",
        fill_value=None,
    )
    out = transform(sample)
    masked_img = out["masked_image"]
    masked_img_tensor = (
        masked_img
        if isinstance(masked_img, torch.Tensor)
        else torch.from_numpy(np.array(masked_img)).permute(2, 0, 1).float() / 255.0
    )
    # All values should be close to the mean
    assert torch.allclose(
        masked_img_tensor,
        torch.full_like(masked_img_tensor, masked_img_tensor.mean()),
        atol=1e-2,
    )
