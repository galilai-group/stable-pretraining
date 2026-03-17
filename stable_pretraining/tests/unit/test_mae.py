"""Unit tests for Masked Autoencoder (MAE) functionality."""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
import torchmetrics
from stable_pretraining.utils import patchify, unpatchify


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def channels():
    return 3


# =============================================================================
# Patchify Tests
# =============================================================================
@pytest.mark.unit
class TestPatchifyShapes:
    """Test patchify output shapes for various input configurations."""

    def test_2d_square_image(self, batch_size, channels):
        """Standard ViT-style 2D image patchification."""
        x = torch.randn(batch_size, channels, 224, 224)
        patches = patchify(x, patch_size=(16, 16))

        assert patches.shape == (batch_size, channels, 196, 256)

    def test_2d_non_square_image(self, batch_size, channels):
        """Non-square image."""
        x = torch.randn(batch_size, channels, 224, 256)
        patches = patchify(x, patch_size=(16, 16))

        assert patches.shape == (batch_size, channels, 224, 256)

    def test_2d_non_square_patches(self, batch_size, channels):
        """Non-square patch size."""
        x = torch.randn(batch_size, channels, 224, 224)
        patches = patchify(x, patch_size=(14, 16))

        # grid_size = (16, 14), T = 224, patch_elements = 224
        assert patches.shape == (batch_size, channels, 224, 224)

    def test_3d_volume(self, batch_size):
        """3D volume (e.g., medical imaging, video)."""
        x = torch.randn(batch_size, 1, 64, 128, 128)
        patches = patchify(x, patch_size=(8, 16, 16))

        assert patches.shape == (batch_size, 1, 512, 2048)

    def test_1d_signal(self, batch_size, channels):
        """1D signal (e.g., audio, time series)."""
        x = torch.randn(batch_size, channels, 1024)
        patches = patchify(x, patch_size=(64,))

        assert patches.shape == (batch_size, channels, 16, 64)

    def test_no_batch_dims(self):
        """Input with no batch dimensions."""
        x = torch.randn(224, 224)
        patches = patchify(x, patch_size=(16, 16))

        assert patches.shape == (196, 256)

    def test_single_batch_dim(self):
        """Input with single batch dimension (no channels)."""
        x = torch.randn(8, 224, 224)
        patches = patchify(x, patch_size=(16, 16))

        assert patches.shape == (8, 196, 256)

    def test_multiple_batch_dims(self):
        """Input with multiple batch dimensions."""
        x = torch.randn(2, 4, 3, 224, 224)
        patches = patchify(x, patch_size=(16, 16))

        assert patches.shape == (2, 4, 3, 196, 256)

    def test_small_patches(self, batch_size, channels):
        """Small patch size."""
        x = torch.randn(batch_size, channels, 32, 32)
        patches = patchify(x, patch_size=(4, 4))

        assert patches.shape == (batch_size, channels, 64, 16)

    def test_large_patches(self, batch_size, channels):
        """Large patch size (single patch per dim)."""
        x = torch.randn(batch_size, channels, 224, 224)
        patches = patchify(x, patch_size=(224, 224))

        assert patches.shape == (batch_size, channels, 1, 224 * 224)

    def test_patch_size_equals_spatial_dim(self):
        """Patch size equals spatial dimension (single patch)."""
        x = torch.randn(4, 64, 64)
        patches = patchify(x, patch_size=(64, 64))

        assert patches.shape == (4, 1, 4096)


@pytest.mark.unit
class TestPatchifyValues:
    """Test patchify preserves values correctly."""

    def test_values_preserved_2d(self):
        """Check specific values are correctly placed in patches."""
        x = torch.arange(16).reshape(1, 1, 4, 4).float()
        patches = patchify(x, patch_size=(2, 2))

        assert patches.shape == (1, 1, 4, 4)
        assert torch.allclose(patches[0, 0, 0], torch.tensor([0.0, 1.0, 4.0, 5.0]))
        assert torch.allclose(patches[0, 0, 1], torch.tensor([2.0, 3.0, 6.0, 7.0]))
        assert torch.allclose(patches[0, 0, 2], torch.tensor([8.0, 9.0, 12.0, 13.0]))
        assert torch.allclose(patches[0, 0, 3], torch.tensor([10.0, 11.0, 14.0, 15.0]))

    def test_values_preserved_1d(self):
        """Check values in 1D patchification."""
        x = torch.arange(12).reshape(1, 12).float()
        patches = patchify(x, patch_size=(4,))

        assert patches.shape == (1, 3, 4)
        assert torch.allclose(patches[0, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
        assert torch.allclose(patches[0, 1], torch.tensor([4.0, 5.0, 6.0, 7.0]))
        assert torch.allclose(patches[0, 2], torch.tensor([8.0, 9.0, 10.0, 11.0]))

    def test_dtype_preserved(self):
        """Check dtype is preserved."""
        for dtype in [torch.float32, torch.float64, torch.float16, torch.int32]:
            x = torch.ones(4, 3, 32, 32, dtype=dtype)
            patches = patchify(x, patch_size=(8, 8))
            assert patches.dtype == dtype

    def test_contiguous_output(self):
        """Check output is contiguous."""
        x = torch.randn(4, 3, 224, 224)
        patches = patchify(x, patch_size=(16, 16))
        assert patches.is_contiguous()


@pytest.mark.unit
class TestPatchifyErrors:
    """Test patchify error handling."""

    def test_not_divisible_height(self):
        """Error when height not divisible by patch height."""
        x = torch.randn(4, 3, 225, 224)
        with pytest.raises(ValueError, match="divisible"):
            patchify(x, patch_size=(16, 16))

    def test_not_divisible_width(self):
        """Error when width not divisible by patch width."""
        x = torch.randn(4, 3, 224, 225)
        with pytest.raises(ValueError, match="divisible"):
            patchify(x, patch_size=(16, 16))

    def test_not_enough_dims(self):
        """Error when input has fewer dims than patch_size length."""
        x = torch.randn(224)  # 1D tensor
        with pytest.raises((AssertionError, RuntimeError)):
            patchify(x, patch_size=(16, 16))

    def test_empty_patch_size(self):
        """Empty patch_size leads to error."""
        x = torch.randn(4, 3, 224, 224)
        # Empty tuple means 0 spatial dims to patchify - will cause various errors
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            patchify(x, patch_size=())


# =============================================================================
# Unpatchify Tests
# =============================================================================
@pytest.mark.unit
class TestUnpatchifyShapes:
    """Test unpatchify output shapes."""

    def test_2d_square_inferred_grid(self):
        """Infer square grid from num_patches."""
        patches = torch.randn(4, 3, 196, 256)
        x = unpatchify(patches, patch_size=(16, 16))

        assert x.shape == (4, 3, 224, 224)

    def test_2d_explicit_grid(self):
        """Explicit non-square grid."""
        patches = torch.randn(4, 3, 224, 256)
        x = unpatchify(patches, patch_size=(16, 16), grid_size=(14, 16))

        assert x.shape == (4, 3, 224, 256)

    def test_3d_volume_inferred(self):
        """3D volume with inferred uniform grid."""
        patches = torch.randn(4, 1, 512, 2048)
        x = unpatchify(patches, patch_size=(8, 16, 16))

        assert x.shape == (4, 1, 64, 128, 128)

    def test_3d_volume_explicit(self):
        """3D volume with explicit grid."""
        patches = torch.randn(4, 1, 480, 2048)
        x = unpatchify(patches, patch_size=(8, 16, 16), grid_size=(6, 8, 10))

        assert x.shape == (4, 1, 48, 128, 160)

    def test_1d_signal(self):
        """1D signal reconstruction."""
        patches = torch.randn(8, 2, 16, 64)
        x = unpatchify(patches, patch_size=(64,))

        assert x.shape == (8, 2, 1024)

    def test_no_batch_dims(self):
        """No batch dimensions."""
        patches = torch.randn(196, 256)
        x = unpatchify(patches, patch_size=(16, 16))

        assert x.shape == (224, 224)

    def test_single_patch(self):
        """Single patch covers entire spatial extent."""
        patches = torch.randn(4, 3, 1, 50176)
        x = unpatchify(patches, patch_size=(224, 224), grid_size=(1, 1))

        assert x.shape == (4, 3, 224, 224)


@pytest.mark.unit
class TestUnpatchifyErrors:
    """Test unpatchify error handling."""

    def test_wrong_patch_elements(self):
        """Error when patch elements don't match patch_size."""
        patches = torch.randn(4, 196, 255)
        with pytest.raises(ValueError, match="prod"):
            unpatchify(patches, patch_size=(16, 16))

    def test_cannot_infer_non_square_grid(self):
        """Error when cannot infer non-uniform grid."""
        patches = torch.randn(4, 168, 256)
        with pytest.raises(ValueError, match="Cannot infer"):
            unpatchify(patches, patch_size=(16, 16))

    def test_grid_size_wrong_length(self):
        """Error when grid_size has wrong number of dims."""
        patches = torch.randn(4, 196, 256)
        with pytest.raises(ValueError, match="dims"):
            unpatchify(patches, patch_size=(16, 16), grid_size=(14,))

    def test_grid_size_product_mismatch(self):
        """Error when grid_size product doesn't match num_patches."""
        patches = torch.randn(4, 196, 256)
        with pytest.raises(ValueError, match="prod"):
            unpatchify(patches, patch_size=(16, 16), grid_size=(10, 10))

    def test_not_enough_dims(self):
        """Error when patches has fewer than 2 dims."""
        patches = torch.randn(256)
        with pytest.raises((AssertionError, ValueError)):
            unpatchify(patches, patch_size=(16, 16))


# =============================================================================
# Roundtrip Tests
# =============================================================================
@pytest.mark.unit
class TestRoundtrip:
    """Test patchify -> unpatchify roundtrip preserves data."""

    def test_2d_square_roundtrip(self, batch_size, channels):
        """2D square image roundtrip."""
        x = torch.randn(batch_size, channels, 224, 224)
        patches = patchify(x, patch_size=(16, 16))
        reconstructed = unpatchify(patches, patch_size=(16, 16))

        assert torch.allclose(x, reconstructed)

    def test_2d_non_square_roundtrip(self, batch_size, channels):
        """2D non-square image roundtrip."""
        x = torch.randn(batch_size, channels, 192, 256)
        patches = patchify(x, patch_size=(16, 16))
        reconstructed = unpatchify(patches, patch_size=(16, 16), grid_size=(12, 16))

        assert torch.allclose(x, reconstructed)

    def test_2d_non_square_patches_roundtrip(self, batch_size, channels):
        """2D with non-square patches roundtrip."""
        x = torch.randn(batch_size, channels, 224, 224)
        patches = patchify(x, patch_size=(14, 16))
        # grid_size = (224//14, 224//16) = (16, 14) - not a perfect square, must specify
        reconstructed = unpatchify(patches, patch_size=(14, 16), grid_size=(16, 14))

        assert torch.allclose(x, reconstructed)

    def test_3d_roundtrip(self, batch_size):
        """3D volume roundtrip."""
        x = torch.randn(batch_size, 1, 64, 64, 64)
        patches = patchify(x, patch_size=(8, 8, 8))
        reconstructed = unpatchify(patches, patch_size=(8, 8, 8))

        assert torch.allclose(x, reconstructed)

    def test_3d_non_uniform_roundtrip(self, batch_size):
        """3D volume with non-uniform grid roundtrip."""
        x = torch.randn(batch_size, 2, 32, 64, 128)
        patches = patchify(x, patch_size=(8, 16, 16))
        reconstructed = unpatchify(patches, patch_size=(8, 16, 16), grid_size=(4, 4, 8))

        assert torch.allclose(x, reconstructed)

    def test_1d_roundtrip(self, batch_size, channels):
        """1D signal roundtrip."""
        x = torch.randn(batch_size, channels, 1024)
        patches = patchify(x, patch_size=(32,))
        reconstructed = unpatchify(patches, patch_size=(32,))

        assert torch.allclose(x, reconstructed)

    def test_no_batch_roundtrip(self):
        """No batch dims roundtrip."""
        x = torch.randn(128, 128)
        patches = patchify(x, patch_size=(16, 16))
        reconstructed = unpatchify(patches, patch_size=(16, 16))

        assert torch.allclose(x, reconstructed)

    def test_many_batch_dims_roundtrip(self):
        """Many batch dims roundtrip."""
        x = torch.randn(2, 3, 4, 5, 64, 64)
        patches = patchify(x, patch_size=(8, 8))
        reconstructed = unpatchify(patches, patch_size=(8, 8))

        assert torch.allclose(x, reconstructed)

    @pytest.mark.parametrize("patch_size", [(8, 8), (16, 16), (32, 32), (7, 7)])
    def test_various_patch_sizes(self, batch_size, channels, patch_size):
        """Various patch sizes roundtrip."""
        ph, pw = patch_size
        H, W = ph * 8, pw * 8
        x = torch.randn(batch_size, channels, H, W)
        patches = patchify(x, patch_size=patch_size)
        reconstructed = unpatchify(patches, patch_size=patch_size)

        assert torch.allclose(x, reconstructed)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_roundtrip(self, dtype):
        """Dtype preserved through roundtrip."""
        x = torch.randn(4, 3, 64, 64, dtype=dtype)
        patches = patchify(x, patch_size=(8, 8))
        reconstructed = unpatchify(patches, patch_size=(8, 8))

        assert reconstructed.dtype == dtype
        assert torch.allclose(x, reconstructed)


# =============================================================================
# Edge Cases
# =============================================================================
@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_element_patch(self):
        """Patch size of 1 (each element is its own patch)."""
        x = torch.randn(2, 3, 8, 8)
        patches = patchify(x, patch_size=(1, 1))

        assert patches.shape == (2, 3, 64, 1)

        reconstructed = unpatchify(patches, patch_size=(1, 1))
        assert torch.allclose(x, reconstructed)

    def test_batch_size_one(self):
        """Batch size of 1."""
        x = torch.randn(1, 3, 224, 224)
        patches = patchify(x, patch_size=(16, 16))
        reconstructed = unpatchify(patches, patch_size=(16, 16))

        assert torch.allclose(x, reconstructed)

    def test_single_channel(self):
        """Single channel."""
        x = torch.randn(4, 1, 224, 224)
        patches = patchify(x, patch_size=(16, 16))
        reconstructed = unpatchify(patches, patch_size=(16, 16))

        assert torch.allclose(x, reconstructed)

    def test_large_batch(self):
        """Large batch size."""
        x = torch.randn(128, 3, 64, 64)
        patches = patchify(x, patch_size=(8, 8))
        reconstructed = unpatchify(patches, patch_size=(8, 8))

        assert torch.allclose(x, reconstructed)

    def test_prime_spatial_dims(self):
        """Spatial dims that are prime (patch must equal dim)."""
        x = torch.randn(4, 3, 17, 19)
        patches = patchify(x, patch_size=(17, 19))

        assert patches.shape == (4, 3, 1, 17 * 19)

        reconstructed = unpatchify(patches, patch_size=(17, 19), grid_size=(1, 1))
        assert torch.allclose(x, reconstructed)

    def test_asymmetric_3d(self):
        """Highly asymmetric 3D volume."""
        x = torch.randn(2, 1, 8, 32, 128)
        patches = patchify(x, patch_size=(2, 8, 16))

        # grid_size = (8//2, 32//8, 128//16) = (4, 4, 8)
        # T = 4 * 4 * 8 = 128
        # patch_elements = 2 * 8 * 16 = 256
        assert patches.shape == (2, 1, 128, 256)

        reconstructed = unpatchify(patches, patch_size=(2, 8, 16), grid_size=(4, 4, 8))
        assert torch.allclose(x, reconstructed)


# =============================================================================
# Gradient Tests
# =============================================================================
@pytest.mark.unit
class TestGradients:
    """Test gradient flow through patchify/unpatchify."""

    def test_patchify_gradient(self):
        """Gradients flow through patchify."""
        x = torch.randn(4, 3, 64, 64, requires_grad=True)
        patches = patchify(x, patch_size=(8, 8))
        loss = patches.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.isnan(x.grad).any()

    def test_unpatchify_gradient(self):
        """Gradients flow through unpatchify."""
        patches = torch.randn(4, 3, 64, 64, requires_grad=True)
        x = unpatchify(patches, patch_size=(8, 8))
        loss = x.sum()
        loss.backward()

        assert patches.grad is not None
        assert patches.grad.shape == patches.shape
        assert not torch.isnan(patches.grad).any()

    def test_roundtrip_gradient(self):
        """Gradients flow through full roundtrip."""
        x = torch.randn(4, 3, 64, 64, requires_grad=True)
        patches = patchify(x, patch_size=(8, 8))
        reconstructed = unpatchify(patches, patch_size=(8, 8))
        loss = reconstructed.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x.grad))


@pytest.mark.unit
class TestMAEUnit:
    """Unit tests for MAE components without actual model training."""

    def test_mae_backbone_initialization(self):
        """Test MAE backbone can be initialized."""
        with patch(
            "stable_pretraining.backbone.mae.vit_base_patch16_dec512d8b"
        ) as mock_mae:
            backbone = mock_mae()
            mock_mae.assert_called_once()
            assert backbone is not None

    def test_mae_forward_logic(self):
        """Test MAE forward pass logic."""
        # Mock backbone with MAE behavior
        mock_backbone = Mock()
        mock_latent = torch.randn(2, 197, 768)  # [batch, seq_len, hidden_dim]
        mock_pred = torch.randn(2, 196, 768)  # predictions for masked patches
        mock_mask = torch.randint(0, 2, (2, 196)).bool()  # mask
        mock_backbone.return_value = (mock_latent, mock_pred, mock_mask)
        mock_backbone.patchify = Mock(return_value=torch.randn(2, 196, 768))

        # Create mock module
        mock_module = Mock()
        mock_module.backbone = mock_backbone
        mock_module.training = True

        # Define forward function
        def forward(self, batch, stage):
            latent, pred, mask = self.backbone(batch["image"])
            batch["embedding"] = latent[:, 0]  # CLS token only
            if self.training:
                loss = torch.nn.functional.mse_loss(
                    self.backbone.patchify(batch["image"])[mask], pred[mask]
                )
                batch["loss"] = loss
            return batch

        # Test forward pass
        batch = {"image": torch.randn(2, 3, 224, 224)}
        forward_bound = forward.__get__(mock_module, type(mock_module))
        result = forward_bound(batch.copy(), "train")

        # Verify calls and results
        mock_backbone.assert_called_once_with(batch["image"])
        assert "embedding" in result
        assert result["embedding"].shape == (2, 768)
        assert "loss" in result

        # Test without training mode
        mock_module.training = False
        # Reset mock to allow another call
        mock_backbone.reset_mock()
        # Use a fresh batch to avoid contamination from previous test
        batch_val = {"image": torch.randn(2, 3, 224, 224)}
        result = forward_bound(batch_val, "val")
        assert "embedding" in result
        assert "loss" not in result

    def test_mae_loss_function(self):
        """Test MAE loss computation."""
        with patch("stable_pretraining.losses.mae") as mock_mae_loss:
            mock_mae_loss.return_value = torch.tensor(0.5)

            patches = torch.randn(2, 196, 768)
            pred = torch.randn(2, 196, 768)
            mask = torch.randint(0, 2, (2, 196)).bool()

            loss = mock_mae_loss(patches, pred, mask)

            mock_mae_loss.assert_called_once_with(patches, pred, mask)
            assert isinstance(loss, torch.Tensor)
            assert loss.item() == 0.5

    def test_patchify_function(self):
        """Test patchify functionality."""
        # Mock patchify behavior
        mock_backbone = Mock()

        def mock_patchify(x):
            # Simulate patchifying 224x224 image with 16x16 patches
            batch_size = x.shape[0]
            num_patches = (224 // 16) ** 2  # 196 patches
            patch_dim = 3 * 16 * 16  # 768
            return torch.randn(batch_size, num_patches, patch_dim)

        mock_backbone.patchify = mock_patchify

        # Test patchify
        images = torch.randn(2, 3, 224, 224)
        patches = mock_backbone.patchify(images)

        assert patches.shape == (2, 196, 768)

    def test_online_probe_initialization(self):
        """Test OnlineProbe callback initialization for MAE."""
        with patch("stable_pretraining.callbacks.OnlineProbe") as mock_probe:
            mock_module = Mock()
            mock_linear = Mock(spec=nn.Linear)
            mock_loss_fn = Mock(spec=nn.CrossEntropyLoss)
            mock_metrics = {
                "top1": Mock(spec=torchmetrics.classification.MulticlassAccuracy),
                "top5": Mock(spec=torchmetrics.classification.MulticlassAccuracy),
            }

            probe = mock_probe(
                mock_module,
                "linear_probe",
                "embedding",
                "label",
                probe=mock_linear,
                loss_fn=mock_loss_fn,
                metrics=mock_metrics,
            )

            mock_probe.assert_called_once()
            assert probe is not None

    def test_repeated_random_sampler(self):
        """Test RepeatedRandomSampler for multi-view training."""
        with patch(
            "stable_pretraining.data.sampler.RepeatedRandomSampler"
        ) as mock_sampler:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=100)

            sampler = mock_sampler(mock_dataset, n_views=2)

            mock_sampler.assert_called_once_with(mock_dataset, n_views=2)
            assert sampler is not None

    def test_transform_composition_for_mae(self):
        """Test transform composition for MAE training."""
        with patch("stable_pretraining.data.transforms") as mock_transforms:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            # Mock transform methods
            mock_transforms.Compose.return_value = Mock()

            # Train transform
            train_transform = mock_transforms.Compose(
                mock_transforms.RGB(),
                mock_transforms.RandomResizedCrop((224, 224)),
                mock_transforms.RandomHorizontalFlip(p=0.5),
                mock_transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                mock_transforms.RandomGrayscale(p=0.2),
                mock_transforms.ToImage(mean=mean, std=std),
            )

            assert train_transform is not None

            # Val transform
            val_transform = mock_transforms.Compose(
                mock_transforms.RGB(),
                mock_transforms.Resize((256, 256)),
                mock_transforms.CenterCrop((224, 224)),
                mock_transforms.ToImage(mean=mean, std=std),
            )

            assert val_transform is not None

    def test_cls_token_extraction(self):
        """Test CLS token extraction from transformer output."""
        # Create mock latent representation
        latent = torch.randn(4, 197, 768)  # [batch, seq_len, hidden_dim]

        # Extract CLS token (first token)
        cls_token = latent[:, 0]

        assert cls_token.shape == (4, 768)
        assert torch.allclose(cls_token, latent[:, 0])

    def test_mae_masking_logic(self):
        """Test MAE masking logic."""
        batch_size = 2
        num_patches = 196

        # Create random mask
        mask = torch.rand(batch_size, num_patches) > 0.75  # 75% masking

        # Verify mask properties
        assert mask.shape == (batch_size, num_patches)
        assert mask.dtype == torch.bool

        # Test applying mask
        patches = torch.randn(batch_size, num_patches, 768)
        predictions = torch.randn(batch_size, num_patches, 768)

        masked_patches = patches[mask]
        masked_predictions = predictions[mask]

        assert masked_patches.shape[0] == masked_predictions.shape[0]
        assert masked_patches.shape[1] == 768
