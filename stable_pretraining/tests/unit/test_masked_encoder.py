"""Unit tests for MaskedEncoder with RoPE (no pos_embed) and learned pos_embed models.

Tests two real timm models:
- vit_base_patch16_224: standard ViT with learned positional embeddings
- vit_base_patch16_dinov3.lvd1689m: DINOv3 with RoPE (pos_embed is None)

Run with: pytest stable_pretraining/tests/unit/test_masked_encoder.py -v -s
"""

import pytest
import torch

from stable_pretraining.backbone import MaskedEncoder, PatchMasking


BATCH_SIZE = 2
CHANNELS = 3

MODELS = {
    "vit_base": {
        "name": "vit_base_patch16_224",
        "img_size": 224,
        "pos_embed_is_none": False,
    },
    "dinov3": {
        "name": "vit_base_patch16_dinov3.lvd1689m",
        "img_size": 256,
        "pos_embed_is_none": True,
    },
}


@pytest.fixture(params=["vit_base", "dinov3"], scope="module")
def model_key(request):
    return request.param


@pytest.fixture(scope="module")
def encoder_no_mask(model_key):
    """MaskedEncoder without masking (inference-like)."""
    cfg = MODELS[model_key]
    enc = MaskedEncoder(cfg["name"], masking=None, pretrained=False)
    enc.eval()
    return enc, cfg


@pytest.fixture(scope="module")
def encoder_with_mask(model_key):
    """MaskedEncoder with 75% random masking."""
    cfg = MODELS[model_key]
    masking = PatchMasking(mask_ratio=0.75)
    enc = MaskedEncoder(cfg["name"], masking=masking, pretrained=False)
    enc.train()
    return enc, cfg


@pytest.fixture
def sample_images(encoder_no_mask):
    """Generate sample images matching the model's expected input size."""
    _, cfg = encoder_no_mask
    img_size = cfg["img_size"]
    return torch.randn(BATCH_SIZE, CHANNELS, img_size, img_size)


def _actual_prefix_count(enc) -> int:
    """Count how many prefix tokens _get_prefix_tokens actually prepends."""
    prefix = enc._get_prefix_tokens(1)
    return prefix.shape[1] if prefix is not None else 0


# ============================================================================
# pos_embed presence
# ============================================================================


@pytest.mark.unit
@pytest.mark.download
class TestPosEmbedPresence:
    """Verify pos_embed is None for RoPE models and exists for standard models."""

    def test_pos_embed_value(self, encoder_no_mask):
        enc, cfg = encoder_no_mask
        pos_embed = enc.vit.pos_embed
        if cfg["pos_embed_is_none"]:
            assert pos_embed is None, (
                f"{cfg['name']}: expected pos_embed=None (RoPE), got {type(pos_embed)}"
            )
        else:
            assert pos_embed is not None, (
                f"{cfg['name']}: expected learned pos_embed, got None"
            )


# ============================================================================
# _get_pos_embed
# ============================================================================


@pytest.mark.unit
@pytest.mark.download
class TestGetPosEmbed:
    """Test _get_pos_embed handles both None and tensor pos_embed."""

    def test_return_types(self, encoder_no_mask):
        enc, cfg = encoder_no_mask
        grid_h, grid_w = enc.default_grid_h, enc.default_grid_w
        prefix_pos, patch_pos = enc._get_pos_embed(grid_h, grid_w)

        if cfg["pos_embed_is_none"]:
            assert prefix_pos is None
            assert patch_pos is None
        else:
            assert patch_pos is not None
            assert patch_pos.shape[-1] == enc.embed_dim
            num_patches = grid_h * grid_w
            assert patch_pos.shape[1] == num_patches


# ============================================================================
# _resize_pos_embed
# ============================================================================


@pytest.mark.unit
@pytest.mark.download
class TestResizePosEmbed:
    """Test _resize_pos_embed is safe when pos_embed is None."""

    @pytest.fixture(params=["vit_base", "dinov3"])
    def fresh_encoder(self, request):
        """A per-test encoder so resize doesn't pollute other tests."""
        cfg = MODELS[request.param]
        enc = MaskedEncoder(cfg["name"], masking=None, pretrained=False)
        enc.eval()
        return enc, cfg

    def test_resize_no_crash(self, fresh_encoder):
        enc, cfg = fresh_encoder
        new_grid = (enc.default_grid_h + 2, enc.default_grid_w + 2)
        enc._resize_pos_embed(new_grid)
        if cfg["pos_embed_is_none"]:
            assert enc.vit.pos_embed is None
        else:
            num_prefix = enc.num_prefix_tokens if not enc.no_embed_class else 0
            new_patches = new_grid[0] * new_grid[1]
            assert enc.vit.pos_embed.shape[1] == num_prefix + new_patches


# ============================================================================
# Forward (no masking)
# ============================================================================


@pytest.mark.unit
@pytest.mark.download
class TestForwardNoMask:
    """Test forward pass without masking for both model types."""

    def test_output_shape(self, encoder_no_mask, sample_images):
        enc, cfg = encoder_no_mask
        output = enc(sample_images)

        grid_h, grid_w = enc.default_grid_h, enc.default_grid_w
        num_patches = grid_h * grid_w
        num_prefix = _actual_prefix_count(enc)
        expected_seq_len = num_prefix + num_patches

        assert output.encoded.shape == (BATCH_SIZE, expected_seq_len, enc.embed_dim)
        assert output.mask.shape == (BATCH_SIZE, num_patches)
        assert output.ids_keep.shape == (BATCH_SIZE, num_patches)
        assert output.grid_size == (grid_h, grid_w)

    def test_no_nan(self, encoder_no_mask, sample_images):
        enc, _ = encoder_no_mask
        with torch.no_grad():
            output = enc(sample_images)
        assert not torch.isnan(output.encoded).any()

    def test_mask_all_zeros(self, encoder_no_mask, sample_images):
        enc, _ = encoder_no_mask
        with torch.no_grad():
            output = enc(sample_images)
        assert (output.mask == 0).all(), "Without masking, mask should be all zeros"

    def test_deterministic(self, encoder_no_mask, sample_images):
        enc, _ = encoder_no_mask
        enc.eval()
        with torch.no_grad():
            out1 = enc(sample_images)
            out2 = enc(sample_images)
        torch.testing.assert_close(out1.encoded, out2.encoded)


# ============================================================================
# Forward (with masking)
# ============================================================================


@pytest.mark.unit
@pytest.mark.download
class TestForwardWithMask:
    """Test forward pass with masking for both model types."""

    def test_output_shape_masked(self, encoder_with_mask, sample_images):
        enc, cfg = encoder_with_mask
        enc.train()
        output = enc(sample_images)

        grid_h, grid_w = enc.default_grid_h, enc.default_grid_w
        num_patches = grid_h * grid_w
        num_visible = num_patches - int(num_patches * 0.75)
        num_prefix = _actual_prefix_count(enc)
        expected_seq_len = num_prefix + num_visible

        assert output.encoded.shape == (BATCH_SIZE, expected_seq_len, enc.embed_dim)
        assert output.mask.shape == (BATCH_SIZE, num_patches)
        assert output.ids_keep.shape == (BATCH_SIZE, num_visible)

    def test_mask_has_ones(self, encoder_with_mask, sample_images):
        enc, _ = encoder_with_mask
        enc.train()
        output = enc(sample_images)
        assert output.mask.sum() > 0, (
            "With 75% masking, mask should have masked entries"
        )

    def test_no_nan_masked(self, encoder_with_mask, sample_images):
        enc, _ = encoder_with_mask
        enc.train()
        output = enc(sample_images)
        assert not torch.isnan(output.encoded).any()


# ============================================================================
# Gradient flow
# ============================================================================


@pytest.mark.unit
@pytest.mark.download
class TestGradientFlow:
    """Test gradients flow correctly for both model types."""

    def test_gradient_no_mask(self, encoder_no_mask, sample_images):
        enc, _ = encoder_no_mask
        enc.train()
        for p in enc.parameters():
            p.requires_grad = True

        output = enc(sample_images)
        loss = output.encoded.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters()
        )
        assert has_grad, "No gradients found in model parameters"

    def test_gradient_with_mask(self, encoder_with_mask, sample_images):
        enc, _ = encoder_with_mask
        enc.train()
        for p in enc.parameters():
            p.requires_grad = True

        output = enc(sample_images)
        loss = output.encoded.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters()
        )
        assert has_grad, "No gradients found in model parameters"


# ============================================================================
# forward_features
# ============================================================================


@pytest.mark.unit
@pytest.mark.download
class TestForwardFeatures:
    """Test forward_features convenience method."""

    def test_forward_features_shape(self, encoder_no_mask, sample_images):
        enc, _ = encoder_no_mask
        features = enc.forward_features(sample_images)

        grid_h, grid_w = enc.default_grid_h, enc.default_grid_w
        num_patches = grid_h * grid_w
        num_prefix = _actual_prefix_count(enc)
        expected_seq_len = num_prefix + num_patches

        assert features.shape == (BATCH_SIZE, expected_seq_len, enc.embed_dim)
        assert not torch.isnan(features).any()
