"""Unit tests for MaskedEncoder across diverse timm ViT families.

Tests six real timm models covering standard ViT, DINOv2, RoPE+registers
(vit_betwixt_patch16_rope_reg4_gap_256), MAE, and CLIP backbones. Validates prefix-token detection, positional embedding
handling, forward passes (with/without masking), gradient flow, and that
string-based vs pre-instantiated model creation produces identical behaviour.

Run with: pytest stable_pretraining/tests/unit/test_masked_encoder.py -v -s
"""

import pytest
import timm
import torch

from stable_pretraining.backbone import MaskedEncoder, PatchMasking


BATCH_SIZE = 2
CHANNELS = 3
MASK_RATIO = 0.75

MODELS = {
    "vit_base": {
        "name": "vit_base_patch16_224",
        "img_size": 224,
        "pos_embed_is_none": False,
        "expected_num_reg": 0,
    },
    "dinov2": {
        "name": "vit_base_patch14_dinov2.lvd142m",
        "img_size": 518,
        "pos_embed_is_none": False,
        "expected_num_reg": 0,
    },
    "dinov3": {
        "name": "vit_betwixt_patch16_rope_reg4_gap_256",
        "img_size": 256,
        "pos_embed_is_none": True,
        "expected_num_reg": 4,
    },
    "mae": {
        "name": "vit_base_patch16_224.mae",
        "img_size": 224,
        "pos_embed_is_none": False,
        "expected_num_reg": 0,
    },
    "clip_openai": {
        "name": "vit_base_patch16_clip_224.openai",
        "img_size": 224,
        "pos_embed_is_none": False,
        "expected_num_reg": 0,
    },
    "clip_laion": {
        "name": "vit_base_patch16_clip_224.laion2b",
        "img_size": 224,
        "pos_embed_is_none": False,
        "expected_num_reg": 0,
    },
}

ALL_KEYS = list(MODELS.keys())


def _actual_prefix_count(enc: MaskedEncoder) -> int:
    """Count how many prefix tokens _get_prefix_tokens actually prepends."""
    prefix = enc._get_prefix_tokens(1)
    return prefix.shape[1] if prefix is not None else 0


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(params=ALL_KEYS, scope="module")
def model_key(request):
    return request.param


@pytest.fixture(scope="module")
def encoder_no_mask(model_key):
    """MaskedEncoder without masking (inference-like), created from string."""
    cfg = MODELS[model_key]
    enc = MaskedEncoder(cfg["name"], masking=None, pretrained=False)
    enc.eval()
    return enc, cfg


@pytest.fixture(scope="module")
def encoder_with_mask(model_key):
    """MaskedEncoder with masking, created from string."""
    cfg = MODELS[model_key]
    masking = PatchMasking(mask_ratio=MASK_RATIO)
    enc = MaskedEncoder(cfg["name"], masking=masking, pretrained=False)
    enc.train()
    return enc, cfg


@pytest.fixture(scope="module")
def encoder_from_model(model_key):
    """MaskedEncoder created from a pre-instantiated timm model (training path)."""
    cfg = MODELS[model_key]
    backbone = timm.create_model(
        cfg["name"], pretrained=False, num_classes=0, img_size=cfg["img_size"]
    )
    masking = PatchMasking(mask_ratio=MASK_RATIO)
    enc = MaskedEncoder(backbone, masking=masking)
    enc.train()
    return enc, cfg


@pytest.fixture
def sample_images(encoder_no_mask):
    """Generate sample images matching the model's expected input size."""
    _, cfg = encoder_no_mask
    s = cfg["img_size"]
    return torch.randn(BATCH_SIZE, CHANNELS, s, s)


# ============================================================================
# Prefix token detection
# ============================================================================


@pytest.mark.unit
@pytest.mark.download
class TestPrefixTokenDetection:
    """Verify prefix token attributes are correctly inferred from timm model."""

    def test_has_class_token(self, encoder_no_mask):
        enc, cfg = encoder_no_mask
        vit_has_cls = hasattr(enc.vit, "cls_token") and enc.vit.cls_token is not None
        assert enc.has_class_token == vit_has_cls

    def test_num_reg_tokens(self, encoder_no_mask):
        enc, cfg = encoder_no_mask
        assert enc.num_reg_tokens == cfg["expected_num_reg"], (
            f"{cfg['name']}: expected {cfg['expected_num_reg']} register tokens, "
            f"got {enc.num_reg_tokens}"
        )

    def test_num_prefix_matches_actual(self, encoder_no_mask):
        """num_prefix_tokens must equal what _get_prefix_tokens actually prepends."""
        enc, _ = encoder_no_mask
        assert enc.num_prefix_tokens == _actual_prefix_count(enc)

    def test_num_prefix_matches_timm(self, encoder_no_mask):
        """Computed prefix count must agree with timm's own attribute."""
        enc, cfg = encoder_no_mask
        timm_val = getattr(enc.vit, "num_prefix_tokens", None)
        if timm_val is not None:
            assert enc.num_prefix_tokens == timm_val, (
                f"{cfg['name']}: computed {enc.num_prefix_tokens} vs timm {timm_val}"
            )


# ============================================================================
# Prefix detection: string vs pre-instantiated model
# ============================================================================


@pytest.mark.unit
@pytest.mark.download
class TestPrefixTokenDetectionPreInstantiated:
    """Ensure pre-instantiated model path gives identical prefix detection."""

    def test_same_prefix_count(self, encoder_with_mask, encoder_from_model):
        enc_str, _ = encoder_with_mask
        enc_mod, _ = encoder_from_model
        assert enc_str.num_prefix_tokens == enc_mod.num_prefix_tokens

    def test_same_reg_count(self, encoder_with_mask, encoder_from_model):
        enc_str, _ = encoder_with_mask
        enc_mod, _ = encoder_from_model
        assert enc_str.num_reg_tokens == enc_mod.num_reg_tokens

    def test_same_has_class_token(self, encoder_with_mask, encoder_from_model):
        enc_str, _ = encoder_with_mask
        enc_mod, _ = encoder_from_model
        assert enc_str.has_class_token == enc_mod.has_class_token

    def test_prefix_matches_actual_pre_instantiated(self, encoder_from_model):
        enc, _ = encoder_from_model
        assert enc.num_prefix_tokens == _actual_prefix_count(enc)


# ============================================================================
# pos_embed presence
# ============================================================================


@pytest.mark.unit
@pytest.mark.download
class TestPosEmbedPresence:
    """Verify pos_embed is None for RoPE models and exists for standard ones."""

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

    @pytest.fixture(params=ALL_KEYS)
    def fresh_encoder(self, request):
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
    """Test forward pass without masking for all model types."""

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
# Forward (with masking) — string path
# ============================================================================


@pytest.mark.unit
@pytest.mark.download
class TestForwardWithMask:
    """Test forward pass with masking (encoder created from model name)."""

    def test_output_shape_masked(self, encoder_with_mask, sample_images):
        enc, cfg = encoder_with_mask
        enc.train()
        output = enc(sample_images)

        grid_h, grid_w = enc.default_grid_h, enc.default_grid_w
        num_patches = grid_h * grid_w
        num_visible = num_patches - int(num_patches * MASK_RATIO)
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

    def test_prefix_strip_matches_ids_keep(self, encoder_with_mask, sample_images):
        """Encoded patches after stripping prefix must match ids_keep length."""
        enc, _ = encoder_with_mask
        enc.train()
        output = enc(sample_images)
        encoded_patches = output.encoded[:, enc.num_prefix_tokens :]
        assert encoded_patches.shape[1] == output.ids_keep.shape[1], (
            f"encoded_patches dim 1 = {encoded_patches.shape[1]}, "
            f"ids_keep dim 1 = {output.ids_keep.shape[1]}"
        )


# ============================================================================
# Forward (with masking) — pre-instantiated model path
# ============================================================================


@pytest.mark.unit
@pytest.mark.download
class TestForwardWithMaskPreInstantiated:
    """Test forward with masking using a pre-instantiated timm model."""

    def test_output_shape(self, encoder_from_model):
        enc, cfg = encoder_from_model
        enc.train()
        s = cfg["img_size"]
        images = torch.randn(BATCH_SIZE, CHANNELS, s, s)
        output = enc(images)

        grid_h, grid_w = enc.default_grid_h, enc.default_grid_w
        num_patches = grid_h * grid_w
        num_visible = num_patches - int(num_patches * MASK_RATIO)
        num_prefix = _actual_prefix_count(enc)

        assert output.encoded.shape == (
            BATCH_SIZE,
            num_prefix + num_visible,
            enc.embed_dim,
        )
        assert output.ids_keep.shape == (BATCH_SIZE, num_visible)

    def test_prefix_strip_matches_ids_keep(self, encoder_from_model):
        enc, cfg = encoder_from_model
        enc.train()
        s = cfg["img_size"]
        images = torch.randn(BATCH_SIZE, CHANNELS, s, s)
        output = enc(images)
        encoded_patches = output.encoded[:, enc.num_prefix_tokens :]
        assert encoded_patches.shape[1] == output.ids_keep.shape[1], (
            f"Pre-instantiated {cfg['name']}: encoded_patches dim 1 = "
            f"{encoded_patches.shape[1]}, ids_keep dim 1 = {output.ids_keep.shape[1]}"
        )

    def test_no_nan(self, encoder_from_model):
        enc, cfg = encoder_from_model
        enc.train()
        s = cfg["img_size"]
        images = torch.randn(BATCH_SIZE, CHANNELS, s, s)
        output = enc(images)
        assert not torch.isnan(output.encoded).any()


# ============================================================================
# Gradient flow
# ============================================================================


@pytest.mark.unit
@pytest.mark.download
class TestGradientFlow:
    """Test gradients flow correctly for all model types."""

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
