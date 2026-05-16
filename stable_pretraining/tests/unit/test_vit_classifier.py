"""Tests for the standalone ViT class and timm/torchvision/HF-matching factories.

Run with: pytest test_vit_classifier.py -v -m unit
"""

import pytest
import torch

from stable_pretraining.backbone import (
    ViT,
    vit_tiny_patch16_224,
    vit_tiny_patch16_384,
    vit_small_patch32_224,
    vit_small_patch32_384,
    vit_small_patch16_224,
    vit_small_patch16_384,
    vit_small_patch14_224,
    vit_small_patch8_224,
    vit_base_patch32_224,
    vit_base_patch32_384,
    vit_base_patch16_224,
    vit_base_patch16_384,
    vit_base_patch14_224,
    vit_base_patch8_224,
    vit_large_patch32_224,
    vit_large_patch32_384,
    vit_large_patch16_224,
    vit_large_patch16_384,
    vit_large_patch14_224,
    vit_huge_patch14_224,
    vit_huge_patch16_224,
    vit_giant_patch14_224,
    vit_gigantic_patch14_224,
)


# (factory, img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio)
# The first 5 columns map directly to the factory name; mlp_ratio=4 except for
# giant (48/11) and gigantic (64/13) which match the standard CLIP/SigLIP recipe.
FACTORY_SPECS = [
    (vit_tiny_patch16_224, 224, 16, 192, 12, 3, 4.0),
    (vit_tiny_patch16_384, 384, 16, 192, 12, 3, 4.0),
    (vit_small_patch32_224, 224, 32, 384, 12, 6, 4.0),
    (vit_small_patch32_384, 384, 32, 384, 12, 6, 4.0),
    (vit_small_patch16_224, 224, 16, 384, 12, 6, 4.0),
    (vit_small_patch16_384, 384, 16, 384, 12, 6, 4.0),
    (vit_small_patch14_224, 224, 14, 384, 12, 6, 4.0),
    (vit_small_patch8_224, 224, 8, 384, 12, 6, 4.0),
    (vit_base_patch32_224, 224, 32, 768, 12, 12, 4.0),
    (vit_base_patch32_384, 384, 32, 768, 12, 12, 4.0),
    (vit_base_patch16_224, 224, 16, 768, 12, 12, 4.0),
    (vit_base_patch16_384, 384, 16, 768, 12, 12, 4.0),
    (vit_base_patch14_224, 224, 14, 768, 12, 12, 4.0),
    (vit_base_patch8_224, 224, 8, 768, 12, 12, 4.0),
    (vit_large_patch32_224, 224, 32, 1024, 24, 16, 4.0),
    (vit_large_patch32_384, 384, 32, 1024, 24, 16, 4.0),
    (vit_large_patch16_224, 224, 16, 1024, 24, 16, 4.0),
    (vit_large_patch16_384, 384, 16, 1024, 24, 16, 4.0),
    (vit_large_patch14_224, 224, 14, 1024, 24, 16, 4.0),
    (vit_huge_patch14_224, 224, 14, 1280, 32, 16, 4.0),
    (vit_huge_patch16_224, 224, 16, 1280, 32, 16, 4.0),
    (vit_giant_patch14_224, 224, 14, 1408, 40, 16, 48 / 11),
    (vit_gigantic_patch14_224, 224, 14, 1664, 48, 16, 64 / 13),
]


# Subset of factories that have a corresponding timm preset with the exact
# same recipe — used to assert parameter-count parity with timm.
TIMM_PARITY = [
    ("vit_tiny_patch16_224", vit_tiny_patch16_224),
    ("vit_small_patch16_224", vit_small_patch16_224),
    ("vit_small_patch32_224", vit_small_patch32_224),
    ("vit_base_patch16_224", vit_base_patch16_224),
    ("vit_base_patch32_224", vit_base_patch32_224),
    ("vit_large_patch16_224", vit_large_patch16_224),
    ("vit_huge_patch14_224", vit_huge_patch14_224),
]


@pytest.mark.unit
class TestViTConstruction:
    """Tests for :class:`ViT` constructor argument handling."""

    def test_default_no_head(self):
        m = ViT(num_classes=0)
        assert m.embed_dim == 768
        assert len(m.blocks) == 12
        assert m.num_classes == 0
        assert isinstance(m.head, torch.nn.Identity)

    def test_with_head(self):
        m = ViT(num_classes=10)
        assert m.num_classes == 10
        assert isinstance(m.head, torch.nn.Linear)
        assert m.head.out_features == 10
        assert m.head.in_features == m.embed_dim

    def test_register_tokens(self):
        m = ViT(num_reg_tokens=4)
        assert m.num_reg_tokens == 4
        assert m.num_prefix_tokens == 1 + 4
        assert m.reg_token.shape == (1, 4, m.embed_dim)

    def test_no_class_token(self):
        m = ViT(class_token=False, global_pool="avg")
        assert m.cls_token is None
        assert m.num_prefix_tokens == 0

    def test_grid_size_inferred(self):
        m = ViT(img_size=224, patch_size=16)
        assert m.grid_size == (14, 14)
        m = ViT(img_size=224, patch_size=14)
        assert m.grid_size == (16, 16)

    def test_pos_embed_shape_learned(self):
        m = ViT(img_size=224, patch_size=16, num_reg_tokens=2)
        # 1 cls + 2 reg + 14*14 patches = 199
        assert m.pos_embed.shape == (1, 1 + 2 + 196, m.embed_dim)

    def test_pos_embed_sincos_2d_is_buffer(self):
        m = ViT(pos_embed_type="sincos_2d")
        assert "pos_embed" in dict(m.named_buffers())
        assert m.pos_embed.shape == (1, 1 + 196, m.embed_dim)

    def test_pos_embed_none_with_rope(self):
        # RoPE forces pos_embed to None regardless of pos_embed_type
        m = ViT(
            class_token=False,
            global_pool="avg",
            pos_embed_type="learned",
            use_rope="2d",
        )
        assert m.pos_embed is None
        assert m.use_rope


@pytest.mark.unit
class TestViTForwardShapes:
    """Tests for :class:`ViT` forward output shapes across pooling modes."""

    def test_default_pooled_feature(self):
        m = ViT(num_classes=0)
        y = m(torch.randn(2, 3, 224, 224))
        assert y.shape == (2, 768)

    def test_classifier_logits(self):
        m = ViT(num_classes=10)
        y = m(torch.randn(2, 3, 224, 224))
        assert y.shape == (2, 10)

    def test_token_output_no_pool(self):
        m = ViT(num_classes=0, global_pool="")
        y = m(torch.randn(2, 3, 224, 224))
        assert y.shape == (2, 1 + 196, 768)

    def test_avg_pool_no_cls(self):
        m = ViT(num_classes=0, class_token=False, global_pool="avg")
        y = m(torch.randn(2, 3, 224, 224))
        assert y.shape == (2, 768)

    def test_avg_token_pool(self):
        m = ViT(num_classes=10, global_pool="avg_token")
        y = m(torch.randn(2, 3, 224, 224))
        assert y.shape == (2, 10)

    def test_with_register_tokens(self):
        m = ViT(num_reg_tokens=4, global_pool="")
        y = m(torch.randn(2, 3, 224, 224))
        assert y.shape == (2, 1 + 4 + 196, 768)

    def test_forward_features_then_head(self):
        m = ViT(num_classes=10)
        feats = m.forward_features(torch.randn(2, 3, 224, 224))
        assert feats.shape == (2, 1 + 196, 768)
        logits = m.forward_head(feats)
        assert logits.shape == (2, 10)


@pytest.mark.unit
class TestViTValidation:
    """Tests for :class:`ViT` constructor input validation."""

    def test_token_pool_requires_class_token(self):
        with pytest.raises(ValueError, match="global_pool='token' requires"):
            ViT(class_token=False, global_pool="token")

    def test_head_requires_pool(self):
        with pytest.raises(ValueError, match="num_classes > 0 requires global_pool"):
            ViT(num_classes=10, global_pool="")

    def test_invalid_global_pool(self):
        with pytest.raises(ValueError, match="global_pool must be one of"):
            ViT(global_pool="bogus")

    def test_invalid_pos_embed_type(self):
        with pytest.raises(ValueError, match="pos_embed_type must be"):
            ViT(pos_embed_type="bogus")

    def test_dim_must_divide_heads(self):
        # num_heads=7 doesn't divide 768 — surfaced by Attention.__init__
        with pytest.raises(ValueError, match="must be divisible by num_heads"):
            ViT(embed_dim=768, num_heads=7)


@pytest.mark.unit
class TestViTGradients:
    """Tests for :class:`ViT` gradient flow through the architecture."""

    def test_backward_classifier(self):
        m = ViT(num_classes=10)
        y = m(torch.randn(2, 3, 224, 224))
        y.sum().backward()
        for name, p in m.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"no grad on {name}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory,img_size,patch_size,embed_dim,depth,num_heads,mlp_ratio",
    FACTORY_SPECS,
    ids=[f.__name__ for f, *_ in FACTORY_SPECS],
)
class TestFactoryConfigs:
    """Each preset has the right architecture and produces the right shape."""

    def test_architecture(
        self, factory, img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio
    ):
        m = factory(num_classes=0)
        assert m.embed_dim == embed_dim
        assert len(m.blocks) == depth
        assert m.blocks[0].attn.num_heads == num_heads
        # mlp_ratio surfaces via the MLP hidden dim (gelu MLP path)
        expected_hidden = int(embed_dim * mlp_ratio)
        # timm's Mlp keeps fc1.out_features = hidden
        assert m.blocks[0].mlp.fc1.out_features == expected_hidden
        assert m.patch_size == (patch_size, patch_size)
        expected_grid = img_size // patch_size
        assert m.grid_size == (expected_grid, expected_grid)

    def test_forward_shape(
        self, factory, img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio
    ):
        # Skip the very large variants in the forward test to keep CI fast.
        if embed_dim * depth >= 1280 * 32:
            pytest.skip("skip huge/giant/gigantic forward to keep tests fast")
        m = factory(num_classes=0)
        y = m(torch.randn(1, 3, img_size, img_size))
        assert y.shape == (1, embed_dim)


@pytest.mark.unit
@pytest.mark.parametrize("name,factory", TIMM_PARITY, ids=[n for n, _ in TIMM_PARITY])
def test_param_count_matches_timm(name, factory):
    """Total trainable param count matches timm's preset with the same recipe."""
    timm = pytest.importorskip("timm")
    ours = factory(num_classes=0)
    theirs = timm.create_model(name, pretrained=False, num_classes=0)
    n_ours = sum(p.numel() for p in ours.parameters())
    n_theirs = sum(p.numel() for p in theirs.parameters())
    # Timm sometimes adds a no-op pre_logits or other 0-param layers, but
    # parameter totals must match exactly.
    assert n_ours == n_theirs, (
        f"{name}: ours={n_ours / 1e6:.3f}M  timm={n_theirs / 1e6:.3f}M"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,factory",
    [
        ("vit_tiny_patch16_224", vit_tiny_patch16_224),
        ("vit_small_patch16_224", vit_small_patch16_224),
        ("vit_base_patch16_224", vit_base_patch16_224),
    ],
)
class TestTimmCheckpointLoading:
    """A timm ViT state_dict loads cleanly and produces bit-identical outputs."""

    def test_state_dict_keys_match_timm(self, name, factory):
        """Default ViT shares the exact same state_dict key set as timm."""
        timm = pytest.importorskip("timm")
        ours = factory(num_classes=0)
        theirs = timm.create_model(name, pretrained=False, num_classes=0)
        assert set(ours.state_dict().keys()) == set(theirs.state_dict().keys())

    def test_load_state_dict_clean(self, name, factory):
        """Plain load_state_dict works — no key remapping needed."""
        timm = pytest.importorskip("timm")
        ours = factory(num_classes=0)
        theirs = timm.create_model(name, pretrained=False, num_classes=0)
        result = ours.load_state_dict(theirs.state_dict(), strict=True)
        assert list(result.missing_keys) == []
        assert list(result.unexpected_keys) == []

    def test_outputs_match_timm_exactly(self, name, factory):
        timm = pytest.importorskip("timm")
        torch.manual_seed(0)
        ours = factory(num_classes=0).eval()
        theirs = timm.create_model(name, pretrained=False, num_classes=0).eval()
        ours.load_state_dict(theirs.state_dict(), strict=True)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            y_ours = ours(x)
            # timm forward_features returns the full token sequence after
            # the final LayerNorm; the CLS token is at position 0.
            y_theirs = theirs.forward_features(x)[:, 0]
        # Same weights, same eps, same architecture → bit-for-bit equal.
        assert torch.equal(y_ours, y_theirs), (
            f"{name}: maxdiff={(y_ours - y_theirs).abs().max().item()}"
        )

    def test_layer_norm_eps_matches_timm(self, name, factory):
        ours = factory(num_classes=0)
        for mod in ours.modules():
            if isinstance(mod, torch.nn.LayerNorm):
                assert mod.eps == 1e-6, (
                    f"LayerNorm eps={mod.eps} != 1e-6 (timm default)"
                )
