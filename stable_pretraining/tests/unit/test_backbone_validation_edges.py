"""Backbone compatibility paths retain model semantics and useful errors."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer

from stable_pretraining.backbone import decoders, pos_embed, utils
from stable_pretraining.backbone.mlp import MLP
from stable_pretraining.backbone.probe import AutoTuneMLP
from stable_pretraining.backbone.vit import MaskedEncoder, ViT

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("training", [True, False])
def test_legacy_distillation_head_uses_both_tokens(training):
    model = nn.Module()
    model.patch_embed = nn.Linear(2, 2)
    model.blocks = nn.Identity()
    model.head, model.head_dist = nn.Linear(2, 3), nn.Linear(2, 3)
    model.dist_token = nn.Parameter(torch.ones(1, 1, 2))
    wrapper = utils.EfficientMaskedTimmViT(model).train(training)
    tokens = torch.randn(2, 4, 2)
    expected = model.head(tokens[:, 0]), model.head_dist(tokens[:, 1])
    result = wrapper._apply_head(tokens)
    torch.testing.assert_close(
        result, expected if training else (expected[0] + expected[1]) / 2
    )
    assert wrapper._get_num_pos_tokens() == 1


def test_masked_image_patch_embedding_keeps_gradients_finite():
    vit = VisionTransformer(
        img_size=8, patch_size=4, embed_dim=8, depth=1, num_heads=2, num_classes=0
    )
    wrapper = utils.EfficientMaskedTimmViT(vit)
    images = torch.randn(2, 3, 8, 8)
    images[:, :, :4, :4] = float("nan")
    result = wrapper(images)
    assert result.shape == (2, 8) and torch.isfinite(result).all()
    result.sum().backward()
    assert torch.isfinite(vit.patch_embed.proj.weight.grad).all()


@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 8, 8)])
def test_masked_wrapper_reports_incompatible_input_or_patch_output(shape):
    model = nn.Module()
    model.patch_embed = nn.Flatten(1)
    model.blocks = nn.Identity()
    wrapper = utils.EfficientMaskedTimmViT(model)
    with pytest.raises(ValueError, match="(Input must|patch_embed output)"):
        wrapper(torch.full(shape, float("nan")))


def test_pretrained_hf_factory_forwards_mask_token_request(monkeypatch):
    loaded = SimpleNamespace(config=SimpleNamespace())
    factory = Mock(return_value=loaded)
    monkeypatch.setattr(utils.ViTModel, "from_pretrained", factory)
    assert (
        utils.vit_hf(
            "tiny", pretrained=True, use_mask_token=True, image_size=32, patch_size=4
        )
        is loaded
    )
    factory.assert_called_once_with(
        "google/vit-tiny-patch4-32", add_pooling_layer=False, use_mask_token=True
    )
    monkeypatch.setattr(utils, "_TRANSFORMERS_AVAILABLE", False)
    with pytest.raises(ImportError, match="transformers"):
        utils.vit_hf("tiny")


def test_lazy_mlp_infers_features_and_unknown_probe_options_fall_back():
    model = MLP(None, [8, 2])
    assert model(torch.ones(3, 4)).shape == (3, 2)
    assert model[0].in_features == 4
    probe = AutoTuneMLP(
        4,
        2,
        [8],
        "probe",
        nn.CrossEntropyLoss(),
        normalization="unknown",
        activation="unknown",
    )
    assert len(probe.mlp) == 1
    assert any(isinstance(layer, nn.Identity) for layer in probe.modules())


@pytest.mark.parametrize("image_size", [2, 6])
def test_cnn_decoder_rejects_incompatible_start_grid(image_size):
    with pytest.raises(ValueError, match="multiple"):
        decoders.CNNImageDecoder(4, image_size, start_size=4)


def test_decoder_validates_embedding_and_cnn_ignores_patch_size():
    decoder = decoders.ViTImageDecoder(8, 8, 4, decoder_dim=8, num_heads=2, depth=1)
    with pytest.raises(ValueError, match="D=8"):
        decoder(torch.zeros(2, 4, 3))
    cnn = decoders.build_image_decoder(
        8,
        (3, 8, 8),
        kind="cnn",
        patch_size=4,
        decoder_kwargs={"base_channels": 8, "min_channels": 4, "num_res_blocks": 1},
    )
    assert cnn(torch.ones(2, 8)).shape == (2, 3, 8, 8)


@pytest.mark.parametrize("dim,length", [(0, 4), (4, 0)])
def test_position_embeddings_reject_nonpositive_geometry(dim, length):
    with pytest.raises(ValueError, match="positive"):
        pos_embed.get_1d_sincos_pos_embed(dim, length)
    with pytest.raises(ValueError, match="positive"):
        pos_embed.get_2d_sincos_pos_embed(8, 0, 2)


@pytest.mark.parametrize("dynamic", [False, True])
def test_masked_encoder_variable_resolution_and_feature_extraction(dynamic):
    vit = VisionTransformer(
        img_size=8,
        patch_size=4,
        embed_dim=8,
        depth=1,
        num_heads=2,
        num_classes=3,
        dynamic_img_size=dynamic,
    )
    encoder = MaskedEncoder(vit, dynamic_img_size=True)
    assert isinstance(vit.head, nn.Identity)
    if not dynamic:
        vit.patch_embed.strict_img_size = False
    features = encoder.forward_features(torch.randn(2, 3, 12, 12))
    assert features.shape == (2, 10, 8) and encoder.training
    assert "embed_dim=8" in repr(encoder)


def test_native_vit_without_positions_supports_resized_inputs():
    model = ViT(
        img_size=8,
        patch_size=4,
        embed_dim=8,
        depth=1,
        num_heads=2,
        num_classes=0,
        pos_embed_type="none",
    )
    model.patch_embed.strict_img_size = False
    assert model(torch.randn(2, 3, 12, 12)).shape == (2, 8)
