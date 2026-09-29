"""Video geometry validation, multi-camera embeddings, and large-model factory contracts."""

import importlib
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from stable_pretraining.backbone.aggregator import TensorAggregator
from stable_pretraining.backbone.convmixer import ConvMixer
from stable_pretraining.backbone.video.causal_conv3d import CausalConv3d, _triple
from stable_pretraining.backbone.video.cosmos import (
    CosmosEncoder,
    CosmosCausalTemporalAttention,
)
from stable_pretraining.backbone.video.magvit2 import MAGVIT2Encoder
from stable_pretraining.backbone.video.norms import GroupNormPerFrame, _fit_groups
from stable_pretraining.backbone.video.predrnn import PredRNNv2, GHU
from stable_pretraining.backbone.video.recurrent_vit import RecurrentViT
from stable_pretraining.backbone.video.videomamba import VideoMamba

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "module_name,constructor,factory,width_key,width,depth",
    [
        ("cosmos", "CosmosEncoder", "cosmos_large", "base_channels", 192, 3),
        ("cosmos", "CosmosEncoder", "cosmos_huge", "base_channels", 256, 3),
        ("cosmos", "CosmosEncoder", "cosmos_giant", "base_channels", 384, 4),
        ("cosmos", "CosmosEncoder", "cosmos_gigantic", "base_channels", 512, 5),
        ("magvit2", "MAGVIT2Encoder", "magvit2_large", "base_channels", 192, 3),
        ("magvit2", "MAGVIT2Encoder", "magvit2_huge", "base_channels", 256, 4),
        ("magvit2", "MAGVIT2Encoder", "magvit2_giant", "base_channels", 384, 4),
        ("magvit2", "MAGVIT2Encoder", "magvit2_gigantic", "base_channels", 512, 6),
        ("predrnn", "PredRNNv2", "predrnn_v2_large", "hidden_channels", 192, 6),
        ("predrnn", "PredRNNv2", "predrnn_v2_huge", "hidden_channels", 256, 8),
        ("videomamba", "VideoMamba", "videomamba_base", "embed_dim", 576, 32),
        ("videomamba", "VideoMamba", "videomamba_large", "embed_dim", 1024, 32),
        ("videomamba", "VideoMamba", "videomamba_huge", "embed_dim", 1280, 48),
        ("videomamba", "VideoMamba", "videomamba_giant", "embed_dim", 1664, 48),
        ("videomamba", "VideoMamba", "videomamba_gigantic", "embed_dim", 2048, 64),
    ],
)
def test_large_model_factory_preserves_scale_and_caller_options(
    monkeypatch, module_name, constructor, factory, width_key, width, depth
):
    module = importlib.import_module(f"stable_pretraining.backbone.video.{module_name}")
    build = Mock()
    monkeypatch.setattr(module, constructor, build)
    assert getattr(module, factory)(in_channels=1) is build.return_value
    options = build.call_args.kwargs
    assert options[width_key] == width
    assert (
        options.get("n_res_blocks", options.get("num_layers", options.get("depth")))
        == depth
    )
    assert options["in_channels"] == 1


@pytest.mark.parametrize(
    "constructor,kwargs,match",
    [
        (CosmosEncoder, {"channel_multipliers": ()}, "non-empty"),
        (CosmosEncoder, {"global_pool": "bad"}, "global_pool"),
        (CosmosCausalTemporalAttention, {"channels": 7, "num_heads": 2}, "divisible"),
        (MAGVIT2Encoder, {"channel_multipliers": ()}, "non-empty"),
        (MAGVIT2Encoder, {"global_pool": "bad"}, "global_pool"),
        (GHU, {"channels": 4, "kernel_size": 2}, "odd"),
        (PredRNNv2, {"num_layers": 0}, "num_layers"),
        (PredRNNv2, {"global_pool": "bad"}, "global_pool"),
        (PredRNNv2, {"patch_size": 0}, "patch_size"),
        (
            RecurrentViT,
            {
                "img_size": 9,
                "patch_size": 4,
                "embed_dim": 8,
                "num_heads": 2,
                "spatial_depth": 1,
            },
            "divisible",
        ),
        (RecurrentViT, {"global_pool": "bad"}, "global_pool"),
        (RecurrentViT, {"max_cams": 0}, "max_cams"),
        (RecurrentViT, {"embed_dim": 7, "num_heads": 2}, "divisible"),
        (VideoMamba, {"global_pool": "bad"}, "global_pool"),
        (VideoMamba, {"global_pool": "token", "class_token": False}, "class_token"),
        (
            VideoMamba,
            {
                "embed_dim": 12,
                "depth": 1,
                "img_size": 8,
                "num_frames": 2,
                "patch_size": 2,
                "pos_embed_type": "bad",
            },
            "pos_embed_type",
        ),
    ],
)
def test_invalid_video_geometry_is_rejected(constructor, kwargs, match):
    with pytest.raises(ValueError, match=match):
        constructor(**kwargs)


def test_multicamera_recurrent_vit_defaults_to_camera_zero_and_backpropagates():
    model = RecurrentViT(
        img_size=8, patch_size=4, embed_dim=8, num_heads=2, spatial_depth=1, max_cams=2
    )
    x = torch.randn(2, 3, 2, 8, 8)
    default = model(x)
    explicit = model(x, cam_id=torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(default.pooled, explicit.pooled)
    other = model(x, cam_id=torch.ones(2, dtype=torch.long))
    assert not torch.allclose(other.pooled, default.pooled)
    other.pooled.sum().backward()
    assert model.cam_embed.grad[1].abs().sum() > 0
    assert model.cam_embed.grad[0].eq(0).all()
    with pytest.raises(ValueError, match="cam_id"):
        model(x, cam_id=torch.zeros(2, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="video must"):
        model(torch.ones(2, 3, 8, 8))


@pytest.mark.parametrize("position", ["none", "sincos_3d"])
def test_mamba_position_modes_propagate_gradients(position):
    model = VideoMamba(
        img_size=8,
        num_frames=2,
        patch_size=2,
        embed_dim=12,
        depth=1,
        pos_embed_type=position,
    )
    x = torch.randn(1, 3, 2, 8, 8, requires_grad=True)
    model(x).pooled.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert (
        model.pos_embed is None
        if position == "none"
        else model.pos_embed.shape[-1] == 12
    )


def test_video_norm_handles_images_and_non_power_of_two_channels():
    assert _fit_groups(8, 10) == 5
    model = GroupNormPerFrame(2, 4)
    x = torch.randn(2, 4, 3, 3)
    torch.testing.assert_close(
        model(x), nn.functional.group_norm(x, 2, model.weight, model.bias, model.eps)
    )
    conv = CausalConv3d(2, 3, 3, bias=True)
    assert conv.bias is conv.conv.bias
    with pytest.raises(ValueError, match="length-3"):
        _triple((1, 2))


def test_convmixer_learns_with_nondefault_input_channels():
    model = ConvMixer(
        in_channels=1, num_classes=3, dim=8, depth=2, kernel_size=3, patch_size=2
    )
    x = torch.randn(2, 1, 8, 8, requires_grad=True)
    output = model(x)
    assert output.shape == (2, 3)
    output.square().mean().backward()
    assert x.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("mode", ["mean", "max", "flatten", "adaptive"])
def test_video_aggregation_dimension_matches_real_output(mode):
    model = TensorAggregator(mode, adaptive_pool_size=2)
    x = torch.randn(2, 3, 4, 4, 4)
    assert model.compute_output_dim(tuple(x.shape[1:])) == model(x).shape[1]
    assert model.compute_output_dim((3,)) == 3
    assert "TensorAggregator" in repr(model)
    with pytest.raises(TypeError, match="input_shapes"):
        model.compute_output_dim(3)
    with pytest.raises(ValueError, match="Cannot compute"):
        model.compute_output_dim((1, 2, 3, 4, 5))
