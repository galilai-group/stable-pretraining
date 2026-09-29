"""Prebuilt encoders support evaluation, local crops, and legacy attention APIs."""

import inspect
from unittest.mock import Mock

import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

from stable_pretraining import methods

pytestmark = pytest.mark.unit


def _encoder(**kwargs):
    return VisionTransformer(
        img_size=32,
        patch_size=8,
        embed_dim=24,
        depth=2,
        num_heads=3,
        num_classes=0,
        **kwargs,
    )


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("CMAE", {"projector_dim": 8}),
        ("MoCov2", {"projector_dims": (16, 8), "queue_length": 8}),
        ("MoCov3", {"projector_dims": (16, 8), "predictor_hidden_dim": 16}),
        (
            "NNCLR",
            {"projector_dims": (16, 8), "queue_length": 8, "predictor_hidden_dim": 16},
        ),
        ("PIRL", {"projector_dim": 8, "queue_length": 8}),
        ("SimSiam", {"projector_dim": 8, "predictor_hidden_dim": 16}),
        ("TiCO", {"projector_dims": (16, 8)}),
        ("VICRegL", {"projector_dim": 16}),
        ("Data2Vec", {"top_k_blocks": 1}),
        ("MaskFeat", {}),
        ("SimMIM", {}),
        ("iGPT", {}),
        ("DINO", {"n_prototypes": 8}),
        ("DINOv2", {"n_cls_prototypes": 8, "n_patch_prototypes": 8}),
        (
            "DINOv3",
            {"n_cls_prototypes": 8, "n_patch_prototypes": 8, "n_register_tokens": 2},
        ),
        ("iBOT", {"n_cls_prototypes": 8, "n_patch_prototypes": 8}),
        ("SwAV", {"n_prototypes": 8, "projector_dims": (16, 8)}),
    ],
)
def test_evaluation_with_prebuilt_encoder_returns_finite_features_and_zero_loss(
    name, kwargs
):
    cls = getattr(methods, name)
    parameters = inspect.signature(cls).parameters
    config = dict(kwargs)
    if "image_size" in parameters:
        config["image_size"] = 32
    if "patch_size" in parameters:
        config["patch_size"] = 8
    model = cls(encoder_name=_encoder(dynamic_img_size=True), **config).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = (
            model(images=images)
            if name in ("DINO", "DINOv2", "DINOv3", "iBOT", "SwAV")
            else model(images)
        )
    assert output.embedding.shape == (2, 24)
    assert torch.isfinite(output.embedding).all()
    assert output.loss.item() == 0
    assert not output.embedding.requires_grad


@pytest.mark.parametrize("name", ["DINO", "DINOv2", "DINOv3"])
def test_local_crops_receive_gradients_and_teacher_remains_detached(name):
    kwargs = (
        {"n_prototypes": 8}
        if name == "DINO"
        else {"n_cls_prototypes": 8, "n_patch_prototypes": 8, "mask_ratio": 0.5}
    )
    if name != "DINO":
        kwargs["image_size"] = 32
    model = getattr(methods, name)(
        encoder_name=_encoder(dynamic_img_size=True), **kwargs
    )
    global_views = [torch.randn(2, 3, 32, 32) for _ in range(2)]
    local = torch.randn(2, 3, 16, 16, requires_grad=True)
    output = model(global_views=global_views, local_views=[local])
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert local.grad is not None and local.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.teacher.parameters())
    with pytest.raises(ValueError, match="global_views"):
        model()


@pytest.mark.parametrize("cls_token", [False, True])
def test_igpt_legacy_attention_fallback_preserves_causality(monkeypatch, cls_token):
    encoder = _encoder(
        class_token=cls_token, global_pool="avg" if not cls_token else "token"
    )
    model = methods.iGPT(encoder_name=encoder, patch_size=8, image_size=32)
    images = torch.randn(2, 3, 32, 32)
    expected = model._causal_forward_features(images)
    changed = images.clone()
    changed[:, :, -8:, -8:] += 20
    torch.testing.assert_close(
        model._causal_forward_features(changed)[:, :-1], expected[:, :-1]
    )
    for block in encoder.blocks:
        monkeypatch.setattr(
            block, "forward", Mock(side_effect=TypeError("unexpected attn_mask"))
        )
    actual = model._causal_forward_features(images)
    torch.testing.assert_close(actual, expected)
    changed = images.clone()
    changed[:, :, -8:, -8:] += 20
    torch.testing.assert_close(
        model._causal_forward_features(changed)[:, :-1], actual[:, :-1]
    )
    loss = model(images).loss
    loss.backward()
    assert encoder.patch_embed.proj.weight.grad.abs().sum() > 0


def test_mim_refiner_freezes_only_requested_lower_student_blocks():
    model = methods.MIMRefiner(
        pretrained_encoder=_encoder(),
        freeze_lower_blocks=1,
        image_size=32,
        n_cls_prototypes=8,
        n_patch_prototypes=8,
    )
    assert all(
        not p.requires_grad for p in model.backbone.student.blocks[0].parameters()
    )
    assert all(p.requires_grad for p in model.backbone.student.blocks[1].parameters())
    assert all(not p.requires_grad for p in model.backbone.teacher.parameters())
