"""SSL methods preserve evaluation and validation semantics at API boundaries."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer

from stable_pretraining import methods
from stable_pretraining.methods.data2vec import _BlockHook

pytestmark = pytest.mark.unit


def encoder(has_cls=True):
    return VisionTransformer(
        img_size=32,
        patch_size=8,
        embed_dim=24,
        depth=2,
        num_heads=3,
        num_classes=0,
        class_token=has_cls,
        global_pool="token" if has_cls else "avg",
        dynamic_img_size=True,
    )


@pytest.mark.parametrize(
    "name", ["BarlowTwins", "SimCLR", "VICReg", "WMSE", "BYOL", "TWIST"]
)
def test_joint_embedding_methods_accept_existing_encoder_for_evaluation(name):
    kwargs = {"projector_dims": (16, 8)}
    if name == "BYOL":
        kwargs["predictor_dims"] = (16, 8)
    model = getattr(methods, name)(encoder_name=encoder(), **kwargs).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(images)
    assert output.embedding.shape == (2, 24) and torch.isfinite(output.embedding).all()
    assert output.loss.item() == 0


@pytest.mark.parametrize(
    "name,fn",
    [
        ("simclr", "_build_projector"),
        ("barlow_twins", "_build_barlow_projector"),
        ("vicreg", "_build_vicreg_projector"),
        ("twist", "_build_twist_projector"),
    ],
)
def test_projector_requires_an_output_dimension(name, fn):
    with pytest.raises(ValueError, match="at least one"):
        getattr(importlib.import_module(f"stable_pretraining.methods.{name}"), fn)(
            4, []
        )


@pytest.mark.parametrize("projector,predictor", [((8,), (8, 4)), ((8, 4), (8, 3))])
def test_byol_rejects_incompatible_prediction_head(projector, predictor):
    with pytest.raises(ValueError, match="(tuples|must match)"):
        methods.BYOL(
            encoder_name=encoder(), projector_dims=projector, predictor_dims=predictor
        )


@pytest.mark.parametrize("has_cls", [False, True])
def test_beit_evaluation_with_existing_encoder_needs_no_tokenizer(has_cls):
    model = methods.BEiT(
        encoder_name=encoder(has_cls), patch_size=8, image_size=32, vocab_size=8
    ).eval()
    result = model(torch.randn(2, 3, 32, 32))
    assert result.embedding.shape == (2, 24) and result.loss.item() == 0


@pytest.mark.parametrize("name", ["CMAE", "VICRegL"])
def test_methods_without_cls_pool_all_patch_features(name):
    model = getattr(methods, name)(
        encoder_name=encoder(False), image_size=32, projector_dim=8
    ).eval()
    images = torch.randn(2, 3, 32, 32)
    result = model(images)
    assert result.embedding.shape == (2, 24)
    backbone = model.encoder if name == "VICRegL" else model.backbone.teacher
    expected = (
        backbone.forward_features(images).mean(1)
        if name == "VICRegL"
        else backbone(images)
    )
    torch.testing.assert_close(result.embedding, expected)
    if name == "CMAE":
        tokens = torch.randn(2, 4, 24)
        pooled, patches = model._split(tokens)
        torch.testing.assert_close(pooled, tokens.mean(1))
        assert patches is tokens


@pytest.mark.parametrize("name", ["dino", "msn"])
def test_cls_conversion_validates_rank(name):
    module = importlib.import_module(f"stable_pretraining.methods.{name}")
    features = torch.randn(2, 4, 8)
    torch.testing.assert_close(module._to_cls(features), features[:, 0])
    with pytest.raises(ValueError, match="shape"):
        module._to_cls(torch.ones(2, 3, 4, 5))


@pytest.mark.parametrize("name", ["ibot", "dinov2"])
def test_patch_only_encoders_keep_all_tokens(name):
    module = importlib.import_module(f"stable_pretraining.methods.{name}")
    features = torch.randn(2, 4, 8)
    cls, patches = module._split_cls_patches(features, False)
    torch.testing.assert_close(cls, features.mean(1))
    assert patches is features
    if name == "ibot":
        with pytest.raises(ValueError, match="sequence"):
            module._split_cls_patches(torch.ones(2, 3, 4, 5), True)


@pytest.mark.parametrize("name", ["DINO", "iBOT"])
def test_teacher_temperature_reaches_final_value_after_warmup(name):
    args = dict(
        encoder_name=encoder(), projector_hidden_dim=16, projector_bottleneck_dim=8
    )
    args.update(
        {"n_prototypes": 8}
        if name == "DINO"
        else {"n_cls_prototypes": 8, "n_patch_prototypes": 8, "image_size": 32}
    )
    model = getattr(methods, name)(**args)
    model._trainer = SimpleNamespace(
        current_epoch=model.warmup_epochs_temperature_teacher + 5
    )
    assert model._teacher_temperature() == model.temperature_teacher
    if name == "iBOT":
        with pytest.raises(ValueError, match="global_views"):
            model()


def test_dinov3_requires_cls_and_handles_pooled_features():
    from stable_pretraining.methods.dinov3 import _split_cls_patches

    features = torch.randn(2, 8)
    cls, patches = _split_cls_patches(features, 1)
    assert cls is features and patches is None
    with pytest.raises(ValueError, match="CLS-token"):
        methods.DINOv3(
            encoder_name=encoder(False),
            image_size=32,
            n_cls_prototypes=8,
            n_patch_prototypes=8,
        )


def test_shuffle_patches_resizes_nondivisible_square_and_rejects_rectangle():
    from stable_pretraining.methods.pirl import _shuffle_patches

    image = torch.rand(2, 3, 7, 7, requires_grad=True)
    result = _shuffle_patches(image, grid=3)
    assert result.shape == image.shape and torch.isfinite(result).all()
    result.sum().backward()
    assert image.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="square"):
        _shuffle_patches(torch.zeros(2, 3, 7, 8))


def test_block_hook_reset_discards_only_previous_features():
    blocks = nn.ModuleList([nn.Linear(4, 4)])
    hook = _BlockHook(blocks)
    first = blocks[0](torch.ones(2, 4))
    assert hook.outputs == [first]
    hook.reset()
    assert hook.outputs == []
    blocks[0](torch.ones(2, 4))
    assert len(hook.outputs) == 1
    hook.remove()
    assert not blocks[0]._forward_hooks


@pytest.mark.parametrize("name", ["LeJEPA", "VISReg"])
def test_distribution_regularized_methods_evaluate_without_views(monkeypatch, name):
    import timm

    backbone = encoder()
    monkeypatch.setattr(timm, "create_model", lambda *a, **kw: backbone)
    model = getattr(methods, name)(
        encoder_name="tiny", projector=nn.Linear(24, 8)
    ).eval()
    images = torch.randn(2, 3, 32, 32)
    result = model(images=images)
    torch.testing.assert_close(result.embedding, backbone(images))
    assert result.loss.item() == result.inv_loss.item() == 0
