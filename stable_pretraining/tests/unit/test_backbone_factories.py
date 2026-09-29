"""Factory options must preserve requested channels, heads, and loading modes."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from stable_pretraining.backbone import utils

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("low_resolution", [False, True])
@pytest.mark.parametrize("classes", [None, 5])
def test_torchvision_resnet_channels_and_output_shape(low_resolution, classes):
    kwargs = {} if classes is None else {"num_classes": classes}
    model = utils.from_torchvision(
        "resnet18", low_resolution=low_resolution, in_channels=1, weights=None, **kwargs
    ).eval()
    assert model.conv1.in_channels == 1
    assert model.conv1.kernel_size == ((3, 3) if low_resolution else (7, 7))
    with torch.no_grad():
        assert model(torch.randn(2, 1, 32, 32)).shape == (
            2,
            512 if classes is None else classes,
        )
    if low_resolution:
        assert isinstance(model.maxpool, nn.Identity)


@pytest.mark.parametrize("container", [False, True])
@pytest.mark.parametrize("classes", [None, 5])
def test_classifier_factory_replaces_only_final_layer(monkeypatch, container, classes):
    model = nn.Module()
    first = nn.Linear(4, 4)
    model.classifier = (
        nn.Sequential(first, nn.Linear(4, 8)) if container else nn.Linear(4, 8)
    )
    factory = Mock(return_value=model)
    monkeypatch.setitem(utils.torchvision.models.__dict__, "tiny_classifier", factory)
    result = utils.from_torchvision(
        "tiny_classifier",
        low_resolution=True,
        **({"num_classes": classes} if classes else {}),
    )
    assert result is model
    assert model.classifier(torch.ones(2, 4)).shape == (2, classes or 4)
    if container:
        assert model.classifier[0] is first


def test_unknown_torchvision_model_reports_name():
    with pytest.raises(ValueError, match="Unknown model: missing_model"):
        utils.from_torchvision("missing_model")


@pytest.mark.parametrize("pretrained", [False, True])
def test_huggingface_factory_forwards_loading_options(monkeypatch, pretrained):
    from transformers import AutoConfig, AutoModel

    backbone = nn.Linear(2, 2)
    model = SimpleNamespace(base_model=backbone)
    cfg = object()
    config_loader = Mock(return_value=cfg)
    weights_loader = Mock(return_value=model)
    constructor = Mock(return_value=model)
    monkeypatch.setattr(AutoConfig, "from_pretrained", config_loader)
    monkeypatch.setattr(AutoModel, "from_pretrained", weights_loader)
    monkeypatch.setattr(AutoModel, "from_config", constructor)
    assert (
        utils.from_huggingface(
            "local-model",
            pretrained,
            attn_implementation="eager",
            local_files_only=True,
        )
        is backbone
    )
    if pretrained:
        weights_loader.assert_called_once_with(
            "local-model", attn_implementation="eager", local_files_only=True
        )
        config_loader.assert_not_called()
    else:
        config_loader.assert_called_once_with("local-model", local_files_only=True)
        constructor.assert_called_once_with(cfg, attn_implementation="eager")
        weights_loader.assert_not_called()


@pytest.mark.parametrize("name", ["resnet18", "tiny_transformer"])
def test_timm_low_resolution_adaptation_does_not_replace_unrelated_layers(
    monkeypatch, name
):
    import timm

    model = nn.Module()
    model.conv1 = nn.Conv2d(3, 64, 7, stride=2)
    model.maxpool = nn.MaxPool2d(2)
    model.head = nn.Linear(64, 2)
    head = model.head
    create = Mock(return_value=model)
    monkeypatch.setattr(timm, "create_model", create)
    assert utils.from_timm(name, low_resolution=True, pretrained=False) is model
    assert model.head is head
    assert model.conv1.kernel_size == ((3, 3) if name.startswith("resnet") else (7, 7))


def test_shape_inference_accepts_namedtuple_and_set_inputs():
    from collections import namedtuple

    Pair = namedtuple("Pair", "tensor label")
    shapes = utils.get_output_shape(
        nn.Identity(), (Pair(torch.zeros(3, 2), "pair"), {torch.zeros(2, 4)})
    )
    assert shapes == (Pair(torch.Size([3, 2]), "pair"), {torch.Size([2, 4])})


@pytest.mark.parametrize("norm", [False, True])
def test_legacy_masked_vit_head_uses_cls_token(norm):
    model = nn.Module()
    model.patch_embed = nn.Linear(2, 2)
    model.blocks = nn.Identity()
    model.head = nn.Linear(2, 3)
    if norm:
        model.fc_norm = nn.LayerNorm(2)
    wrapper = utils.EfficientMaskedTimmViT(model)
    tokens = torch.randn(2, 4, 2)
    expected = model.head(model.fc_norm(tokens[:, 0]) if norm else tokens[:, 0])
    torch.testing.assert_close(wrapper._apply_head(tokens), expected)
    del model.head
    torch.testing.assert_close(wrapper._apply_head(tokens), tokens)


@pytest.mark.parametrize("missing", ["patch_embed", "blocks"])
def test_masked_vit_rejects_incompatible_backbone(missing):
    model = nn.Module()
    if missing == "blocks":
        model.patch_embed = nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match=missing):
        utils.EfficientMaskedTimmViT(model)
