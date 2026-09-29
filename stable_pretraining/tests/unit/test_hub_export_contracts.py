"""Hub export validates local artifacts without making network requests."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from stable_pretraining.utils import timm_to_hf_hub as export

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def no_network(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(export, "create_repo", Mock())
    monkeypatch.setattr(export, "upload_folder", Mock())


@pytest.mark.parametrize(
    "value,expected", [(16, (16, 16)), ([16], (16, 16)), ((8, 12), (8, 12))]
)
def test_image_dimensions(value, expected):
    assert export._normalize_img_size(value) == expected


@pytest.mark.parametrize(
    "family,structure",
    [
        ("vit", "tensor"),
        ("vit", "cls_token"),
        ("deit", "x"),
        ("vit", "x_norm_cls"),
        ("swin", "tokens"),
        ("swin", "x"),
        ("swin", "spatial"),
        ("convnext", "spatial"),
        ("convnext", "x"),
    ],
)
def test_timm_feature_extraction_matches_pooling_contract(family, structure):
    tokens = torch.arange(24.0).reshape(2, 3, 4)
    spatial = torch.arange(48.0).reshape(2, 4, 2, 3)
    expected = (
        tokens[:, 0]
        if family in ("vit", "deit")
        else tokens.mean(1)
        if family == "swin" and structure != "spatial"
        else spatial.mean((2, 3))
    )
    if structure in ("cls_token", "x_norm_cls"):
        output = {structure: tokens[:, 0]}
    elif structure == "x":
        output = {"x": spatial if family == "convnext" else tokens}
    else:
        output = spatial if structure == "spatial" else tokens
    model = SimpleNamespace(forward_features=lambda _: output)
    torch.testing.assert_close(
        export._extract_features_timm(model, torch.empty(0), family), expected
    )


@pytest.mark.parametrize("family", ["vit", "deit", "swin", "convnext"])
@pytest.mark.parametrize("pooled", [False, True])
def test_hf_feature_extraction_matches_pooling_contract(family, pooled):
    hidden = (
        torch.arange(48.0).reshape(2, 4, 2, 3)
        if family == "convnext"
        else torch.arange(24.0).reshape(2, 3, 4)
    )
    average = hidden.mean((2, 3)) if family == "convnext" else hidden.mean(1)
    output = SimpleNamespace(
        last_hidden_state=hidden, pooler_output=average if pooled else None
    )
    expected = hidden[:, 0] if family in ("vit", "deit") else average
    torch.testing.assert_close(
        export._extract_features_hf(lambda _: output, torch.empty(0), family), expected
    )


def test_unknown_feature_layout_is_rejected():
    with pytest.raises(RuntimeError):
        export._extract_features_timm(
            SimpleNamespace(forward_features=lambda _: {}), torch.empty(0), "vit"
        )
    for extract in [export._extract_features_hf, export._extract_features_timm]:
        with pytest.raises(NotImplementedError):
            extract(None, torch.empty(0), "unknown")


class _TimmFeatures(nn.Module):
    img_size = (4, 6)

    def forward_features(self, x):
        return x.mean((2, 3)).unsqueeze(1)


class _HFFeatures(nn.Module):
    def __init__(self, sign=1):
        super().__init__()
        self.sign = sign

    def forward(self, x):
        return SimpleNamespace(
            last_hidden_state=self.sign * x.mean((2, 3)).unsqueeze(1)
        )


@pytest.mark.parametrize("sign,strict", [(1, True), (-1, False), (-1, True)])
def test_validation_checks_actual_features(sign, strict, capsys):
    with torch.random.fork_rng():
        if sign == -1 and strict:
            with pytest.raises(ValueError, match="Sanity check failed"):
                export._validate_timm_vs_hf(
                    _TimmFeatures(),
                    _HFFeatures(sign),
                    "vit",
                    2,
                    1e-4,
                    1e-4,
                    "cpu",
                    strict,
                )
        else:
            export._validate_timm_vs_hf(
                _TimmFeatures(), _HFFeatures(sign), "vit", 2, 1e-4, 1e-4, "cpu", strict
            )
            assert ("WARNING" in capsys.readouterr().out) is (sign == -1)


def test_plain_export_preserves_weights_and_metadata(tmp_path):
    model = nn.Linear(3, 2)
    url = export.push_timm_to_hf(
        "custom_encoder", model, "team/model", hf_token="test-token"
    )
    assert url == "https://huggingface.co/team/model"
    saved = torch.load(tmp_path / "team__model/pytorch_model.bin", weights_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(saved[key], value)
    export.upload_folder.assert_called_once()
    assert export.upload_folder.call_args.kwargs["repo_id"] == "team/model"
    assert "custom_encoder" in (tmp_path / "team__model/model_type.txt").read_text()


def test_missing_token_fails_before_creating_remote_resources(monkeypatch):
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(export, "get_token", lambda: None)
    with pytest.raises(RuntimeError, match="token not found"):
        export.push_timm_to_hf("custom", nn.Linear(2, 2), "team/model")
    export.create_repo.assert_not_called()
    export.upload_folder.assert_not_called()


def test_incompatible_state_dict_falls_back_without_exporting_random_weights(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(export, "ViTConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(export, "ViTModel", lambda _: nn.Sequential(nn.Linear(3, 2)))
    model = nn.Linear(3, 2)
    export.push_timm_to_hf(
        "vit_custom", model, "team/model", hf_token="test-token", validate=False
    )
    root = tmp_path / "team__model"
    assert (root / "pytorch_model.bin").exists()
    assert not (root / "model.safetensors").exists()
    saved = torch.load(root / "pytorch_model.bin", weights_only=True)
    torch.testing.assert_close(saved["weight"], model.weight)


def test_image_processor_configuration_is_readable_json(tmp_path):
    export._save_image_processor(str(tmp_path), (8, 12), "vit")
    config = json.loads((tmp_path / "preprocessor_config.json").read_text())
    assert config["size"] == {"height": 8, "width": 12}
    assert config["do_normalize"] is True
    assert config["image_mean"] == [0.485, 0.456, 0.406]


def test_compatible_export_validates_features_and_preserves_safe_tensor_weights(
    monkeypatch, tmp_path
):
    from safetensors.torch import load_file

    class Config:
        def __init__(self, **kwargs):
            self.values = kwargs

        def save_pretrained(self, directory):
            (tmp_path / directory / "config.json").write_text(json.dumps(self.values))

    class TimmFeatures(_TimmFeatures):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(3.0))

        def forward_features(self, x):
            return super().forward_features(x) * self.scale

    class HFFeatures(_HFFeatures):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            output = super().forward(x)
            output.last_hidden_state = output.last_hidden_state * self.scale
            return output

    monkeypatch.setattr(export, "ViTConfig", Config)
    monkeypatch.setattr(export, "ViTModel", HFFeatures)
    model = TimmFeatures()
    export.push_timm_to_hf(
        "vit_local",
        model,
        "team/compatible",
        hf_token="test-token",
        strict=True,
        device="cpu",
    )
    root = tmp_path / "team__compatible"
    torch.testing.assert_close(
        load_file(root / "model.safetensors")["scale"], model.scale
    )
    assert json.loads((root / "config.json").read_text())["num_channels"] == 3
    assert (root / "preprocessor_config.json").exists()
    assert not (root / "pytorch_model.bin").exists()
    export.upload_folder.assert_called_once()
