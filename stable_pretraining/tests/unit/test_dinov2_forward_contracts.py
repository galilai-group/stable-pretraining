"""Real tiny ViTs verify DINOv2 masking, labels, and student-only gradients."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from stable_pretraining.backbone.utils import TeacherStudentWrapper, vit_hf
from stable_pretraining.forward import dinov2
from stable_pretraining.losses.dino import DINOv2Loss

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolate_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield


class _RawTokens(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, **kwargs):
        return self.model(image, **kwargs).last_hidden_state


def _model(raw=False):
    backbone = vit_hf(
        "tiny",
        image_size=16,
        patch_size=4,
        hidden_size=12,
        num_hidden_layers=1,
        num_attention_heads=3,
        intermediate_size=24,
    )
    return SimpleNamespace(
        training=True,
        current_epoch=0,
        mask_ratio=0.5,
        backbone=TeacherStudentWrapper(_RawTokens(backbone) if raw else backbone),
        projector=TeacherStudentWrapper(nn.Linear(12, 8)),
        patch_projector=TeacherStudentWrapper(nn.Linear(12, 8)),
        dinov2_loss=DINOv2Loss(),
        log=Mock(),
    )


def _views(local=True, local_size=8):
    views = {
        name: {"image": torch.randn(2, 3, 16, 16), "label": torch.tensor([0, 1])}
        for name in ["global_1", "global_2"]
    }
    if local:
        views["local_1"] = {
            "image": torch.randn(2, 3, local_size, local_size),
            "label": torch.tensor([0, 1]),
        }
    return {"views": views}


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("local", [False, True])
def test_dinov2_masks_exact_patch_count_and_trains_only_student(raw, local):
    model = _model(raw)
    masks = []

    def capture(module, args, kwargs):
        if "bool_masked_pos" in kwargs:
            masks.append(kwargs["bool_masked_pos"].clone())

    handle = model.backbone.student.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        result = dinov2(model, _views(local), "train")
    finally:
        handle.remove()
    assert result["embedding"].shape == (4, 12)
    assert not result["embedding"].requires_grad
    assert result["label"].tolist() == [0, 1, 0, 1]
    assert len(masks) == 1 and masks[0].shape == (2, 16)
    assert masks[0].sum(1).tolist() == [8, 8]
    assert result["loss"].ndim == 0 and torch.isfinite(result["loss"])
    result["loss"].backward()
    for wrapper in [model.backbone, model.projector, model.patch_projector]:
        gradients = [p.grad for p in wrapper.student.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert all(p.grad is None for p in wrapper.teacher.parameters())
    assert model.log.call_args.args[0] == "train/loss"


@pytest.mark.parametrize("epoch,expected", [(0, 0.04), (2, 0.055), (4, 0.07)])
def test_teacher_temperature_warmup_reaches_requested_value(
    epoch, expected, monkeypatch
):
    model = _model()
    model.current_epoch = epoch
    model.warmup_epochs_temperature_teacher = 4
    model.warmup_temperature_teacher = 0.04
    model.temperature_teacher = 0.07
    sinkhorn = Mock(wraps=model.dinov2_loss.dino_loss.sinkhorn_knopp_teacher)
    monkeypatch.setattr(model.dinov2_loss.dino_loss, "sinkhorn_knopp_teacher", sinkhorn)
    dinov2(model, _views(local=False), "train")
    assert sinkhorn.call_args.kwargs["teacher_temp"] == pytest.approx(expected)


@pytest.mark.parametrize("multiview", [False, True])
def test_validation_returns_teacher_embeddings_with_aligned_labels(multiview):
    model = _model()
    model.training = False
    batch = (
        _views(local_size=16)
        if multiview
        else {"image": torch.randn(2, 3, 16, 16), "label": torch.tensor([0, 1])}
    )
    result = dinov2(model, batch, "val")
    assert result["embedding"].shape == (6 if multiview else 2, 12)
    assert result["label"].tolist() == [0, 1] * (3 if multiview else 1)
    assert not result["embedding"].requires_grad
    assert "loss" not in result


def test_ibot_requires_patch_projector():
    model = _model()
    del model.patch_projector
    with pytest.raises(ValueError, match="patch_projector"):
        dinov2(model, _views(local=False), "train")


def test_unsupported_vit_size_has_clear_error():
    with pytest.raises(ValueError, match="Invalid size"):
        vit_hf("not-a-size")
