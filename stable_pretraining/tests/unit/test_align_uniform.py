"""Unit tests for Alignment and Uniformity."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from stable_pretraining.forward import align_uniform
from stable_pretraining.losses import AlignUniformLoss
from stable_pretraining.methods import AlignUniform

pytestmark = pytest.mark.unit


def test_align_uniform_loss_matches_definition_and_backpropagates():
    torch.manual_seed(0)
    z_i = torch.randn(4, 8, requires_grad=True)
    z_j = torch.randn(4, 8, requires_grad=True)
    alpha, t = 2.0, 2.0

    loss_fn = AlignUniformLoss(alpha=alpha, t=t)
    loss, alignment, uniformity = loss_fn.loss_and_terms(z_i, z_j)

    normalized_i = F.normalize(z_i, dim=-1)
    normalized_j = F.normalize(z_j, dim=-1)
    expected_alignment = (normalized_i - normalized_j).norm(dim=-1).pow(alpha).mean()
    expected_uniformity = torch.stack(
        [
            torch.log(torch.exp(-t * torch.pdist(view).square()).mean())
            for view in (normalized_i, normalized_j)
        ]
    ).mean()

    assert torch.allclose(alignment, expected_alignment)
    assert torch.allclose(uniformity, expected_uniformity)
    assert torch.allclose(loss, expected_alignment + expected_uniformity)
    assert torch.allclose(loss_fn(z_i, z_j), loss)
    loss.backward()
    assert z_i.grad is not None and torch.isfinite(z_i.grad).all()
    assert z_j.grad is not None and torch.isfinite(z_j.grad).all()


def test_align_uniform_loss_normalizes_inputs_and_applies_weights():
    torch.manual_seed(1)
    z_i = torch.randn(4, 8)
    z_j = torch.randn(4, 8)
    loss_fn = AlignUniformLoss(
        alignment_weight=0.25,
        uniformity_weight=3.0,
    )

    loss, alignment, uniformity = loss_fn.loss_and_terms(z_i, z_j)
    scaled_loss, scaled_alignment, scaled_uniformity = loss_fn.loss_and_terms(
        z_i * 10, z_j * 10
    )

    assert torch.allclose(loss, 0.25 * alignment + 3.0 * uniformity)
    assert torch.allclose(loss, scaled_loss)
    assert torch.allclose(alignment, scaled_alignment)
    assert torch.allclose(uniformity, scaled_uniformity)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0},
        {"t": 0},
        {"alignment_weight": -1},
        {"uniformity_weight": -1},
    ],
)
def test_align_uniform_loss_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        AlignUniformLoss(**kwargs)


@pytest.mark.parametrize(
    ("z_i", "z_j"),
    [
        (torch.randn(2, 3, 4), torch.randn(2, 3, 4)),
        (torch.randn(2, 3), torch.randn(3, 3)),
        (torch.empty(0, 3), torch.empty(0, 3)),
        (torch.randn(1, 3), torch.randn(1, 3)),
    ],
)
def test_align_uniform_loss_rejects_invalid_inputs(z_i, z_j):
    with pytest.raises(ValueError):
        AlignUniformLoss()(z_i, z_j)


def test_align_uniform_forward_logs_components_and_supports_single_view():
    module = SimpleNamespace(
        backbone=nn.Identity(),
        projector=nn.Identity(),
        align_uniform_loss=AlignUniformLoss(),
        training=True,
        log=Mock(),
    )
    views = [
        {"image": torch.randn(3, 8), "label": torch.tensor([0, 1, 2])},
        {"image": torch.randn(3, 8), "label": torch.tensor([0, 1, 2])},
    ]

    output = align_uniform(module, {"views": views}, "train")

    assert output["embedding"].shape == (6, 8)
    assert output["label"].shape == (6,)
    assert output["loss"].ndim == 0
    assert output["alignment"].ndim == 0
    assert output["uniformity"].ndim == 0
    assert module.log.call_count == 3

    module.training = False
    single_view = {"image": torch.randn(3, 8), "label": torch.tensor([0, 1, 2])}
    eval_output = align_uniform(module, single_view, "val")
    assert eval_output["embedding"].shape == (3, 8)
    assert torch.equal(eval_output["label"], single_view["label"])
    assert "loss" not in eval_output


def test_align_uniform_forward_rejects_wrong_number_of_views():
    module = SimpleNamespace()
    with pytest.raises(ValueError, match="exactly 2 views"):
        align_uniform(module, {"views": [{"image": torch.randn(2, 3)}]}, "train")


def test_align_uniform_method_forward_and_eval():
    backbone = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    model = AlignUniform(encoder_name=backbone, projector_dims=(16,))
    view1 = torch.randn(2, 3, 32, 32)
    view2 = torch.randn(2, 3, 32, 32)

    output = model(view1, view2)
    assert output.loss.ndim == 0 and torch.isfinite(output.loss)
    assert output.alignment.ndim == 0
    assert output.uniformity.ndim == 0
    output.loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )

    model.eval()
    with torch.no_grad():
        eval_output = model(view1)
    assert eval_output.embedding.shape == (2, 8)
    assert eval_output.projection is None
    assert eval_output.loss.item() == 0
