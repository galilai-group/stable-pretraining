"""Numerical, gradient, and public API contracts for FlatNCE."""

from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import stable_pretraining as spt
from stable_pretraining.losses import FlatNCELoss, NTXEntLoss

pytestmark = pytest.mark.unit


def _reference(z1, z2, temperature):
    features = F.normalize(torch.cat([z1, z2]), dim=-1)
    terms = []
    b = len(z1)
    for i, anchor in enumerate(features):
        positive = (i + b) % len(features)
        negative_indices = [j for j in range(len(features)) if j not in (i, positive)]
        negative_scores = features[negative_indices] @ anchor / temperature
        positive_score = features[positive] @ anchor / temperature
        terms.append(torch.logsumexp(negative_scores, dim=0) - positive_score)
    return torch.stack(terms).mean()


@pytest.mark.parametrize("temperature", [0.05, 0.5, 2.0])
def test_loss_gradients_match_independent_negative_only_reference(temperature):
    generator = torch.Generator().manual_seed(12)
    z1 = torch.randn(4, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    z2 = torch.randn(4, 7, generator=generator, dtype=torch.float64, requires_grad=True)
    loss = FlatNCELoss(temperature)(z1, z2)
    actual = torch.autograd.grad(loss, (z1, z2))
    expected = torch.autograd.grad(_reference(z1, z2, temperature), (z1, z2))

    assert loss.shape == () and loss.item() == 1.0
    for gradient, reference in zip(actual, expected):
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        torch.testing.assert_close(gradient, reference, rtol=1e-12, atol=1e-12)


def test_gradients_survive_a_saturated_ntxent_objective():
    z1 = torch.eye(2, requires_grad=True)
    z2 = torch.eye(2, requires_grad=True)
    flat = torch.autograd.grad(FlatNCELoss(1e-4)(z1, z2), (z1, z2))
    ntx = torch.autograd.grad(NTXEntLoss(1e-4)(z1, z2), (z1, z2))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in flat)
    assert all(g.abs().max() == 0 for g in ntx)


def test_view_swap_and_sample_permutation_preserve_gradients():
    generator = torch.Generator().manual_seed(4)
    z1 = torch.randn(4, 5, generator=generator, requires_grad=True)
    z2 = torch.randn(4, 5, generator=generator, requires_grad=True)
    objective = FlatNCELoss()
    original = torch.autograd.grad(objective(z1, z2), (z1, z2))
    swapped = torch.autograd.grad(objective(z2, z1), (z1, z2))
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = torch.autograd.grad(
        objective(z1[permutation], z2[permutation]), (z1, z2)
    )
    for expected, swapped_grad, permuted_grad in zip(original, swapped, permuted):
        torch.testing.assert_close(swapped_grad, expected)
        torch.testing.assert_close(permuted_grad, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_low_precision_inputs_have_finite_gradients_and_float32_loss(dtype):
    z1 = torch.randn(3, 5, dtype=dtype, requires_grad=True)
    z2 = torch.randn(3, 5, dtype=dtype, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = FlatNCELoss(0.05)(z1, z2)
    gradients = torch.autograd.grad(loss, (z1, z2))
    assert loss.dtype == torch.float32 and loss.item() == 1.0
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients)


@pytest.mark.parametrize("temperature", [0.0, -0.1, float("inf"), float("nan")])
def test_temperature_must_be_finite_and_positive(temperature):
    with pytest.raises(ValueError, match="temperature"):
        FlatNCELoss(temperature)


@pytest.mark.parametrize(
    "z1,z2,message",
    [
        (torch.randn(3, 4), torch.randn(2, 4), "same.*shape"),
        (torch.randn(2, 3, 4), torch.randn(2, 3, 4), "same.*shape"),
        (torch.empty(0, 4), torch.empty(0, 4), "nonempty"),
        (torch.empty(2, 0), torch.empty(2, 0), "nonempty"),
        (torch.ones(2, 4, dtype=torch.long), torch.ones(2, 4), "floating-point"),
        (torch.randn(2, 4), torch.randn(2, 4, dtype=torch.float64), "device and dtype"),
        (torch.randn(2, 4), torch.empty(2, 4, device="meta"), "device and dtype"),
        (torch.randn(1, 4), torch.randn(1, 4), "at least two images globally"),
    ],
)
def test_invalid_projection_inputs(z1, z2, message):
    with pytest.raises(ValueError, match=message):
        FlatNCELoss()(z1, z2)


def _module():
    module = spt.Module(
        backbone=nn.Linear(4, 6),
        projector=nn.Linear(6, 3),
        forward=spt.forward.flatnce,
        flatnce_loss=FlatNCELoss(),
    )
    module.log = Mock()
    return module


@pytest.mark.parametrize("named", [False, True])
def test_forward_preserves_two_view_embedding_and_label_alignment(named):
    module = _module().train()
    views = [
        {"image": torch.randn(3, 4), "label": torch.tensor([0, 1, 2])},
        {"image": torch.randn(3, 4), "label": torch.tensor([0, 1, 2])},
    ]
    batch = {"views": dict(zip(["first", "second"], views)) if named else views}
    result = module(batch, stage="fit")
    expected = torch.cat([module.backbone(view["image"]) for view in views])
    torch.testing.assert_close(result["embedding"], expected)
    assert result["label"].tolist() == [0, 1, 2, 0, 1, 2]
    result["loss"].backward()
    assert module.backbone.weight.grad.norm() > 0
    assert module.projector.weight.grad.norm() > 0
    assert module.log.call_args.args[0] == "fit/loss"


def test_single_view_evaluation_skips_loss_and_projector():
    module = _module().eval()
    module.projector.forward = Mock(
        side_effect=AssertionError("evaluation ran the projector")
    )
    batch = {"image": torch.randn(3, 4), "label": torch.tensor([0, 1, 2])}
    result = module(batch, stage="validate")
    assert "loss" not in result
    torch.testing.assert_close(result["embedding"], module.backbone(batch["image"]))
    torch.testing.assert_close(result["label"], batch["label"])


@pytest.mark.parametrize("count", [0, 1, 3])
def test_forward_requires_exactly_two_views(count):
    with pytest.raises(ValueError, match="FlatNCE requires exactly 2 views"):
        _module()({"views": [{"image": torch.randn(3, 4)}] * count}, stage="fit")


def test_public_method_supports_tensor_and_module_batch_interfaces():
    from stable_pretraining.methods import FlatNCE

    assert spt.FlatNCE is FlatNCE
    backbone = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 6))
    model = spt.FlatNCE(encoder_name=backbone, projector_dims=(8, 4))
    model.log = Mock()
    v1, v2 = torch.randn(3, 3, 8, 8), torch.randn(3, 3, 8, 8)
    direct = model(v1, v2)
    assert direct.embedding.shape == (6, 6)
    assert direct.projection.shape == (6, 4)
    direct.loss.backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in backbone.parameters())

    batch = {"views": [{"image": v1}, {"image": v2}]}
    output = model(batch, stage="fit")
    assert output["loss"].item() == 1.0
    assert output["embedding"].shape == (6, 6)

    model.eval()
    assert model(v1).loss.item() == 0
    assert model(v1).embedding.shape == (3, 6)
    assert "loss" not in model({"image": v1}, stage="validate")
    with pytest.raises(ValueError, match="view2 must be omitted"):
        model(batch, v2)
