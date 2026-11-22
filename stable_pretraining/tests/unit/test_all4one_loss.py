"""Numerical and gradient contracts for the released All4One objective."""

import math

import pytest
import torch
import torch.nn.functional as F

from stable_pretraining.losses import All4OneLoss


pytestmark = pytest.mark.unit


def _inputs(dtype=torch.float64, requires_grad=False):
    generator = torch.Generator().manual_seed(43)
    return {
        name: tuple(
            torch.randn(4, 3, generator=generator, dtype=dtype).requires_grad_(
                requires_grad
            )
            for _ in range(2)
        )
        for name in (
            "projections",
            "momentum_projections",
            "neighbours",
            "predictions",
            "centroids",
            "shifted_centroids",
        )
    }


def _reference(inputs, temperature=0.2, centroid_temperature=0.1):
    # Explicit loops and sqrt(mean(square)) provide an independent oracle for
    # cross-view pairing, normalization axes, and the released loss scaling.
    def contrast(anchor, prediction, tau):
        logits = F.normalize(anchor, dim=-1) @ F.normalize(prediction, dim=-1).T
        return F.cross_entropy(logits / tau, torch.arange(anchor.size(0)))

    neighbour = (
        sum(
            contrast(
                inputs["neighbours"][i].detach(),
                inputs["predictions"][1 - i],
                temperature,
            )
            for i in range(2)
        )
        / 2
    )
    centroid = (
        sum(
            contrast(
                inputs["centroids"][i],
                inputs["shifted_centroids"][1 - i],
                centroid_temperature,
            )
            for i in range(2)
        )
        / 2
    )
    diagonal_errors, off_errors = [], []
    for view in range(2):
        teacher = F.normalize(inputs["momentum_projections"][view].detach(), dim=1)
        teacher = F.normalize(teacher, dim=0)
        student = F.normalize(inputs["projections"][1 - view], dim=0)
        for i in range(student.size(1)):
            for j in range(student.size(1)):
                correlation = (teacher[:, i] * student[:, j]).sum()
                if i == j:
                    diagonal_errors.append((correlation - 1).square())
                else:
                    off_errors.append(correlation.square())
    diagonal = torch.stack(diagonal_errors).mean().sqrt()
    off = torch.stack(off_errors).mean().sqrt()
    feature = 2.5 * (diagonal + off)
    return {
        "loss": 0.5 * neighbour + 0.5 * centroid + feature,
        "neighbour_loss": neighbour,
        "centroid_loss": centroid,
        "feature_loss": feature,
        "feature_diagonal": diagonal,
        "feature_off_diagonal": off,
    }


def test_values_and_gradients_match_reference():
    inputs = _inputs(requires_grad=True)
    expected_inputs = {
        name: tuple(x.detach().clone().requires_grad_() for x in pair)
        for name, pair in inputs.items()
    }
    actual = All4OneLoss()(**inputs)
    expected = _reference(expected_inputs)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
    actual["loss"].backward()
    expected["loss"].backward()
    for name in inputs:
        for actual_tensor, reference_tensor in zip(inputs[name], expected_inputs[name]):
            if name in ("momentum_projections", "neighbours"):
                assert actual_tensor.grad is None
                assert reference_tensor.grad is None
            else:
                assert actual_tensor.grad.abs().sum() > 0
                torch.testing.assert_close(actual_tensor.grad, reference_tensor.grad)


def test_configurable_weights_and_independent_temperatures():
    inputs = _inputs()
    actual = All4OneLoss(
        temperature=0.7,
        centroid_temperature=0.3,
        neighbour_weight=1.2,
        centroid_weight=0.8,
        feature_diagonal_weight=5.0,
        feature_off_diagonal_weight=1.5,
    )(**inputs)
    expected = _reference(inputs, temperature=0.7, centroid_temperature=0.3)
    torch.testing.assert_close(actual["neighbour_loss"], expected["neighbour_loss"])
    torch.testing.assert_close(actual["centroid_loss"], expected["centroid_loss"])
    torch.testing.assert_close(
        actual["loss"],
        1.2 * expected["neighbour_loss"]
        + 0.8 * expected["centroid_loss"]
        + 5 * expected["feature_diagonal"]
        + 1.5 * expected["feature_off_diagonal"],
    )


@pytest.mark.parametrize("value", ["identity", "zero"])
def test_degenerate_features_have_finite_gradients(value):
    inputs = _inputs(requires_grad=True)
    base = (
        torch.eye(3, dtype=torch.float64)
        if value == "identity"
        else torch.zeros(3, 3, dtype=torch.float64)
    )
    inputs = {
        name: tuple(base.clone().requires_grad_() for _ in pair)
        for name, pair in inputs.items()
    }
    output = All4OneLoss()(**inputs)
    assert all(torch.isfinite(x) for x in output.values())
    if value == "identity":
        assert output["feature_loss"].item() == 0
        for key, temperature in (("neighbour_loss", 0.2), ("centroid_loss", 0.1)):
            expected = math.log1p(2 * math.exp(-1 / temperature))
            assert output[key].item() == pytest.approx(expected)
    else:
        torch.testing.assert_close(
            output["feature_loss"], torch.tensor(2.5, dtype=torch.float64)
        )
    output["loss"].backward()
    for name in ("projections", "predictions", "centroids", "shifted_centroids"):
        assert all(torch.isfinite(t.grad).all() for t in inputs[name])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduced_precision_inputs_are_computed_in_float32(dtype):
    inputs = _inputs(dtype=dtype, requires_grad=True)
    full_precision = {
        name: tuple(x.float() for x in pair) for name, pair in inputs.items()
    }
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = All4OneLoss()(**inputs)
    expected = All4OneLoss()(**full_precision)
    for key, tensor in actual.items():
        assert tensor.dtype == torch.float32
        torch.testing.assert_close(tensor, expected[key])
    actual["loss"].backward()
    assert all(torch.isfinite(t.grad).all() for t in inputs["projections"])


def test_missing_view_and_mismatched_shapes():
    inputs = _inputs()
    inputs["centroids"] = inputs["centroids"][:1]
    with pytest.raises(ValueError, match="two views"):
        All4OneLoss()(**inputs)
    inputs = _inputs()
    inputs["predictions"] = (torch.zeros(5, 3), inputs["predictions"][1])
    with pytest.raises(ValueError, match="same"):
        All4OneLoss()(**inputs)
