"""Numerical and gradient contracts for SimCo's dual-temperature loss."""

import importlib

import pytest
import torch
import torch.nn.functional as F

from stable_pretraining.losses.simco import SimCoLoss

pytestmark = pytest.mark.unit


def _author_reference(
    z1: torch.Tensor,
    z2: torch.Tensor,
    intra_temperature: float = 0.1,
    inter_temperature: float = 1.0,
    detach_ratio: bool = True,
) -> torch.Tensor:
    """Apply equation (9) directly using the author's probability-space form."""
    x, y = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    terms = []
    for query, key in ((x, y), (y, x)):
        similarities = query @ key.T
        intra = F.softmax(similarities / intra_temperature, dim=1)
        inter = F.softmax(similarities / inter_temperature, dim=1)
        ratio = (1 - inter.diagonal()) / (1 - intra.diagonal())
        if detach_ratio:
            ratio = ratio.detach()
        cross_entropy = -F.log_softmax(
            similarities / intra_temperature, dim=1
        ).diagonal()
        terms.append((ratio * cross_entropy).mean())
    return sum(terms) / 2


def _inputs(seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(4, 5, generator=generator, dtype=torch.float64, requires_grad=True)
        for _ in range(2)
    )


def test_matches_author_loss_and_both_view_gradients() -> None:
    z1, z2 = _inputs()
    actual = SimCoLoss()(z1, z2)
    expected = _author_reference(z1, z2)
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(actual, (z1, z2))
    expected_gradients = torch.autograd.grad(expected, (z1, z2))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
        assert torch.isfinite(actual_gradient).all()
        assert actual_gradient.norm() > 0


def test_equal_temperatures_reduce_to_cross_view_symmetric_cross_entropy() -> None:
    z1, z2 = _inputs(8)
    similarities = F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).T / 0.3
    targets = torch.arange(z1.shape[0])
    expected = (
        F.cross_entropy(similarities, targets)
        + F.cross_entropy(similarities.T, targets)
    ) / 2
    actual = SimCoLoss(0.3, 0.3)(z1, z2)
    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        torch.autograd.grad(actual, (z1, z2)),
        torch.autograd.grad(expected, (z1, z2)),
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_probability_ratio_is_stopped_without_detaching_either_view() -> None:
    z1, z2 = _inputs(9)
    actual = SimCoLoss()(z1, z2)
    unstopped = _author_reference(z1, z2, detach_ratio=False)
    torch.testing.assert_close(actual, unstopped)
    for actual_gradient, unstopped_gradient in zip(
        torch.autograd.grad(actual, (z1, z2)),
        torch.autograd.grad(unstopped, (z1, z2)),
    ):
        assert not torch.allclose(actual_gradient, unstopped_gradient)
        assert actual_gradient.norm() > 0


def test_symmetry_pair_permutation_rotation_and_positive_row_scaling() -> None:
    z1, z2 = _inputs(10)
    generator = torch.Generator().manual_seed(10)
    rotation, _ = torch.linalg.qr(
        torch.randn(5, 5, generator=generator, dtype=torch.float64)
    )
    scales = [
        torch.rand(4, 1, generator=generator, dtype=torch.float64) + 0.2
        for _ in range(2)
    ]
    permutation = torch.tensor([2, 0, 3, 1])
    loss_fn = SimCoLoss()
    expected = loss_fn(z1, z2)
    for actual in (
        loss_fn(z2, z1),
        loss_fn(z1[permutation], z2[permutation]),
        loss_fn(z1 @ rotation, z2 @ rotation),
        loss_fn(z1 * scales[0], z2 * scales[1]),
    ):
        torch.testing.assert_close(actual, expected)
    assert not torch.allclose(loss_fn(z1, z2[permutation]), expected)


def test_equal_temperature_gradcheck() -> None:
    # Unequal temperatures intentionally stop the ratio's derivative, so an
    # ordinary finite-difference check of that scalar function is inappropriate.
    assert torch.autograd.gradcheck(SimCoLoss(0.3, 0.3), _inputs(11))


def test_global_gather_preserves_pair_order_and_both_source_gradients(
    monkeypatch,
) -> None:
    z1, z2 = _inputs(12)
    local = (z1[:2], z2[:2])
    remote = iter((F.normalize(z1[2:], dim=1), F.normalize(z2[2:], dim=1)))

    def gather_with_remote(tensor: torch.Tensor) -> list[torch.Tensor]:
        return [tensor, next(remote)]

    module = importlib.import_module("stable_pretraining.losses.simco")
    monkeypatch.setattr(module, "all_gather", gather_with_remote)
    actual = SimCoLoss()(*local)
    expected = _author_reference(z1, z2)
    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        torch.autograd.grad(actual, (z1, z2)),
        torch.autograd.grad(expected, (z1, z2)),
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("temperature", [0.1, 0.01])
def test_saturated_positive_probability_has_finite_loss_and_gradients(
    dtype: torch.dtype, temperature: float
) -> None:
    z1 = torch.tensor([[1.0, 0.05], [-1.0, 0.0]], dtype=dtype, requires_grad=True)
    z2 = torch.tensor([[1.0, 0.0], [-1.0, 0.05]], dtype=dtype, requires_grad=True)
    loss_fn = SimCoLoss(temperature, 1.0)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = loss_fn(z1, z2)
    expected = loss_fn(z1.double(), z2.double())
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=3e-5, atol=1e-7)
    assert actual > 0
    gradients = torch.autograd.grad(actual, (z1, z2))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient.norm() > 0 for gradient in gradients)


def test_float64_precision_is_preserved() -> None:
    assert SimCoLoss()(*_inputs()).dtype == torch.float64


@pytest.mark.parametrize(
    ("shape1", "shape2"),
    [
        ((4,), (4,)),
        ((2, 3, 4), (2, 3, 4)),
        ((2, 4), (3, 4)),
        ((0, 4), (0, 4)),
        ((2, 0), (2, 0)),
        ((1, 4), (1, 4)),
    ],
)
def test_invalid_shapes_raise(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        SimCoLoss()(torch.zeros(shape1), torch.zeros(shape2))


@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64])
def test_non_real_floating_inputs_raise(dtype: torch.dtype) -> None:
    with pytest.raises(TypeError, match="floating-point"):
        SimCoLoss()(torch.ones(2, 4, dtype=dtype), torch.ones(2, 4, dtype=dtype))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf"), float("nan")])
@pytest.mark.parametrize("parameter", ["intra_temperature", "inter_temperature"])
def test_invalid_temperatures_raise(parameter: str, temperature: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        SimCoLoss(**{parameter: temperature})
