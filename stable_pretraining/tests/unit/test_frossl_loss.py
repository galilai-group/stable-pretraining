"""Numerical and input contracts for FroSSL's multiview loss."""

import math

import pytest
import torch

from stable_pretraining.losses import FroSSLLoss


pytestmark = pytest.mark.unit


def test_two_view_reference_value():
    first = torch.eye(2, dtype=torch.float64)
    second = first.flip(0)

    aligned = FroSSLLoss(invariance_weight=1.4)([first, first])
    mismatched = FroSSLLoss(invariance_weight=1.4)([first, second])

    torch.testing.assert_close(
        aligned, torch.tensor(-2 * math.log(2), dtype=first.dtype)
    )
    torch.testing.assert_close(
        mismatched,
        torch.tensor(-2 * math.log(2) + 2 * 1.4, dtype=first.dtype),
    )


def test_multiview_matches_pairwise_invariance():
    views = [
        torch.tensor([[1.0, 2.0], [2.0, -1.0], [-1.0, 1.0]], dtype=torch.float64),
        torch.tensor([[2.0, 1.0], [1.0, -2.0], [1.0, 1.0]], dtype=torch.float64),
        torch.tensor([[1.0, -1.0], [-2.0, 1.0], [1.0, 2.0]], dtype=torch.float64),
    ]
    normalized = [view / view.norm(dim=0, keepdim=True) for view in views]
    regularizer = sum(
        ((z.T @ z).square().sum() / (z.square().sum().square())).log()
        for z in normalized
    )
    pairwise = sum(
        (normalized[i] - normalized[j]).square().mean()
        for i in range(len(views))
        for j in range(i + 1, len(views))
    )

    actual = FroSSLLoss(invariance_weight=0.7)(views)
    torch.testing.assert_close(actual, regularizer + 0.7 * views[0].shape[1] * pairwise)
    torch.testing.assert_close(actual, FroSSLLoss(0.7)(list(reversed(views))))


@pytest.mark.parametrize(
    ("count", "batch", "features", "expected_loss", "expected_gradient"),
    [
        (4, 3, 5, 47.566494933409814, 0.001810185782627538),
        (8, 7, 3, 46.25868643630915, 0.004227094021293232),
    ],
)
def test_matches_authors_multiview_reference(
    count, batch, features, expected_loss, expected_gradient
):
    # Pin the authors' released multiview_frossl_loss_func on fixed inputs:
    # https://github.com/OFSkean/FroSSL/blob/main/solo/losses/frossl.py
    rows = torch.arange(batch)[:, None]
    columns = torch.arange(features)[None, :]
    views = [
        (
            ((rows * 7 + columns * 11 + index * 13) % 17 - 8).to(torch.float64)
            + (((rows + 1) * (columns + 1) * (index + 1)) % 5).to(torch.float64) / 5
        ).requires_grad_()
        for index in range(count)
    ]

    loss = FroSSLLoss(invariance_weight=2.0)(views)
    first_gradient = torch.autograd.grad(loss, views)[0][0, 0]

    assert loss.item() == pytest.approx(expected_loss, rel=1e-11, abs=1e-11)
    assert first_gradient.item() == pytest.approx(
        expected_gradient, rel=1e-11, abs=1e-11
    )


def test_wide_projection_uses_equivalent_gram_matrix():
    first = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, -1.0, 3.0]])
    second = first.flip(0)
    normalized = [view / view.norm(dim=0, keepdim=True) for view in (first, second)]
    expected_regularizer = sum(
        ((z.T @ z).square().sum() / z.square().sum().square()).log() for z in normalized
    )
    expected_invariance = (
        first.shape[1] * (normalized[0] - normalized[1]).square().mean()
    )

    torch.testing.assert_close(
        FroSSLLoss()([first, second]), expected_regularizer + expected_invariance
    )


def test_gradients_and_half_precision():
    torch.manual_seed(4)
    views = [
        torch.randn(3, 2, dtype=torch.float64, requires_grad=True) for _ in range(3)
    ]
    loss_fn = FroSSLLoss()

    assert torch.autograd.gradcheck(lambda *xs: loss_fn(xs), tuple(views))
    loss = loss_fn([view.detach().to(torch.bfloat16) for view in views])
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_invalid_weight(weight):
    with pytest.raises(ValueError, match="invariance_weight"):
        FroSSLLoss(weight)


def test_invalid_views():
    loss_fn = FroSSLLoss()
    valid = torch.eye(2)
    with pytest.raises(ValueError, match="at least two"):
        loss_fn([valid])
    with pytest.raises(ValueError, match="shape"):
        loss_fn([valid, torch.ones(3, 2)])
    with pytest.raises(ValueError, match="floating dtype"):
        loss_fn([valid, valid.double()])
    with pytest.raises(ValueError, match="finite"):
        loss_fn([valid, torch.full_like(valid, float("nan"))])
    with pytest.raises(ValueError, match="nonzero"):
        loss_fn([valid, torch.zeros_like(valid)])
