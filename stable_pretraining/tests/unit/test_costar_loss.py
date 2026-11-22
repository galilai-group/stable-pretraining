"""Core numerical checks for COSTAR's loss and history augmentation."""

import math

import pytest
import torch

from stable_pretraining.losses import COSTARLoss
from stable_pretraining.methods.costar import COSTARAugmentation

pytestmark = pytest.mark.unit


def _scalar_reference(
    query: torch.Tensor,
    keys: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Calculate temperature-one InfoNCE and query gradients with scalar math."""
    rows = query.detach().tolist()
    candidates = keys.detach().tolist()
    key_norms = [math.hypot(*key) for key in candidates]
    losses, gradients = [], []
    for index, row in enumerate(rows):
        norm = math.hypot(*row)
        cosines = [
            math.fsum(x * y for x, y in zip(row, key, strict=True)) / (norm * key_norm)
            for key, key_norm in zip(candidates, key_norms, strict=True)
        ]
        maximum = max(cosines)
        exponentials = [math.exp(value - maximum) for value in cosines]
        probabilities = [value / sum(exponentials) for value in exponentials]
        losses.append(maximum + math.log(sum(exponentials)) - cosines[index])
        gradients.append(
            [
                sum(
                    (probability - (column == index))
                    * (
                        key[feature] / (norm * key_norm)
                        - cosine * row[feature] / norm**2
                    )
                    for column, (key, key_norm, cosine, probability) in enumerate(
                        zip(candidates, key_norms, cosines, probabilities, strict=True)
                    )
                )
                / len(rows)
                for feature in range(len(row))
            ]
        )
    return math.fsum(losses) / len(rows), torch.tensor(gradients, dtype=query.dtype)


def test_info_nce_matches_signed_scalar_values_and_gradients() -> None:
    """Signed cosine scores use diagonal positives and the local candidate set."""
    query = torch.tensor(
        [[2.0, 0.0], [-3.0, 0.0], [1.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    keys = torch.tensor([[-1.0, 0.0], [2.0, 1.0], [0.0, -2.0]], dtype=torch.float64)
    expected, gradient = _scalar_reference(query, keys)
    actual = COSTARLoss(temperature=1.0).info_nce(query, keys)
    assert actual.item() == pytest.approx(expected, abs=1e-12)
    actual.backward()
    torch.testing.assert_close(query.grad, gradient, rtol=1e-12, atol=1e-12)


def test_bidirectional_component_mean_and_target_detachment() -> None:
    """Equation (2) sums view directions and averages the three component terms."""
    streams = ("covariate", "treatment", "outcome", "history")
    generator = torch.Generator().manual_seed(17)
    views = tuple(
        {
            name: torch.randn(
                3, 4, generator=generator, dtype=torch.float64
            ).requires_grad_()
            for name in streams
        }
        for _ in range(4)
    )
    expected, gradients = {}, {}
    for name in streams:
        first, gradients[0, name] = _scalar_reference(views[0][name], views[3][name])
        second, gradients[1, name] = _scalar_reference(views[1][name], views[2][name])
        expected[name] = first + second
    actual = COSTARLoss(temperature=1.0)(*views)
    total = expected["history"] + sum(expected[name] for name in streams[:3]) / 3
    assert actual["loss"].item() == pytest.approx(total, abs=1e-12)
    for name in streams:
        assert actual[f"loss_{name}"].item() == pytest.approx(expected[name], abs=1e-12)
    actual["loss"].backward()
    for index, view in enumerate(views[:2]):
        for name, value in view.items():
            weight = 1 if name == "history" else 1 / 3
            torch.testing.assert_close(
                value.grad,
                gradients[index, name] * weight,
                rtol=1e-12,
                atol=1e-12,
            )
    assert all(value.grad is None for view in views[2:] for value in view.values())


def test_batch_augmentation_matches_noise_padding_and_analytic_gradients() -> None:
    """Batch-shared affine noise and elementwise jitter exclude padded NaNs."""
    mask = torch.tensor([[True, True, True], [True, True, False]])
    history = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 10
    history = history.masked_fill(~mask[:, :, None], float("nan")).requires_grad_()
    original = history.detach().clone()
    clean = history.detach().masked_fill(~mask[:, :, None], 0)
    generator = torch.Generator().manual_seed(11)
    scale = 1 + 0.5 * torch.randn(4, generator=generator, dtype=torch.float64)
    shift = 0.5 * torch.randn(4, generator=generator, dtype=torch.float64)
    jitter = 0.5 * torch.randn(history.shape, generator=generator, dtype=torch.float64)
    expected = (clean * scale + shift + jitter).masked_fill(~mask[:, :, None], 0)
    gradient = scale.expand_as(history).masked_fill(~mask[:, :, None], 0)
    augmentation = COSTARAugmentation(
        1,
        2,
        1,
        scale_std=0.5,
        shift_std=0.5,
        jitter_std=0.5,
        probability=1.0,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        actual = augmentation(history, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(history, original, equal_nan=True, rtol=0, atol=0)
    actual.sum().backward()
    torch.testing.assert_close(history.grad, gradient, rtol=0, atol=0)
    assert torch.isfinite(actual).all() and torch.isfinite(history.grad).all()
