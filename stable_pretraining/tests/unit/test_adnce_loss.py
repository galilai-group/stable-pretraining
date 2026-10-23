"""Reference values, detached gradients, and numerical contracts for ADNCE."""

import math

import pytest
import torch
import torch.nn.functional as F

from stable_pretraining.losses import ADNCELoss, NTXEntLoss

pytestmark = pytest.mark.unit


def reference_loss(a, b, temperature=0.4, mu=0.7, sigma=0.5, detach=True):
    """Independent direct arithmetic matching the authors' image objective."""
    features = F.normalize(torch.cat((a, b)), dim=1)
    size = a.size(0)
    terms = []
    for row in range(2 * size):
        positive_idx = (row + size) % (2 * size)
        positive = (features[row] * features[positive_idx]).sum().div(temperature).exp()
        negatives = torch.stack(
            [
                (features[row] * features[col]).sum()
                for col in range(2 * size)
                if col not in (row, positive_idx)
            ]
        )
        weights = torch.exp(-((negatives - mu) ** 2) / (2 * sigma**2))
        weights = weights / weights.mean()
        if detach:
            weights = weights.detach()
        denominator = positive + (weights * (negatives / temperature).exp()).sum()
        terms.append(-(positive / denominator).log())
    return torch.stack(terms).mean()


@pytest.mark.parametrize("batch,dim", [(2, 3), (5, 7)])
@pytest.mark.parametrize("params", [(0.4, 0.7, 0.5), (0.2, -0.3, 0.2)])
def test_reference_values_and_gradients(batch, dim, params):
    generator = torch.Generator().manual_seed(19)
    a = torch.randn(
        batch, dim, generator=generator, dtype=torch.float64, requires_grad=True
    )
    b = torch.randn(
        batch, dim, generator=generator, dtype=torch.float64, requires_grad=True
    )
    actual = ADNCELoss(*params)(a, b)
    expected = reference_loss(a, b, *params)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for got, want in zip(
        torch.autograd.grad(actual, (a, b)), torch.autograd.grad(expected, (a, b))
    ):
        torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-12)


def test_weights_are_detached_not_just_value_equivalent():
    torch.manual_seed(71)
    a = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    actual = ADNCELoss(sigma=0.3)(a, b)
    live = reference_loss(a, b, sigma=0.3, detach=False)
    torch.testing.assert_close(actual, live)
    grad_actual = torch.autograd.grad(actual, a)[0]
    grad_live = torch.autograd.grad(live, a)[0]
    assert not torch.allclose(grad_actual, grad_live, rtol=1e-3, atol=1e-5)


def test_symmetry_permutation_and_scale_invariance():
    torch.manual_seed(12)
    a, b = (
        torch.randn(5, 9, dtype=torch.float64),
        torch.randn(5, 9, dtype=torch.float64),
    )
    perm = torch.tensor([3, 0, 4, 1, 2])
    loss = ADNCELoss()
    expected = loss(a, b)
    for actual in (loss(b, a), loss(a[perm], b[perm]), loss(a * 7, b * 0.2)):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(loss(a, b[perm]), expected)


def test_uniform_weight_limit_matches_ntxent():
    torch.manual_seed(4)
    a = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    actual = ADNCELoss(temperature=0.4, sigma=1e8)(a, b)
    expected = NTXEntLoss(temperature=0.4)(a, b)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for got, want in zip(
        torch.autograd.grad(actual, (a, b)), torch.autograd.grad(expected, (a, b))
    ):
        torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("kind", ["zeros", "identical", "random"])
@pytest.mark.parametrize("params", [(0.4, 0.7, 0.5), (0.001, 0.7, 0.0001)])
def test_finite_loss_and_gradients_for_degenerate_and_sharp_inputs(kind, params):
    torch.manual_seed(90)
    if kind == "zeros":
        a = torch.zeros(4, 8)
    elif kind == "identical":
        a = torch.ones(4, 8)
    else:
        a = torch.randn(4, 8)
    a.requires_grad_()
    b = a.detach().clone().requires_grad_()
    loss = ADNCELoss(*params)(a, b)
    assert loss.ndim == 0 and torch.isfinite(loss)
    if kind != "random":
        torch.testing.assert_close(
            loss, torch.tensor(math.log(7)), atol=1e-6, rtol=1e-6
        )
    loss.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_precision_and_autocast(dtype):
    torch.manual_seed(1)
    a = torch.randn(3, 8).to(dtype).requires_grad_()
    b = torch.randn(3, 8).to(dtype).requires_grad_()
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = ADNCELoss()(a, b)
    expected = ADNCELoss()(a.to(expected_dtype), b.to(expected_dtype))
    assert actual.dtype == expected_dtype
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


@pytest.mark.parametrize(
    "name,value",
    [
        ("temperature", 0),
        ("temperature", -1),
        ("temperature", float("nan")),
        ("mu", float("inf")),
        ("sigma", 0),
        ("sigma", -1),
        ("sigma", float("inf")),
    ],
)
def test_invalid_parameters(name, value):
    with pytest.raises(ValueError, match=name):
        ADNCELoss(**{name: value})


@pytest.mark.parametrize(
    "a,b,error",
    [
        ([1.0], torch.ones(2, 3), TypeError),
        (torch.ones(2, 3, dtype=torch.long), torch.ones(2, 3), TypeError),
        (torch.ones(2, 3, dtype=torch.complex64), torch.ones(2, 3), TypeError),
        (torch.ones(3), torch.ones(3), ValueError),
        (torch.ones(2, 3), torch.ones(3, 3), ValueError),
        (torch.ones(0, 3), torch.ones(0, 3), ValueError),
        (torch.ones(2, 0), torch.ones(2, 0), ValueError),
        (torch.ones(1, 3), torch.ones(1, 3), ValueError),
        (torch.ones(2, 3), torch.ones(2, 3, dtype=torch.float64), ValueError),
        (torch.ones(2, 3), torch.ones(2, 3, device="meta"), ValueError),
    ],
)
def test_invalid_inputs(a, b, error):
    with pytest.raises(error):
        ADNCELoss()(a, b)


def test_single_process_ignores_stale_rank_environment(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    a = torch.eye(3)
    torch.testing.assert_close(ADNCELoss()(a, a), reference_loss(a, a))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_autocast(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")
    torch.manual_seed(1)
    model = torch.nn.Linear(8, 6).cuda()
    x, y = torch.randn(4, 8, device="cuda"), torch.randn(4, 8, device="cuda")
    with torch.autocast("cuda", dtype=dtype):
        a, b = model(x), model(y)
        loss = ADNCELoss()(a, b)
    torch.testing.assert_close(loss, ADNCELoss()(a.float(), b.float()))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
