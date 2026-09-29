"""LARS updates agree with explicit trust ratios and SGD's momentum semantics."""

import copy

import pytest
import torch

from stable_pretraining.optim.lars import LARS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "options",
    [
        {"lr": -0.1},
        {"momentum": -0.1},
        {"weight_decay": -0.1},
        {"nesterov": True},
        {"nesterov": True, "momentum": 0.9, "dampening": 0.1},
    ],
)
def test_invalid_optimizer_options_are_rejected(options):
    with pytest.raises(ValueError):
        LARS([torch.nn.Parameter(torch.ones(2))], **{"lr": 0.1, **options})


@pytest.mark.parametrize(
    "momentum,nesterov,dampening",
    [(0.0, False, 0.0), (0.9, False, 0.2), (0.9, True, 0.0)],
)
@pytest.mark.parametrize("excluded", ["no_decay", "bias"])
def test_excluded_parameters_match_sgd_across_steps(
    momentum, nesterov, dampening, excluded
):
    a = torch.nn.Parameter(
        torch.tensor([1.0, 2.0, 3.0]) if excluded == "bias" else torch.ones(2, 3)
    )
    b = torch.nn.Parameter(a.detach().clone())
    opts = dict(lr=0.1, momentum=momentum, nesterov=nesterov, dampening=dampening)
    lars = LARS(
        [a],
        weight_decay=0.2 if excluded == "bias" else 0.0,
        exclude_bias_n_norm=True,
        **opts,
    )
    sgd = torch.optim.SGD([b], **opts)
    for i in range(3):
        a.grad = torch.full_like(a, 0.3 + i)
        b.grad = a.grad.clone()
        lars.step()
        sgd.step()
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("eta", [0.001, 10.0])
def test_trust_ratio_scales_weight_decay_and_gradient(clip, eta):
    p = torch.nn.Parameter(torch.tensor([[3.0, 4.0]], dtype=torch.float64))
    p.grad = torch.tensor([[0.6, 0.8]], dtype=torch.float64)
    before = p.detach().clone()
    ratio = eta * 5 / (1 + 0.2 * 5 + 1e-8)
    if clip:
        ratio = min(ratio / 0.1, 1.0)
    expected = before - 0.1 * ratio * (p.grad + 0.2 * before)
    LARS([p], lr=0.1, weight_decay=0.2, eta=eta, clip_lr=clip).step()
    torch.testing.assert_close(p, expected)


@pytest.mark.parametrize("zero", ["parameter", "gradient", "missing_gradient"])
def test_zero_norms_and_missing_gradients_are_finite(zero):
    p = torch.nn.Parameter(
        torch.zeros(2, 2) if zero == "parameter" else torch.ones(2, 2)
    )
    if zero != "missing_gradient":
        p.grad = torch.zeros_like(p) if zero == "gradient" else torch.ones_like(p)
    before = p.detach().clone()
    LARS([p], lr=0.1, weight_decay=0.2).step()
    expected = before - 0.1 if zero == "parameter" else before
    torch.testing.assert_close(p, expected)


def test_closure_and_checkpoint_resume_preserve_momentum():
    p = torch.nn.Parameter(torch.tensor([[1.0, 2.0]]))
    optimizer = LARS([p], lr=0.1, momentum=0.9, weight_decay=0.1)

    def closure():
        optimizer.zero_grad()
        loss = p.square().sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss.item() == 5.0
    restored = torch.nn.Parameter(p.detach().clone())
    resumed = LARS([restored], lr=0.1, momentum=0.9, weight_decay=0.1)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for parameter, opt in [(p, optimizer), (restored, resumed)]:
        parameter.grad = torch.ones_like(parameter)
        opt.step()
    torch.testing.assert_close(p, restored)
    torch.testing.assert_close(
        optimizer.state[p]["momentum_buffer"],
        resumed.state[restored]["momentum_buffer"],
    )
