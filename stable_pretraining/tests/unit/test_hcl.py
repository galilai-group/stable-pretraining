"""Unit tests for HCL (Contrastive Learning with Hard Negative Samples)."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.losses import NTXEntLoss
from stable_pretraining.methods import HCL
from stable_pretraining.methods.hcl import HCLLoss

pytestmark = pytest.mark.unit


def official_hcl_criterion(out_1, out_2, tau_plus, batch_size, beta, temperature):
    """Reference loss from the official HCL implementation.

    Adapted from ``criterion(..., estimator="hard")`` in ``image/main.py`` of
    https://github.com/joshr17/HCL (MIT License, Copyright (c) 2020 Josh
    Robinson). Changes: inputs are normalised here (the official ``Model``
    returns normalised projections), ``temperature`` is an argument instead of
    a global, and the negative mask is built inline on the inputs' device.
    """
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        mask[i, i] = 0
        mask[i, i + batch_size] = 0
    mask = torch.cat((mask, mask), 0).to(out.device)
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    N = batch_size * 2 - 2
    imp = (beta * neg.log()).exp()
    reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
    Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
    Ng_clamped = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
    loss = (-torch.log(pos / (pos + Ng_clamped))).mean()
    return loss, bool((Ng < Ng_clamped).any())


def _pair(batch_size=16, dim=32, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch_size, dim, generator=g, dtype=dtype),
        torch.randn(batch_size, dim, generator=g, dtype=dtype),
    )


# --- loss correctness -----------------------------------------------------


@pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("batch_size", [2, 8, 64])
def test_reduces_to_ntxent(temperature, batch_size):
    z_i, z_j = _pair(batch_size, dtype=torch.float32)
    hcl = HCLLoss(temperature=temperature, beta=0.0, tau_plus=0.0)(z_i, z_j)
    ntxent = NTXEntLoss(temperature=temperature)(z_i, z_j)
    torch.testing.assert_close(hcl, ntxent, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("temperature", [0.2, 0.5, 1.0])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 2.0, 6.0])
@pytest.mark.parametrize("tau_plus", [0.0, 0.1, 0.5])
def test_matches_official_implementation(temperature, beta, tau_plus):
    z_i, z_j = _pair(batch_size=16)
    ours = HCLLoss(temperature=temperature, beta=beta, tau_plus=tau_plus)(z_i, z_j)
    ref, _ = official_hcl_criterion(z_i, z_j, tau_plus, 16, beta, temperature)
    torch.testing.assert_close(ours, ref, rtol=1e-10, atol=1e-10)


def test_matches_official_when_clamp_is_active():
    # Nearly identical views make the debiasing term exceed the negatives term,
    # which is exactly when the official lower clamp kicks in.
    z_i, _ = _pair(batch_size=16)
    z_j = z_i + 1e-3 * torch.randn_like(z_i)
    ours = HCLLoss(temperature=0.5, beta=1.0, tau_plus=0.5)(z_i, z_j)
    ref, clamp_active = official_hcl_criterion(z_i, z_j, 0.5, 16, 1.0, 0.5)
    assert clamp_active
    torch.testing.assert_close(ours, ref, rtol=1e-10, atol=1e-10)


def test_debiased_special_case_matches_official():
    z_i, z_j = _pair(batch_size=32)
    ours = HCLLoss(temperature=0.5, beta=0.0, tau_plus=0.1)(z_i, z_j)
    ref, _ = official_hcl_criterion(z_i, z_j, 0.1, 32, 0.0, 0.5)
    torch.testing.assert_close(ours, ref, rtol=1e-10, atol=1e-10)


def test_stable_where_official_overflows():
    z_i, z_j = _pair(batch_size=32, dtype=torch.float32)
    z_i, z_j = z_i.requires_grad_(), z_j.requires_grad_()
    # A near-duplicate negative at low temperature overflows exp(s / t) in float32.
    with torch.no_grad():
        z_i[1] = z_i[0]

    ref, _ = official_hcl_criterion(z_i, z_j, 0.1, 32, 6.0, 0.02)
    assert not torch.isfinite(ref)

    loss = HCLLoss(temperature=0.02, beta=6.0, tau_plus=0.1)(z_i, z_j)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(z_i.grad).all() and torch.isfinite(z_j.grad).all()


def test_loss_increases_with_hardness():
    # The reweighted negatives term is a weighted mean of exp(s) with weights
    # proportional to exp(beta * s), so it can only grow as beta grows.
    z_i, z_j = _pair(batch_size=32)
    losses = [
        HCLLoss(temperature=0.5, beta=beta, tau_plus=0.0)(z_i, z_j).item()
        for beta in [0.0, 0.5, 1.0, 2.0, 4.0]
    ]
    assert all(a < b for a, b in zip(losses, losses[1:]))


def test_symmetric_in_views():
    z_i, z_j = _pair()
    loss_fn = HCLLoss()
    torch.testing.assert_close(loss_fn(z_i, z_j), loss_fn(z_j, z_i))


def test_invariant_to_projection_scale():
    z_i, z_j = _pair()
    loss_fn = HCLLoss()
    torch.testing.assert_close(loss_fn(z_i, z_j), loss_fn(3.0 * z_i, 0.5 * z_j))


def test_gradients_reach_both_views():
    z_i, z_j = _pair(dtype=torch.float32)
    z_i.requires_grad_()
    z_j.requires_grad_()
    HCLLoss()(z_i, z_j).backward()
    assert z_i.grad.abs().sum() > 0
    assert z_j.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.0},
        {"temperature": -0.5},
        {"beta": -1.0},
        {"tau_plus": 1.0},
        {"tau_plus": -0.1},
    ],
)
def test_invalid_hyperparameters_raise(kwargs):
    with pytest.raises(ValueError):
        HCLLoss(**kwargs)


@pytest.mark.parametrize(
    "shapes",
    [((8, 16), (8, 32)), ((8, 16), (4, 16)), ((16,), (16,))],
)
def test_invalid_input_shapes_raise(shapes):
    with pytest.raises(ValueError):
        HCLLoss()(torch.randn(*shapes[0]), torch.randn(*shapes[1]))


def test_single_sample_batch_raises():
    with pytest.raises(ValueError):
        HCLLoss()(torch.randn(1, 16), torch.randn(1, 16))


# --- method class and integration ----------------------------------------


def _tiny_backbone():
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 32))


def test_method_train_forward():
    model = HCL(encoder_name=_tiny_backbone(), projector_dims=(16, 8))
    model.train()
    v1, v2 = torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8)
    out = model(v1, v2)

    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    assert out.embedding.shape == (8, 32)
    assert out.projection.shape == (8, 8)
    out.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )


def test_method_eval_forward_returns_embedding_only():
    model = HCL(encoder_name=_tiny_backbone(), projector_dims=(16, 8))
    model.eval()
    out = model(torch.randn(4, 3, 8, 8))

    assert out.embedding.shape == (4, 32)
    assert out.projection is None
    assert out.loss.item() == 0


def test_method_uses_official_projector_and_hyperparameters():
    model = HCL(encoder_name=_tiny_backbone(), temperature=0.3, beta=2.0, tau_plus=0.2)
    layers = list(model.projector)

    assert isinstance(layers[0], nn.Linear) and layers[0].bias is None
    assert isinstance(layers[1], nn.BatchNorm1d)
    assert isinstance(layers[2], nn.ReLU)
    assert isinstance(layers[3], nn.Linear) and layers[3].out_features == 128
    assert (
        model.hcl_loss.temperature,
        model.hcl_loss.beta,
        model.hcl_loss.tau_plus,
    ) == (
        0.3,
        2.0,
        0.2,
    )


def test_drop_in_loss_for_simclr_forward():
    module = spt.Module(
        backbone=nn.Linear(8, 16),
        projector=nn.Linear(16, 4),
        forward=forward.simclr,
        simclr_loss=HCLLoss(),
    )
    views = [{"image": torch.randn(6, 8), "label": torch.arange(6)} for _ in range(2)]

    module.train()
    out = module.forward({"views": views}, stage="train")
    assert torch.isfinite(out["loss"])
    assert out["embedding"].shape == (12, 16)

    module.eval()
    out = module.forward(
        {"image": torch.randn(6, 8), "label": torch.arange(6)}, stage="val"
    )
    assert "loss" not in out
    assert out["embedding"].shape == (6, 16)
