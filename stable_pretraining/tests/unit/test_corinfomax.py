"""Unit tests for CorInfoMax (Correlative Information Maximization)."""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_pretraining import forward as forward_module
from stable_pretraining.losses import CorInfoMaxLoss
from stable_pretraining.methods import CorInfoMax
from stable_pretraining.methods.corinfomax import _build_corinfomax_projector

pytestmark = pytest.mark.unit

N, D = 16, 8


def _batches(n_steps, n=N, d=D, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(n, d, generator=g, dtype=dtype),
            torch.randn(n, d, generator=g, dtype=dtype),
        )
        for _ in range(n_steps)
    ]


def _reference_losses(batches, alpha=250.0, la_r=0.01, la_mu=0.01, r_ini=1.0, eps=1e-8):
    """Equation-level CorInfoMax loss in float64, written out step by step.

    mu_t = la_mu * mu_{t-1} + (1 - la_mu) * mean(z)
    R_t = la_r * R_{t-1} + (1 - la_r) * (z - mu_t)^T (z - mu_t) / N
    L = alpha * MSE(z1, z2) - (logdet(R1_t + eps I) + logdet(R2_t + eps I)) / D
    """
    eye = torch.eye(D, dtype=torch.float64)
    cov = [r_ini * eye, r_ini * eye]
    mean = [torch.zeros(D, dtype=torch.float64)] * 2
    losses = []
    for z1, z2 in batches:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logdets = []
        for k, z in enumerate((z1.double(), z2.double())):
            mean[k] = la_mu * mean[k] + (1 - la_mu) * z.mean(dim=0)
            z_c = z - mean[k]
            cov[k] = la_r * cov[k] + (1 - la_r) * z_c.T @ z_c / z.size(0)
            logdets.append(torch.logdet(cov[k] + eps * eye))
        mse = F.mse_loss(z1, z2).double()
        losses.append(alpha * mse - (logdets[0] + logdets[1]) / D)
    return losses


# --- CorInfoMaxLoss ----------------------------------------------------------


def test_loss_matches_reference_over_consecutive_steps():
    """The EMA estimates carry over between calls exactly as in the paper."""
    batches = _batches(3)
    loss_fn = CorInfoMaxLoss(proj_dim=D).train()
    expected = _reference_losses(batches)
    for (z1, z2), ref in zip(batches, expected):
        torch.testing.assert_close(loss_fn(z1, z2), ref, rtol=0, atol=1e-10)


def test_buffers_initialized_in_float64():
    loss_fn = CorInfoMaxLoss(proj_dim=D, init_scale=0.5)
    for name in ("cov_i", "cov_j"):
        buf = getattr(loss_fn, name)
        assert buf.dtype == torch.float64
        torch.testing.assert_close(buf, 0.5 * torch.eye(D, dtype=torch.float64))
    for name in ("mean_i", "mean_j"):
        buf = getattr(loss_fn, name)
        assert buf.dtype == torch.float64
        assert torch.count_nonzero(buf) == 0
    assert {"cov_i", "cov_j", "mean_i", "mean_j"} <= set(loss_fn.state_dict())


def test_buffers_update_only_in_training_mode():
    ((z1, z2),) = _batches(1)
    loss_fn = CorInfoMaxLoss(proj_dim=D).eval()
    before = {k: v.clone() for k, v in loss_fn.state_dict().items()}
    loss_fn(z1, z2)
    for k, v in loss_fn.state_dict().items():
        torch.testing.assert_close(v, before[k])

    loss_fn.train()(z1, z2)
    assert not torch.equal(loss_fn.cov_i, before["cov_i"])
    assert not torch.equal(loss_fn.mean_j, before["mean_j"])


def test_eval_mode_loss_uses_current_estimates_without_changing_them():
    batches = _batches(2)
    loss_fn = CorInfoMaxLoss(proj_dim=D).train()
    loss_fn(*batches[0])
    loss_fn.eval()
    first = loss_fn(*batches[1])
    second = loss_fn(*batches[1])
    torch.testing.assert_close(first, second)


def test_gradients_reach_both_views_and_buffers_hold_no_graph():
    loss_fn = CorInfoMaxLoss(proj_dim=D).train()
    for z1, z2 in _batches(2, dtype=torch.float32):
        z1.requires_grad_(True)
        z2.requires_grad_(True)
        # A graph kept inside the buffers would make the second step raise
        # "Trying to backward through the graph a second time".
        loss_fn(z1, z2).backward()
        for z in (z1, z2):
            assert z.grad is not None
            assert torch.isfinite(z.grad).all()
            assert z.grad.abs().sum() > 0
    for buf in loss_fn.buffers():
        assert not buf.requires_grad
        assert buf.grad_fn is None


def test_state_dict_round_trip_resumes_the_estimates():
    batches = _batches(4)
    loss_fn = CorInfoMaxLoss(proj_dim=D).train()
    for z1, z2 in batches[:3]:
        loss_fn(z1, z2)

    resumed = CorInfoMaxLoss(proj_dim=D).train()
    resumed.load_state_dict(loss_fn.state_dict())
    torch.testing.assert_close(resumed(*batches[3]), loss_fn(*batches[3]))


def test_big_bang_term_penalizes_collapse():
    g = torch.Generator().manual_seed(0)
    spread = torch.randn(64, D, generator=g)
    collapsed = spread[:1].expand(64, D).clone()

    def cov_term(z):
        loss_fn = CorInfoMaxLoss(proj_dim=D, sim_coeff=0.0).train()
        return loss_fn(z, z).item()

    assert cov_term(collapsed) > cov_term(spread) + 1.0


def test_rank_deficient_batches_stay_finite():
    """Batches smaller than the projection dimension are still well defined."""
    loss_fn = CorInfoMaxLoss(proj_dim=32).train()
    for z1, z2 in _batches(10, n=4, d=32, dtype=torch.float32):
        assert torch.isfinite(loss_fn(z1, z2))


def test_normalize_makes_loss_invariant_to_row_scaling():
    ((z1, z2),) = _batches(1)
    scale = torch.rand(N, 1, dtype=torch.float64) + 0.5

    def fresh_loss(normalize, a, b):
        return CorInfoMaxLoss(proj_dim=D, normalize=normalize).train()(a, b)

    torch.testing.assert_close(
        fresh_loss(True, z1, z2), fresh_loss(True, scale * z1, scale * z2)
    )
    assert not torch.allclose(
        fresh_loss(False, z1, z2), fresh_loss(False, scale * z1, scale * z2)
    )


def test_autocast_keeps_estimates_in_float64():
    ((z1, z2),) = _batches(1, dtype=torch.float32)
    loss_fn = CorInfoMaxLoss(proj_dim=D).train()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = loss_fn(z1, z2)
    assert torch.isfinite(loss)
    assert loss_fn.cov_i.dtype == torch.float64
    reference = _reference_losses([(z1, z2)])[0]
    torch.testing.assert_close(loss.double(), reference, rtol=1e-5, atol=1e-5)


def test_wrong_projection_dim_raises():
    loss_fn = CorInfoMaxLoss(proj_dim=D)
    with pytest.raises(ValueError, match="proj_dim=8"):
        loss_fn(torch.randn(N, D + 1), torch.randn(N, D + 1))


# --- CorInfoMax method class -------------------------------------------------


def _tiny_backbone():
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )


def test_projector_matches_paper_architecture():
    projector = _build_corinfomax_projector(16, [32, 32, 4])
    linears = [m for m in projector if isinstance(m, nn.Linear)]
    assert [m.out_features for m in linears] == [32, 32, 4]
    assert linears[-1].bias is None
    assert all(m.bias is not None for m in linears[:-1])
    assert not any(isinstance(m, nn.BatchNorm1d) for m in projector)


def test_projector_requires_dims():
    with pytest.raises(ValueError, match="hidden_dims"):
        _build_corinfomax_projector(16, [])


def test_method_train_and_eval_paths():
    model = CorInfoMax(encoder_name=_tiny_backbone(), projector_dims=(32, 4))
    assert model.corinfomax_loss.proj_dim == 4
    v1, v2 = torch.randn(N, 3, 16, 16), torch.randn(N, 3, 16, 16)

    model.train()
    out = model(v1, v2)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    assert out.embedding.shape == (2 * N, 8)
    assert out.projection.shape == (2 * N, 4)

    model.eval()
    out = model(v1)
    assert out.loss.item() == 0.0
    assert out.embedding.shape == (N, 8)
    assert out.projection is None


# --- corinfomax forward function --------------------------------------------


def _mock_module(training=True):
    module = Mock()
    module.backbone = nn.Linear(12, 6)
    module.projector = nn.Linear(6, D)
    module.corinfomax_loss = CorInfoMaxLoss(proj_dim=D)
    module.training = training
    module.log = Mock()
    return module


def _view(n=N):
    return {"image": torch.randn(n, 12), "label": torch.arange(n)}


def test_forward_function_training_step():
    module = _mock_module()
    out = forward_module.corinfomax(module, {"views": [_view(), _view()]}, "fit")
    assert out["embedding"].shape == (2 * N, 6)
    assert out["label"].shape == (2 * N,)
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    assert module.log.call_args.args[0] == "fit/loss"


def test_forward_function_requires_two_views():
    with pytest.raises(ValueError, match="exactly 2 views"):
        forward_module.corinfomax(
            _mock_module(), {"views": [_view(), _view(), _view()]}, "fit"
        )


def test_forward_function_single_view_validation():
    module = _mock_module(training=False)
    out = forward_module.corinfomax(module, _view(), "validate")
    assert out["embedding"].shape == (N, 6)
    assert "loss" not in out
    torch.testing.assert_close(out["label"], torch.arange(N))
