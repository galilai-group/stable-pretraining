"""Unit tests for DirectPred (BYOL with a directly set linear predictor)."""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

import stable_pretraining.utils as spt_utils
from stable_pretraining import forward as forward_module
from stable_pretraining.backbone import DirectPredictor, TeacherStudentWrapper
from stable_pretraining.losses import BYOLLoss
from stable_pretraining.methods import DirectPred

pytestmark = pytest.mark.unit

N, D = 32, 8


def _batches(n_steps, n=N, d=D, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, d, generator=g) for _ in range(n_steps)]


def _corr(z):
    """Uncentered correlation E_B[f f^T] of the rows of ``z``."""
    return z.T @ z / z.size(0)


def _random_rotation(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    return q


# --- closed-form weight (Eq. 18) ----------------------------------------------


def test_closed_form_weight_on_diagonal_spectrum():
    """p_j = sqrt(s_j / max_k s_k) + eps, worked out by hand."""
    corr = torch.diag(torch.tensor([4.0, 1.0, 0.0]))
    w = DirectPredictor(3, eps=0.1).closed_form_weight(corr)
    torch.testing.assert_close(w, torch.diag(torch.tensor([1.1, 0.6, 0.1])))
    # Without the boost, a zero eigenvalue is floored at 1e-4.
    w = DirectPredictor(3, eps=0.0).closed_form_weight(corr)
    torch.testing.assert_close(w, torch.diag(torch.tensor([1.0, 0.5, 1e-4])))


def test_closed_form_weight_uses_the_eigenvectors_of_corr():
    """Built from a known eigenbasis U: W_p = U diag(p) U^T."""
    u = _random_rotation(D)
    s = torch.linspace(0.05, 2.0, D)
    corr = u @ torch.diag(s) @ u.T
    w = DirectPredictor(D, eps=0.1).closed_form_weight(corr)
    expected = u @ torch.diag((s / s.max()).sqrt() + 0.1) @ u.T
    torch.testing.assert_close(w, expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(w, w.T)


def test_without_boost_weight_is_square_root_of_corr():
    """With eps = 0, W_p is proportional to F^(1/2): W_p @ W_p = F / s_max."""
    g = torch.Generator().manual_seed(1)
    a = torch.randn(D, D, generator=g)
    corr = a @ a.T + 0.1 * torch.eye(D)  # well-conditioned, no floored eigenvalue
    w = DirectPredictor(D, eps=0.0).closed_form_weight(corr)
    s_max = torch.linalg.matrix_norm(corr, ord=2)
    torch.testing.assert_close(w @ w, corr / s_max, rtol=1e-4, atol=1e-5)


# --- running estimate (Eq. 19) ------------------------------------------------


def test_update_matches_running_average_reference():
    pred = DirectPredictor(D, rho=0.3, eps=0.1)
    expected = None
    for z in _batches(4):
        batch_corr = _corr(z)
        expected = batch_corr if expected is None else 0.3 * expected + 0.7 * batch_corr
        pred.update(z)
        torch.testing.assert_close(pred.corr, expected)
    assert pred.num_updates.item() == 4


def test_correlation_is_uncentered():
    """Inputs with a nonzero mean give the second moment, not the covariance."""
    (z,) = _batches(1)
    z = z + 3.0
    pred = DirectPredictor(D)
    pred.update(z)
    torch.testing.assert_close(pred.corr, _corr(z))
    centered = _corr(z - z.mean(0))
    assert (pred.corr - centered).abs().max() > 5.0


def test_weight_does_not_depend_on_the_scale_of_the_projections():
    (z,) = _batches(1)
    a, b = DirectPredictor(D), DirectPredictor(D)
    a.update(z)
    b.update(7.0 * z)
    torch.testing.assert_close(b.corr, 49.0 * a.corr)
    torch.testing.assert_close(b.weight, a.weight, rtol=1e-4, atol=1e-5)


def test_weight_is_recomputed_at_every_update():
    pred = DirectPredictor(D)
    for z in _batches(3):
        pred.update(z)
        torch.testing.assert_close(pred.weight, pred.closed_form_weight(pred.corr))


def test_update_gathers_projections_across_processes(monkeypatch):
    """Under DDP the estimate uses the projections of every process."""
    (z,) = _batches(1)
    (other,) = _batches(1, seed=1)
    monkeypatch.setattr(spt_utils, "all_gather", lambda t: [t, other])
    pred = DirectPredictor(D)
    pred.update(z)
    torch.testing.assert_close(pred.corr, _corr(torch.cat([z, other])))


def test_forward_is_linear_and_weight_is_not_trained():
    pred = DirectPredictor(D)
    pred.update(_batches(1)[0])
    x = torch.randn(5, D, requires_grad=True)
    out = pred(x)
    torch.testing.assert_close(out, x @ pred.weight.T)
    out.sum().backward()
    assert x.grad is not None
    assert len(list(pred.parameters())) == 0


def test_update_after_forward_keeps_backward_valid():
    """Backward stays valid after the weight is refreshed.

    The weight is replaced rather than modified in place, so backward uses
    the weight the predictions were made with.
    """
    pred = DirectPredictor(D)
    z1, z2 = _batches(2)
    pred.update(z1)
    w_used = pred.weight.clone()
    x = torch.randn(5, D, requires_grad=True)
    out = pred(x)
    pred.update(z2)
    assert not torch.allclose(pred.weight, w_used)
    out.sum().backward()
    torch.testing.assert_close(x.grad, w_used.sum(0).expand(5, D))


def test_state_dict_round_trip_resumes_the_estimate():
    batches = _batches(4)
    pred = DirectPredictor(D)
    for z in batches[:3]:
        pred.update(z)
    resumed = DirectPredictor(D)
    resumed.load_state_dict(pred.state_dict())
    pred.update(batches[3])
    resumed.update(batches[3])
    torch.testing.assert_close(resumed.corr, pred.corr)
    torch.testing.assert_close(resumed.weight, pred.weight)
    assert resumed.num_updates.item() == pred.num_updates.item() == 4


def test_update_under_autocast_stays_float32():
    z = _batches(1)[0].to(torch.bfloat16)
    pred = DirectPredictor(D)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        pred.update(z)
    assert pred.corr.dtype == torch.float32
    assert pred.weight.dtype == torch.float32
    torch.testing.assert_close(pred.corr, _corr(z.float()))


def test_invalid_arguments_raise():
    with pytest.raises(ValueError, match="rho"):
        DirectPredictor(D, rho=1.0)
    with pytest.raises(ValueError, match="eps"):
        DirectPredictor(D, eps=-0.1)
    with pytest.raises(ValueError, match="dim=8"):
        DirectPredictor(D).update(torch.randn(4, D + 1))


# --- directpred forward function --------------------------------------------


def _mock_module(training=True):
    module = Mock()
    module.backbone = TeacherStudentWrapper(nn.Linear(12, 16))
    module.projector = TeacherStudentWrapper(nn.Linear(16, D))
    # Make the teacher differ from the student, so the test can tell them apart.
    with torch.no_grad():
        for p in module.projector.teacher.parameters():
            p.add_(1.0)
    module.predictor = DirectPredictor(D)
    module.byol_loss = BYOLLoss()
    module.training = training
    module.log = Mock()
    return module


def _view(n=N, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"image": torch.randn(n, 12, generator=g), "label": torch.arange(n)}


@torch.no_grad()
def _projections(module, views):
    online = [
        module.projector.forward_student(module.backbone.forward_student(v["image"]))
        for v in views
    ]
    target = [
        module.projector.forward_teacher(module.backbone.forward_teacher(v["image"]))
        for v in views
    ]
    return online, target


def _expected_loss(online, target, weight):
    pred = [f @ weight.T for f in online]
    return (BYOLLoss()(pred[0], target[1]) + BYOLLoss()(pred[1], target[0])) / 2


def test_forward_function_feeds_online_projections_of_both_views():
    """F-hat is fed by the online projections of both views."""
    module = _mock_module()
    views = [_view(seed=1), _view(seed=2)]
    out = forward_module.directpred(module, {"views": views}, "fit")
    online, target = _projections(module, views)
    torch.testing.assert_close(module.predictor.corr, _corr(torch.cat(online)))
    # The predictor starts at the identity, so the first loss used W_p = I.
    torch.testing.assert_close(
        out["loss"].detach(), _expected_loss(online, target, torch.eye(D))
    )
    assert out["embedding"].shape == (2 * N, 16)
    assert out["label"].shape == (2 * N,)
    assert module.log.call_args.args[0] == "fit/loss"


def test_step_uses_the_predictor_from_the_previous_step():
    """The loss of step t is computed with W_p set at the end of step t - 1."""
    module = _mock_module()
    forward_module.directpred(module, {"views": [_view(seed=1), _view(seed=2)]}, "fit")
    w_prev = module.predictor.weight.clone()
    views = [_view(seed=3), _view(seed=4)]
    out = forward_module.directpred(module, {"views": views}, "fit")
    online, target = _projections(module, views)
    torch.testing.assert_close(
        out["loss"].detach(), _expected_loss(online, target, w_prev)
    )
    assert not torch.allclose(module.predictor.weight, w_prev)


def test_forward_function_requires_two_views():
    with pytest.raises(ValueError, match="exactly 2 views"):
        forward_module.directpred(
            _mock_module(), {"views": [_view(), _view(), _view()]}, "fit"
        )


def test_forward_function_requires_byol_loss():
    module = _mock_module()
    del module.byol_loss
    with pytest.raises(ValueError, match="byol_loss"):
        forward_module.directpred(module, {"views": [_view(), _view()]}, "fit")


def test_forward_function_eval_does_not_update_predictor():
    module = _mock_module(training=False)
    out = forward_module.directpred(module, {"views": [_view(), _view()]}, "val")
    assert out["embedding"].shape == (2 * N, 16)
    assert "loss" not in out
    out = forward_module.directpred(module, _view(), "val")
    assert out["embedding"].shape == (N, 16)
    torch.testing.assert_close(out["label"], torch.arange(N))
    assert module.predictor.num_updates.item() == 0


# --- DirectPred method class ------------------------------------------------


def _tiny_backbone():
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )


def test_method_train_and_eval_paths():
    model = DirectPred(encoder_name=_tiny_backbone(), projector_dims=(16, 4))
    assert model.predictor.eps == 0.1
    v1, v2 = torch.randn(N, 3, 16, 16), torch.randn(N, 3, 16, 16)

    model.train()
    out = model(v1, v2)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    assert out.prediction.shape == (2 * N, 4)
    assert out.target.shape == (2 * N, 4)
    assert model.predictor.num_updates.item() == 1

    model.eval()
    out = model(v1, v2)
    assert model.predictor.num_updates.item() == 1
    out = model(v1)
    assert out.loss.item() == 0.0
    assert out.embedding.shape == (N, 8)
    assert out.prediction is None


def test_method_rejects_bad_projector_dims():
    with pytest.raises(ValueError, match="projector_dims"):
        DirectPred(encoder_name=_tiny_backbone(), projector_dims=(16, 8, 4))
