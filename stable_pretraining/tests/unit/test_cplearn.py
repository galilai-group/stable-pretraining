"""Unit tests for CPLearn: loss, projector, method class and forward function."""

import math
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_pretraining import forward as forward_module
from stable_pretraining.losses import CPLearnLoss
from stable_pretraining.methods import CPLearn
from stable_pretraining.methods.cplearn import CPLearnProjector

pytestmark = pytest.mark.unit


def _logits(n=8, c=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, c, generator=g), torch.randn(n, c, generator=g)


def _hard_assignments(assigned_codes, c, eps):
    """Log-probabilities at the extrema of the simplex (Lemma 1)."""
    n = len(assigned_codes)
    probs = torch.full((n, c), eps, dtype=torch.float64)
    probs[torch.arange(n), assigned_codes] = 1 - eps * (c - 1)
    return probs.log()


def _tiny_encoder(out_dim=16):
    return nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(48, out_dim))


# --- CPLearnLoss ---------------------------------------------------------


def test_loss_one_directional_ce_prior_matches_eq2():
    l1, l2 = _logits()
    beta = 0.3
    p1, p2 = l1.softmax(-1), l2.softmax(-1)
    c = l1.size(1)
    # Eq. 2: -(beta/n) sum_ij p_ij log p'_ij - sum_j q_j log(1/n sum_i p_ij), q_j = 1/c
    expected = -beta * (p1 * p2.log()).sum(1).mean() - (p1.mean(0).log() / c).sum()
    loss = CPLearnLoss(beta=beta, prior="ce", symmetric=False)(l1, l2)
    torch.testing.assert_close(loss, expected)


def test_loss_defaults_match_official_implementation():
    """Defaults reproduce ``cplearn_loss_func`` from github.com/emsansone/CPLearn."""
    l1, l2 = _logits()
    beta = 0.7
    c = l1.size(1)

    def one_direction(target_logits, pred_logits):
        # Reverse-KL prior on the target view plus beta * CE(target, pred), both
        # written per sample as in the official code.
        p = target_logits.softmax(1)
        prior = (p * torch.log(c * p.mean(0, keepdim=True))).sum(1).mean()
        log_pred = pred_logits - pred_logits.logsumexp(1, keepdim=True)
        return prior - beta * (p * log_pred).sum(1).mean()

    expected = 0.5 * one_direction(l2, l1) + 0.5 * one_direction(l1, l2)
    torch.testing.assert_close(CPLearnLoss(beta=beta)(l1, l2), expected)


@pytest.mark.parametrize("prior", ["ce", "reverse_kl"])
def test_symmetric_loss_averages_both_directions(prior):
    l1, l2 = _logits()
    one_way = CPLearnLoss(beta=0.5, prior=prior, symmetric=False)
    both_ways = CPLearnLoss(beta=0.5, prior=prior, symmetric=True)
    torch.testing.assert_close(
        both_ways(l1, l2), 0.5 * (one_way(l1, l2) + one_way(l2, l1))
    )


@pytest.mark.parametrize("eps", [1e-6, 1e-8])
def test_ce_prior_loss_attains_lemma1_lower_bound(eps):
    """Balanced hard assignments with P == P' reach the bound of Eq. 3."""
    c, beta = 8, 0.5
    logits = _hard_assignments(torch.arange(2 * c) % c, c, eps)
    loss = CPLearnLoss(beta=beta, prior="ce", symmetric=False)(logits, logits)
    a = 1 - eps * (c - 1)
    bound = -beta * a * math.log(a) - beta * eps * (c - 1) * math.log(eps)
    bound += math.log(c)  # entropy of the uniform prior
    assert loss.item() == pytest.approx(bound, rel=1e-5)


@pytest.mark.parametrize("prior", ["ce", "reverse_kl"])
def test_prior_term_penalizes_cluster_collapse(prior):
    c, eps = 8, 1e-8
    balanced = _hard_assignments(torch.arange(2 * c) % c, c, eps)
    collapsed = _hard_assignments(torch.zeros(2 * c, dtype=torch.long), c, eps)
    loss_fn = CPLearnLoss(beta=0.0, prior=prior)
    if prior == "reverse_kl":
        assert loss_fn(balanced, balanced).item() == pytest.approx(0.0, abs=1e-6)
        assert loss_fn(collapsed, collapsed).item() == pytest.approx(math.log(c), 1e-4)
    else:
        assert loss_fn(balanced, balanced).item() == pytest.approx(math.log(c), 1e-6)
        assert loss_fn(collapsed, collapsed) > 2 * math.log(c)


@pytest.mark.parametrize("symmetric", [False, True])
def test_gradients_flow_through_both_views(symmetric):
    l1, l2 = (t.requires_grad_() for t in _logits())
    CPLearnLoss(symmetric=symmetric)(l1, l2).backward()
    assert l1.grad.abs().sum() > 0 and l2.grad.abs().sum() > 0


def test_gather_distributed_is_noop_in_single_process():
    l1, l2 = _logits()
    torch.testing.assert_close(
        CPLearnLoss(gather_distributed=True)(l1, l2), CPLearnLoss()(l1, l2)
    )


def test_loss_is_computed_in_float32_for_low_precision_logits():
    l1, l2 = _logits(c=4096)
    loss = CPLearnLoss()(l1.bfloat16() * 50, l2.bfloat16() * 50)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)


def test_loss_rejects_unknown_prior():
    with pytest.raises(ValueError, match="prior"):
        CPLearnLoss(prior="kl")


@pytest.mark.parametrize("symmetric", [False, True])
def test_terms_decompose_the_loss(symmetric):
    l1, l2 = _logits()
    loss_fn = CPLearnLoss(beta=0.3, prior="ce", symmetric=symmetric)
    terms = loss_fn.terms(l1, l2)
    torch.testing.assert_close(terms["loss"], loss_fn(l1, l2))
    torch.testing.assert_close(
        terms["loss"], 0.3 * terms["invariance"] + terms["prior"]
    )
    p1, p2 = l1.softmax(-1), l2.softmax(-1)
    invariance = -(p1 * p2.log()).sum(1).mean()
    prior = -(p1.mean(0).log() / l1.size(1)).sum()
    if symmetric:
        invariance = 0.5 * (invariance - (p2 * p1.log()).sum(1).mean())
        prior = 0.5 * (prior - (p2.mean(0).log() / l1.size(1)).sum())
    torch.testing.assert_close(terms["invariance"], invariance)
    torch.testing.assert_close(terms["prior"], prior)


# --- CPLearnProjector ----------------------------------------------------


def test_projector_codes_are_frozen_rademacher_buffer():
    proj = CPLearnProjector(16, projector_dim=32, dictionary_size=64, seed=0)
    assert proj.codes.shape == (32, 64)
    assert set(proj.codes.unique().tolist()) == {-1.0, 1.0}
    assert "codes" in proj.state_dict()
    assert all(p is not proj.codes for p in proj.parameters())
    assert not proj.codes.requires_grad


def test_projector_seed_controls_codes():
    a = CPLearnProjector(16, projector_dim=32, dictionary_size=64, seed=0)
    b = CPLearnProjector(16, projector_dim=32, dictionary_size=64, seed=0)
    c = CPLearnProjector(16, projector_dim=32, dictionary_size=64, seed=1)
    assert torch.equal(a.codes, b.codes)
    assert not torch.equal(a.codes, c.codes)


def test_l2_logits_reduce_to_scaled_cosine_similarity():
    """With Eq. 1, sqrt(f/n) cancels the sqrt(n) in tau: logits = log_odds * cos."""
    proj = CPLearnProjector(
        16, projector_dim=32, dictionary_size=64, activation="l2", seed=0
    ).eval()
    x = torch.randn(8, 16)
    with torch.no_grad():
        h = proj.bn(proj.linear(x))
        expected = (
            proj.log_odds * F.normalize(h, dim=-1) @ F.normalize(proj.codes, dim=0)
        )
        torch.testing.assert_close(proj(x), expected)
        # Independent of the batch size.
        torch.testing.assert_close(proj(x[:4]), proj(x)[:4])


def test_tanh_logits_follow_official_code_and_scale_with_sqrt_batch_size():
    proj = CPLearnProjector(16, projector_dim=32, dictionary_size=64, seed=0).eval()
    x = torch.randn(8, 16)
    with torch.no_grad():
        h = torch.tanh(proj.bn(proj.linear(x)))
        expected = math.sqrt(8) * proj.log_odds * (h @ proj.codes) / 32
        torch.testing.assert_close(proj(x), expected)
        torch.testing.assert_close(proj(x)[:4], math.sqrt(2) * proj(x[:4]))


def test_log_odds_uses_epsilon():
    proj = CPLearnProjector(4, projector_dim=8, dictionary_size=10, epsilon=1e-6)
    assert proj.log_odds == pytest.approx(math.log((1 - 9e-6) / 1e-6))


@pytest.mark.parametrize("eps", [0.0, 0.1, 1.0])
def test_projector_rejects_epsilon_outside_lemma1_range(eps):
    with pytest.raises(ValueError, match="epsilon"):
        CPLearnProjector(4, projector_dim=8, dictionary_size=10, epsilon=eps)


def test_projector_rejects_unknown_activation():
    with pytest.raises(ValueError, match="activation"):
        CPLearnProjector(4, activation="relu")


# --- CPLearn method class ------------------------------------------------


def test_method_forward_backward_leaves_codes_untouched():
    model = CPLearn(
        encoder_name=_tiny_encoder(), projector_dim=32, dictionary_size=64, codes_seed=0
    ).train()
    codes = model.projector.codes.clone()
    v1, v2 = torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)
    out = model(v1, v2)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    assert out.embedding.shape == (8, 16)
    assert out.logits.shape == (8, 64)
    torch.testing.assert_close(out.loss.detach(), out.invariance + out.prior)
    out.loss.backward()
    assert model.projector.linear.weight.grad.abs().sum() > 0
    assert model.backbone[-1].weight.grad.abs().sum() > 0
    assert torch.equal(model.projector.codes, codes)


def test_method_single_view_returns_embedding_and_zero_loss():
    model = CPLearn(
        encoder_name=_tiny_encoder(), projector_dim=32, dictionary_size=64
    ).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32))
    assert out.embedding.shape == (2, 16)
    assert out.loss.item() == 0 and out.logits is None


def test_method_passes_options_to_projector_and_loss():
    model = CPLearn(
        encoder_name=_tiny_encoder(),
        projector_dim=32,
        dictionary_size=64,
        beta=0.25,
        activation="l2",
        epsilon=1e-6,
        prior="ce",
        symmetric=False,
        gather_distributed=True,
    )
    assert model.projector.activation == "l2"
    assert model.projector.epsilon == 1e-6
    loss = model.cplearn_loss
    assert (loss.beta, loss.prior, loss.symmetric, loss.gather_distributed) == (
        0.25,
        "ce",
        False,
        True,
    )


# --- forward function ----------------------------------------------------


def _module_for_forward():
    module = Mock()
    module.backbone = _tiny_encoder()
    module.projector = CPLearnProjector(16, projector_dim=32, dictionary_size=64)
    module.cplearn_loss = CPLearnLoss()
    module.training = True
    module.log = Mock()
    module.log_dict = Mock()
    return module


def test_forward_function_trains_on_two_views():
    module = _module_for_forward()
    views = [
        {"image": torch.randn(4, 3, 32, 32), "label": torch.arange(4)} for _ in range(2)
    ]
    out = forward_module.cplearn(module, {"views": views}, "train")
    assert out["embedding"].shape == (8, 16)
    assert out["label"].shape == (8,)
    assert torch.isfinite(out["loss"])
    module.log.assert_called_once()
    assert module.log.call_args.args[0] == "train/loss"
    logged = module.log_dict.call_args.args[0]
    assert set(logged) == {"train/cplearn_invariance", "train/cplearn_prior"}
    torch.testing.assert_close(
        out["loss"].detach(),
        logged["train/cplearn_invariance"] + logged["train/cplearn_prior"],
    )


def test_forward_function_single_view_validation():
    module = _module_for_forward()
    module.training = False
    batch = {"image": torch.randn(4, 3, 32, 32), "label": torch.arange(4)}
    out = forward_module.cplearn(module, batch, "val")
    assert out["embedding"].shape == (4, 16)
    assert "loss" not in out
    assert torch.equal(out["label"], batch["label"])


def test_forward_function_requires_two_views():
    module = _module_for_forward()
    views = [{"image": torch.randn(2, 3, 32, 32)} for _ in range(3)]
    with pytest.raises(ValueError, match="exactly 2 views"):
        forward_module.cplearn(module, {"views": views}, "train")
