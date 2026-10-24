"""Unit tests for DenseCL: CPU only, no downloads, tiny tensors, each runs in well under a second."""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_pretraining.forward import densecl
from stable_pretraining.losses.densecl_loss import (
    DenseCLLoss,
    dense_contrastive_loss,
    dense_correspondence,
    global_contrastive_loss,
)

pytestmark = pytest.mark.unit

B, D, K, H, W = 4, 32, 8, 4, 4
S2 = H * W
TEMP = 0.2


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _loss_inputs(with_grad=False):
    t = dict(
        q=torch.randn(B, D),
        k=torch.randn(B, D),
        r=torch.randn(B, D, H, W),
        t=torch.randn(B, D, H, W),
        feat_q=torch.randn(B, K, H, W),
        feat_k=torch.randn(B, K, H, W),
    )
    if with_grad:
        for v in t.values():
            v.requires_grad_()
    return t


# tets for dense correspondence
def test_correspondence_identity_for_identical_maps():
    feat = torch.randn(B, K, H, W)
    idx = dense_correspondence(feat, feat)
    assert idx.shape == (B, S2)
    assert torch.equal(idx, torch.arange(S2).expand(B, S2))


def test_correspondence_recovers_permutation():
    feat_q = torch.randn(B, K, H, W)
    perm = torch.randperm(S2)
    feat_k = feat_q.flatten(2)[:, :, perm].reshape(
        B, K, H, W
    )  # key location p holds query location perm[p]
    idx = dense_correspondence(feat_q, feat_k)
    assert torch.equal(idx, torch.argsort(perm).expand(B, S2))


def test_correspondence_is_integer_and_detached():
    feat_q = torch.randn(B, K, H, W, requires_grad=True)
    idx = dense_correspondence(feat_q, torch.randn(B, K, H, W))
    assert idx.dtype == torch.long
    assert not idx.requires_grad


# tests for global loss
def test_global_loss_in_batch_matches_manual():
    q, k = torch.randn(2, D), torch.randn(2, D)
    qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    expected = []
    for i in range(2):
        j = 1 - i  # the only negative once the own image is excluded
        pos = torch.exp((qn[i] @ kn[i]) / TEMP)
        neg = torch.exp((qn[i] @ kn[j]) / TEMP)
        expected.append(-torch.log(pos / (pos + neg)))
    expected = torch.stack(expected).mean()
    got = global_contrastive_loss(q, k, k, TEMP, exclude_self=True)
    assert torch.allclose(got, expected, atol=1e-5)


def test_global_loss_with_queue_matches_manual():
    q, k, queue = torch.randn(3, D), torch.randn(3, D), torch.randn(10, D)
    qn, kn, qu = (F.normalize(x, dim=-1) for x in (q, k, queue))
    pos = torch.exp((qn * kn).sum(-1) / TEMP)
    neg = torch.exp(qn @ qu.T / TEMP).sum(-1)
    expected = (-torch.log(pos / (pos + neg))).mean()
    got = global_contrastive_loss(q, k, queue, TEMP, exclude_self=False)
    assert torch.allclose(got, expected, atol=1e-5)


# Test for dense loss
def _manual_dense(r, t_all, idx, t_neg, exclude_self):
    r, t_all, t_neg = (F.normalize(x, dim=-1) for x in (r, t_all, t_neg))
    terms = []
    for b in range(r.size(0)):
        for s in range(r.size(1)):
            pos = torch.exp(r[b, s] @ t_all[b, idx[b, s]] / TEMP)
            neg_ids = [n for n in range(t_neg.size(0)) if not (exclude_self and n == b)]
            neg = sum(torch.exp(r[b, s] @ t_neg[n] / TEMP) for n in neg_ids)
            terms.append(-torch.log(pos / (pos + neg)))
    return torch.stack(terms).mean()


def test_dense_loss_with_queue_matches_manual():
    r, t_all = torch.randn(2, 3, D), torch.randn(2, 3, D)
    idx = torch.tensor([[2, 0, 1], [1, 1, 0]])
    t_neg = torch.randn(5, D)
    got = dense_contrastive_loss(r, t_all, idx, t_neg, TEMP, exclude_self=False)
    assert torch.allclose(got, _manual_dense(r, t_all, idx, t_neg, False), atol=1e-5)


def test_dense_loss_exclude_self_matches_manual():
    r, t_all = torch.randn(2, 3, D), torch.randn(2, 3, D)
    idx = torch.tensor([[2, 0, 1], [1, 1, 0]])
    t_neg = torch.randn(2, D)  # one pooled vector per image in the batch
    got = dense_contrastive_loss(r, t_all, idx, t_neg, TEMP, exclude_self=True)
    assert torch.allclose(got, _manual_dense(r, t_all, idx, t_neg, True), atol=1e-5)


# DenseCLLoss
def test_loss_is_finite_nonnegative_scalar():
    loss = DenseCLLoss(TEMP, 0.5)(**_loss_inputs())
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0


@pytest.mark.parametrize("lam", [0.0, 0.3, 1.0])
def test_loss_combines_parts_with_lambda(lam):
    loss, parts = DenseCLLoss(TEMP, lam)(**_loss_inputs(), return_parts=True)
    expected = (1 - lam) * parts["loss_global"] + lam * parts["loss_dense"]
    assert torch.allclose(loss, expected, atol=1e-6)


def test_queue_path_runs():
    loss = DenseCLLoss(TEMP, 0.5)(
        **_loss_inputs(),
        queue_global=torch.randn(32, D),
        queue_dense=torch.randn(32, D),
    )
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_partial_queue_falls_back_to_in_batch():
    fn, inputs = DenseCLLoss(TEMP, 0.5), _loss_inputs()
    assert torch.allclose(fn(**inputs), fn(**inputs, queue_global=torch.randn(32, D)))


def test_aligned_views_score_lower_than_unrelated_views():
    fn = DenseCLLoss(TEMP, 0.5)
    g, dn, f = torch.randn(B, D), torch.randn(B, D, H, W), torch.randn(B, K, H, W)
    aligned = fn(g, g, dn, dn, f, f)
    unrelated = fn(torch.randn(B, D), g, torch.randn(B, D, H, W), dn, f, f)
    assert aligned < unrelated


def test_gradients_reach_queries_but_not_keys():
    inputs = _loss_inputs(with_grad=True)
    DenseCLLoss(TEMP, 0.5)(**inputs).backward()
    for name in ("q", "r"):
        assert inputs[name].grad is not None
        assert torch.isfinite(inputs[name].grad).all()
    for name in ("k", "t", "feat_q", "feat_k"):
        assert inputs[name].grad is None


# forward function tests
class _Host(nn.Module):
    """Minimal stand-in exposing exactly what `densecl` expects from its module."""

    def __init__(self, k=K, d=16, momentum=0.9):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, k, kernel_size=4, stride=4), nn.ReLU()
        )
        self.momentum_backbone = copy.deepcopy(self.backbone)
        for p in self.momentum_backbone.parameters():
            p.requires_grad = False
        self.global_projector = nn.Linear(k, d)
        self.local_projector = nn.Conv2d(k, d, kernel_size=1)
        self.densecl_loss = DenseCLLoss(TEMP, 0.5)
        self.momentum = momentum

    @torch.no_grad()
    def update_momentum_encoder(self):
        for online, target in zip(
            self.backbone.parameters(), self.momentum_backbone.parameters()
        ):
            target.mul_(self.momentum).add_(online, alpha=1 - self.momentum)


def _two_views(with_label=True):
    views = []
    for _ in range(2):
        v = {"image": torch.randn(B, 3, 16, 16)}  # 16x16 to 4x4 feature map
        if with_label:
            v["label"] = torch.randint(0, 10, (B,))
        views.append(v)
    return {"views": views}  # dict form, as _get_views_list expects


def test_forward_train_returns_loss_embedding_label():
    host = _Host().train()
    out = densecl(host, _two_views(), "fit")
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    assert out["embedding"].shape == (2 * B, K)
    assert not out["embedding"].requires_grad
    assert out["label"].shape == (2 * B,)


def test_forward_train_without_labels_omits_label():
    out = densecl(_Host().train(), _two_views(with_label=False), "fit")
    assert "label" not in out


def test_forward_backward_updates_online_only():
    host = _Host().train()
    densecl(host, _two_views(), "fit")["loss"].backward()
    online = [
        *host.backbone.parameters(),
        *host.global_projector.parameters(),
        *host.local_projector.parameters(),
    ]
    assert all(p.grad is not None for p in online)
    assert all(p.grad is None for p in host.momentum_backbone.parameters())


def test_forward_eval_single_view_returns_embedding_only():
    host = _Host().eval()
    out = densecl(host, {"image": torch.randn(B, 3, 16, 16)}, "validate")
    assert "loss" not in out
    assert out["embedding"].shape == (B, K)


def test_momentum_encoder_updates_in_training_only():
    host = _Host(momentum=0.9).train()
    with torch.no_grad():
        for p in host.backbone.parameters():
            p.add_(1.0)  # make online and target differ
    before = [p.clone() for p in host.momentum_backbone.parameters()]

    densecl(host, _two_views(), "fit")  # self.training is True -> EMA step
    for old, target, online in zip(
        before, host.momentum_backbone.parameters(), host.backbone.parameters()
    ):
        assert torch.allclose(target, 0.9 * old + 0.1 * online, atol=1e-6)

    host.eval()
    frozen = [p.clone() for p in host.momentum_backbone.parameters()]
    densecl(host, _two_views(), "validate")  # self.training is False -> no EMA step
    for old, target in zip(frozen, host.momentum_backbone.parameters()):
        assert torch.equal(old, target)
