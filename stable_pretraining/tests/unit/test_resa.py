"""Unit tests for ReSA: the loss and the forward function.

The reference loss below is the official ReSA CIFAR implementation
(https://github.com/winci-ai/resa_cifar, method/base.py and method/resa.py),
reduced to a single process. Copyright (c) Winci, licensed under the Apache
License, Version 2.0.
"""

from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from stable_pretraining.forward import resa
from stable_pretraining.losses import ReSALoss
from stable_pretraining.losses.joint_embedding import InfoNCELoss


@torch.no_grad()
def _official_sinkhorn_knopp(scores, temp=0.05, n_iterations=3):
    q = torch.exp(scores / temp).t()
    num_samples, num_prototypes = q.shape[1], q.shape[0]
    q /= q.sum()
    for _ in range(n_iterations):
        q /= q.sum(dim=1, keepdim=True)
        q /= num_prototypes
        q /= q.sum(dim=0, keepdim=True)
        q /= num_samples
    q *= num_samples
    return q.t()


def _official_loss(h_student, z_student, h_teacher, z_teacher, temp=0.4):
    h = F.normalize(h_student).detach()
    emb = [F.normalize(z) for z in z_student]
    with torch.no_grad():
        h_m = F.normalize(h_teacher)
        emb_m = [F.normalize(z) for z in z_teacher[:2]]
        assign = _official_sinkhorn_knopp(h @ h_m.T)
    losses = [
        -(assign * F.log_softmax(emb[q] @ emb_m[v].T / temp, dim=1)).sum(dim=1).mean()
        for q in range(len(emb))
        for v in range(len(emb_m))
        if v != q
    ]
    return sum(losses) / len(losses)


def _embeddings(n=64, views=2, scale=1.0, seed=0):
    """Backbone features (student and a nearby teacher) and projections."""
    g = torch.Generator().manual_seed(seed)
    h_s = torch.randn(n, 512, generator=g) * scale
    h_t = h_s + 0.3 * scale * torch.randn(n, 512, generator=g)
    z_s = [torch.randn(n, 128, generator=g).requires_grad_() for _ in range(views)]
    z_t = [torch.randn(n, 128, generator=g) for _ in range(2)]
    return h_s, h_t, z_s, z_t


@pytest.mark.unit
class TestReSALoss:
    """ReSALoss against the official implementation, plus edge cases."""

    @pytest.mark.parametrize("views", [2, 4])
    def test_matches_official_loss_and_gradients(self, views):
        h_s, h_t, z_s, z_t = _embeddings(views=views)
        loss = ReSALoss()(h_s, h_t, z_s, z_t)
        grads = torch.autograd.grad(loss, z_s)

        z_ref = [z.detach().clone().requires_grad_() for z in z_s]
        ref = _official_loss(h_s, z_ref, h_t, z_t)
        ref_grads = torch.autograd.grad(ref, z_ref)

        torch.testing.assert_close(loss, ref, rtol=0, atol=1e-5)
        for grad, ref_grad in zip(grads, ref_grads):
            torch.testing.assert_close(grad, ref_grad, rtol=0, atol=1e-6)

    def test_identity_assignment_reduces_to_infonce(self):
        h_s, h_t, z_s, z_t = _embeddings()
        n = h_s.shape[0]
        loss = ReSALoss()(h_s, h_t, z_s, z_t, assignment=torch.eye(n))

        infonce = InfoNCELoss(temperature=0.4)
        targets = torch.arange(n)
        ref = 0.5 * (
            infonce(z_s[0], z_t[1], targets) + infonce(z_s[1], z_t[0], targets)
        )
        torch.testing.assert_close(loss, ref, rtol=0, atol=1e-5)

    def test_assignment_is_a_float32_soft_label_matrix(self):
        h_s, h_t, _, _ = _embeddings()
        assignment = ReSALoss().assignment(h_s, h_t)

        assert assignment.shape == (64, 64)
        assert assignment.dtype == torch.float32
        assert not assignment.requires_grad
        assert (assignment >= 0).all()
        torch.testing.assert_close(
            assignment.sum(dim=1), torch.ones(64), rtol=0, atol=1e-5
        )

    def test_gradients_reach_only_student_projections(self):
        h_s, h_t, z_s, z_t = _embeddings()
        h_s.requires_grad_()
        h_t.requires_grad_()
        for z in z_t:
            z.requires_grad_()

        ReSALoss()(h_s, h_t, z_s, z_t).backward()

        assert all(z.grad is not None and z.grad.abs().sum() > 0 for z in z_s)
        assert h_s.grad is None and h_t.grad is None
        assert all(z.grad is None for z in z_t)

    def test_bf16_autocast_computes_in_float32(self):
        h_s, h_t, z_s, z_t = (
            [x.detach().to(torch.bfloat16) for x in xs]
            if isinstance(xs, list)
            else xs.to(torch.bfloat16)
            for xs in _embeddings()
        )
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = ReSALoss()(h_s, h_t, z_s, z_t)

        ref = _official_loss(
            h_s.float(), [z.float() for z in z_s], h_t.float(), [z.float() for z in z_t]
        )
        assert loss.dtype == torch.float32
        torch.testing.assert_close(loss, ref, rtol=0, atol=1e-5)

    def test_collapsed_batch_stays_finite(self):
        h_s, _, z_s, z_t = _embeddings()
        h_s = h_s[:1].expand_as(h_s).clone()
        z_s = [z[:1].detach().expand_as(z).clone().requires_grad_() for z in z_s]
        z_t = [z[:1].expand_as(z).clone() for z in z_t]

        loss = ReSALoss()(h_s, h_s.clone(), z_s, z_t)
        loss.backward()

        assert torch.isfinite(loss)
        assert all(torch.isfinite(z.grad).all() for z in z_s)

    def test_huge_features_stay_finite(self):
        h_s, h_t, z_s, z_t = _embeddings(scale=1e4)
        loss = ReSALoss()(h_s, h_t, z_s, z_t)
        loss.backward()

        assert torch.isfinite(loss)
        assert all(torch.isfinite(z.grad).all() for z in z_s)

    def test_raises_on_mismatched_embeddings(self):
        h_s, h_t, z_s, z_t = _embeddings()
        with pytest.raises(ValueError, match="same shape"):
            ReSALoss()(h_s, h_t[:10], z_s, z_t)

    def test_raises_without_a_cross_view_pair(self):
        h_s, h_t, z_s, z_t = _embeddings()
        with pytest.raises(ValueError, match="at least one student view"):
            ReSALoss()(h_s, h_t, z_s[:1], z_t[:1])


def _two_rank_resa(rank, rendezvous):
    from datetime import timedelta

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        # Every rank builds the same full batch, keeps its half, and compares
        # with the single-process reference on the whole batch.
        h_s, h_t, z_s, z_t = _embeddings(n=16)
        z_ref = [z.detach().clone().requires_grad_() for z in z_s]
        ref = _official_loss(h_s, z_ref, h_t, z_t)
        ref.backward()
        ref_assignment = _official_sinkhorn_knopp(F.normalize(h_s) @ F.normalize(h_t).T)

        rows = slice(rank * 8, (rank + 1) * 8)
        loss_fn = ReSALoss()
        local_z_s = [z[rows].detach().clone().requires_grad_() for z in z_s]
        assignment = loss_fn.assignment(h_s[rows], h_t[rows])
        loss = loss_fn(
            h_s[rows],
            h_t[rows],
            local_z_s,
            [z[rows] for z in z_t],
            assignment=assignment,
        )

        # Sinkhorn sums over the global batch, so each rank's soft labels are
        # its rows of the full-batch soft labels.
        torch.testing.assert_close(
            assignment, ref_assignment[rows], rtol=1e-4, atol=1e-6
        )
        mean_loss = loss.detach().clone()
        dist.all_reduce(mean_loss)
        torch.testing.assert_close(mean_loss / 2, ref.detach(), rtol=0, atol=1e-5)
        # Each rank averages over half the rows, so its gradients are twice the
        # full-batch ones; DDP's averaging across the two ranks restores them.
        loss.backward()
        for local, full in zip(local_z_s, z_ref):
            torch.testing.assert_close(
                local.grad / 2, full.grad[rows], rtol=1e-4, atol=1e-6
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_rank_loss_matches_full_batch(tmp_path):
    torch.multiprocessing.spawn(
        _two_rank_resa,
        args=((tmp_path / "two-rank-rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )


def _module(training=True):
    """A module with mocked teacher-student wrappers and the real ReSA loss."""
    module = Mock()
    module.training = training
    module.backbone.forward_student = Mock(
        side_effect=lambda x: torch.randn(x.shape[0], 512, requires_grad=True)
    )
    module.backbone.forward_teacher = Mock(
        side_effect=lambda x: torch.randn(x.shape[0], 512)
    )
    module.projector.forward_student = Mock(
        side_effect=lambda x: torch.randn(x.shape[0], 128, requires_grad=True)
    )
    module.projector.forward_teacher = Mock(
        side_effect=lambda x: torch.randn(x.shape[0], 128)
    )
    module.resa_loss = ReSALoss()
    module.log = Mock()
    return module


def _views(n_views, n=8):
    return {
        "views": [
            {"image": torch.randn(n, 3, 32, 32), "label": torch.arange(n)}
            for _ in range(n_views)
        ]
    }


@pytest.mark.unit
class TestReSAForward:
    """The resa forward function with mocked teacher-student wrappers."""

    def test_training_returns_loss_embedding_and_assignment(self):
        module = _module()
        out = resa(module, _views(2), "train")

        assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
        assert out["embedding"].shape == (16, 512)
        assert not out["embedding"].requires_grad
        assert out["resa_assignment"].shape == (8, 8)
        assert out["label"].shape == (16,)
        module.log.assert_called_once()

    def test_teacher_sees_only_the_first_two_views(self):
        module = _module()
        out = resa(module, _views(4), "train")

        assert module.backbone.forward_student.call_count == 4
        assert module.backbone.forward_teacher.call_count == 2
        assert out["embedding"].shape == (16, 512)

    def test_eval_returns_teacher_embedding_without_loss(self):
        module = _module(training=False)
        batch = {"image": torch.randn(8, 3, 32, 32), "label": torch.arange(8)}
        out = resa(module, batch, "val")

        assert "loss" not in out
        assert out["embedding"].shape == (8, 512)
        module.backbone.forward_student.assert_not_called()

    def test_raises_with_a_single_view(self):
        with pytest.raises(ValueError, match="at least 2 views"):
            resa(_module(), _views(1), "train")

    def test_raises_without_resa_loss(self):
        module = _module()
        del module.resa_loss
        with pytest.raises(ValueError, match="resa_loss"):
            resa(module, _views(2), "train")
