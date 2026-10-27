"""Unit tests for λ-JEPA (LambdaJEPA) and SACReg."""

from typing import List

import pytest
import torch
import torch.distributed as dist

from stable_pretraining.methods.lambda_jepa import LambdaJEPA, SACRegLoss

pytestmark = pytest.mark.unit

TINY_VIT = "vit_tiny_patch16_224"
B = 4
V = 3
C, H, W = 3, 224, 224

SMALL_KWARGS = dict(
    encoder_name=TINY_VIT,
    h_slice_dim=128,
    z_queue_length=0,
    h_queue_length=0,
    ema_target=False,
)


def _views() -> List[torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return [torch.randn(B, C, H, W, generator=g) for _ in range(V)]


# --- SACReg -----------------------------------------------------------------


def test_sacreg_matches_manual_computation():
    torch.manual_seed(0)
    x = torch.randn(64, 16) * 2.0 + 0.5
    value = SACRegLoss(slice_dim=8, eps=1e-4)(x, torch.Generator().manual_seed(0))

    q, _ = torch.linalg.qr(
        torch.randn(16, 8, generator=torch.Generator().manual_seed(0))
    )
    p = x @ q
    mu = p.mean(0)
    cov = (p - mu).T @ (p - mu) / (p.size(0) - 1) + 1e-4 * torch.eye(8)
    expected = 0.5 * (mu.square().sum() + cov.trace() - torch.logdet(cov) - 8) / 8

    assert torch.allclose(value, expected, atol=1e-5)


def test_sacreg_statistic_includes_queued_inputs():
    torch.manual_seed(0)
    loss = SACRegLoss(slice_dim=8, queue_length=16)
    loss(torch.randn(16, 16))
    queued = loss.queue.get().clone()

    x = torch.randn(16, 16)
    value = loss(x, torch.Generator().manual_seed(0))
    expected = SACRegLoss(slice_dim=8)(
        torch.cat([x, queued]), torch.Generator().manual_seed(0)
    )
    assert torch.allclose(value, expected, atol=1e-5)


def test_sacreg_queue_keeps_previous_inputs_detached():
    loss = SACRegLoss(slice_dim=8, queue_length=32)
    for _ in range(3):
        loss(torch.randn(16, 8, requires_grad=True))
    queued = loss.queue.get()
    assert queued.shape == (32, 8)
    assert not queued.requires_grad


# --- forward / backward -----------------------------------------------------


def test_lambdajepa_forward_backward():
    model = LambdaJEPA(**SMALL_KWARGS)
    model.train()
    output = model(global_views=_views())

    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert output.embedding.shape == (V * B, model.embed_dim)

    output.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert any(g.abs().sum() > 0 for g in grads)


def test_lambdajepa_loss_is_weighted_sum_of_terms():
    model = LambdaJEPA(**SMALL_KWARGS)
    model.train()
    output = model(global_views=_views())

    expected = (
        model.w_inv * output.inv_loss
        + model.beta_z * output.sacreg_z_loss
        + model.beta_h * output.sacreg_h_loss
    )
    assert torch.allclose(output.loss, expected, atol=1e-6)


def test_lambdajepa_invariance_is_mean_over_view_pairs():
    z = torch.randn(4, 5, 6)
    pairs = [(z[u] - z[w]).square().mean() for u in range(4) for w in range(u + 1, 4)]
    assert torch.allclose(LambdaJEPA._invariance(z), torch.stack(pairs).mean())


def test_lambdajepa_invariance_with_target_keeps_the_pairwise_scale():
    z = torch.randn(3, 5, 6)
    target = torch.randn(5, 6)
    expected = (z - target).square().mean() * (2 * 3 / (3 - 1))
    assert torch.allclose(LambdaJEPA._invariance(z, target), expected)


def test_lambdajepa_slice_generator_is_seeded_by_the_step():
    model = LambdaJEPA(**SMALL_KWARGS)
    first = torch.randn(4, generator=model._slice_generator(torch.device("cpu")))
    second = torch.randn(4, generator=model._slice_generator(torch.device("cpu")))
    assert model.slice_step == 2
    assert torch.equal(
        first, torch.randn(4, generator=torch.Generator().manual_seed(0))
    )
    assert not torch.equal(first, second)


def test_lambdajepa_projector_and_backbone_slices_are_independent():
    model = LambdaJEPA(**SMALL_KWARGS)
    generator = model._slice_generator(torch.device("cpu"))
    assert not torch.equal(
        torch.randn(8, 8, generator=generator), torch.randn(8, 8, generator=generator)
    )


def test_lambdajepa_ema_target_forward_backward():
    model = LambdaJEPA(**{**SMALL_KWARGS, "ema_target": True})
    model.train()
    output = model(global_views=_views())

    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.backbone.student.patch_embed.proj.weight.grad is not None
    assert model.backbone.teacher.patch_embed.proj.weight.grad is None


@pytest.mark.parametrize("n_views", [0, 1])
def test_lambdajepa_training_requires_two_views(n_views):
    model = LambdaJEPA(**SMALL_KWARGS)
    model.train()
    with pytest.raises(AssertionError, match="at least two global_views"):
        model(global_views=_views()[:n_views] or None)


@pytest.mark.parametrize("ema_target", [False, True])
def test_lambdajepa_eval_mode_returns_embedding_without_loss(ema_target):
    model = LambdaJEPA(**{**SMALL_KWARGS, "ema_target": ema_target})
    model.eval()
    images = _views()[0]

    with torch.no_grad():
        output = model(images=images)
        backbone = model.backbone.student if ema_target else model.backbone
        expected = backbone(images)

    assert output.loss == 0
    assert output.embedding.shape == (B, model.embed_dim)
    torch.testing.assert_close(output.embedding, expected)


def test_lambdajepa_eval_mode_requires_images():
    model = LambdaJEPA(**SMALL_KWARGS)
    model.eval()
    with pytest.raises(AssertionError, match="images must be provided"):
        model()


# --- distributed ------------------------------------------------------------


def _two_rank_sacreg(rank, rendezvous):
    from datetime import timedelta

    torch.set_num_threads(1)
    torch.manual_seed(0)
    full = torch.randn(32, 16)
    reference = full.clone().requires_grad_()
    expected = SACRegLoss(slice_dim=8)(reference, torch.Generator().manual_seed(0))
    expected.backward()

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        local = full[rank * 16 : (rank + 1) * 16].clone().requires_grad_()
        value = SACRegLoss(slice_dim=8)(local, torch.Generator().manual_seed(0))
        torch.testing.assert_close(value, expected)
        value.backward()
        # Both ranks consume the gathered batch, so backward sums two copies
        # of the gradient of the global objective.
        torch.testing.assert_close(
            local.grad, 2 * reference.grad[rank * 16 : (rank + 1) * 16]
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_rank_sacreg_matches_full_batch_value_and_gradient(tmp_path):
    torch.multiprocessing.spawn(
        _two_rank_sacreg,
        args=((tmp_path / "two-rank-rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )


def _two_rank_lambdajepa(rank, rendezvous):
    from copy import deepcopy
    from datetime import timedelta

    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    torch.manual_seed(0)
    # A linear projector and no drop path, so both runs see the same statistics.
    model = LambdaJEPA(
        **{**SMALL_KWARGS, "projector": torch.nn.Linear(192, 16)},
        z_slice_dim=8,
        drop_path_rate=0.0,
    )
    model.train()
    reference = deepcopy(model)
    views = _views()
    expected = reference(global_views=views).loss
    expected.backward()

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        ddp = DistributedDataParallel(model)
        local = [v[rank * 2 : (rank + 1) * 2] for v in views]
        loss = ddp(global_views=local).loss
        loss.backward()
        mean_loss = loss.detach().clone()
        dist.all_reduce(mean_loss)
        torch.testing.assert_close(mean_loss / 2, expected.detach())
        for (name, actual), (_, target) in zip(
            model.named_parameters(), reference.named_parameters()
        ):
            torch.testing.assert_close(
                actual.grad, target.grad, atol=1e-2, rtol=1e-3, msg=name
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_rank_lambdajepa_matches_full_batch_loss_and_gradients(tmp_path):
    torch.multiprocessing.spawn(
        _two_rank_lambdajepa,
        args=((tmp_path / "two-rank-rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )
