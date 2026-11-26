"""MECLoss across two real gloo processes matches the single-process global loss."""

import pytest
import torch
import torch.distributed as dist

from stable_pretraining.losses import MECLoss

pytestmark = pytest.mark.unit

WORLD_SIZE = 2
LOCAL_BATCH, DIM = 6, 4


def _two_rank_mec_loss(rank, rendezvous):
    from datetime import timedelta

    torch.set_num_threads(1)
    # Every rank builds the same global batch, then keeps only its own shard.
    g = torch.Generator().manual_seed(0)
    global_pred = torch.randn(WORLD_SIZE * LOCAL_BATCH, DIM, generator=g)
    global_target = torch.randn(WORLD_SIZE * LOCAL_BATCH, DIM, generator=g)
    shard = slice(rank * LOCAL_BATCH, (rank + 1) * LOCAL_BATCH)

    configs = [
        dict(symmetric=False, lambda_inv_scale=1.0),
        dict(symmetric=True, lambda_inv_scale=1.0),
        dict(symmetric=False, lambda_inv_scale=8.0),
    ]

    # Single-process references, computed before the process group exists so
    # that ``all_gather`` is the identity.
    references = []
    for cfg in configs:
        pred = global_pred.clone().requires_grad_()
        loss = MECLoss(symmetric=cfg["symmetric"])(
            pred, global_target, cfg["lambda_inv_scale"]
        )
        loss.backward()
        references.append((loss.detach(), pred.grad))

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=45),
    )
    try:
        for cfg, (expected_loss, expected_grad) in zip(configs, references):
            pred = global_pred[shard].clone().requires_grad_()
            target = global_target[shard].clone()  # no grad, like the EMA target
            loss = MECLoss(symmetric=cfg["symmetric"])(
                pred, target, cfg["lambda_inv_scale"]
            )

            # The loss uses the global batch, so every rank gets the same value.
            torch.testing.assert_close(loss.detach(), expected_loss)
            losses = [torch.zeros_like(loss) for _ in range(WORLD_SIZE)]
            dist.all_gather(losses, loss.detach())
            torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)

            # Each rank backpropagates its own copy of the same global loss,
            # and the autograd all_gather sums those copies into each shard,
            # so the shard gradient is WORLD_SIZE x the global one. DDP's
            # gradient averaging over ranks then recovers the global gradient.
            loss.backward()
            torch.testing.assert_close(pred.grad, WORLD_SIZE * expected_grad[shard])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_rank_mec_loss_matches_global_loss_and_gradients(tmp_path):
    torch.multiprocessing.spawn(
        _two_rank_mec_loss,
        args=((tmp_path / "mec-rendezvous").as_uri(),),
        nprocs=WORLD_SIZE,
        join=True,
    )
