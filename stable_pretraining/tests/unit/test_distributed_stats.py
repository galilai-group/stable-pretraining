"""Distributed statistics agree with torch references, including low precision."""

import pytest
import torch
import torch.distributed as dist

from stable_pretraining.utils.stats import mean_std, mean_var

pytestmark = pytest.mark.unit


def _two_rank_statistics(rank, rendezvous):
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
        for same_shape in (False, True):
            lengths = (2, 2) if same_shape else (2, 3)
            start = 0 if rank == 0 else lengths[0]
            stop = start + lengths[rank]
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                values = (torch.arange(sum(lengths) * 3).reshape(-1, 3) * 32 + 1024).to(
                    dtype
                )
                reference = values.double().clone().requires_grad_()
                expected_var, expected_mean = torch.var_mean(reference, dim=0)
                (expected_mean.sum() + expected_var.sum()).backward()
                local = values[start:stop].clone().requires_grad_()
                mean, variance, count = mean_var(
                    local, same_shape_across_devices=same_shape, unbiased=True
                )
                assert count == sum(lengths)
                torch.testing.assert_close(
                    mean, expected_mean.to(dtype), rtol=0.01, atol=0.01
                )
                torch.testing.assert_close(
                    variance, expected_var.to(dtype), rtol=0.01, atol=0.01
                )
                (mean.sum() + variance.sum()).backward()
                # Both ranks consume the collective output, so backward sums
                # two copies of the gradient of the global objective.
                torch.testing.assert_close(
                    local.grad,
                    (reference.grad[start:stop] * 2).to(dtype),
                    rtol=0.06,
                    atol=0.06,
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_rank_statistics_match_global_values_and_gradients(tmp_path):
    torch.multiprocessing.spawn(
        _two_rank_statistics,
        args=((tmp_path / "two-rank-rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )


@pytest.fixture
def process_group(tmp_path):
    dist.init_process_group(
        "gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("same_shape", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("unbiased,keepdim,dim", [(False, False, 0), (True, True, 1)])
def test_gloo_statistics_match_reference(
    process_group, same_shape, dtype, unbiased, keepdim, dim
):
    x = (torch.arange(24.0).reshape(4, 6) * 8 + 1024).to(dtype).requires_grad_()
    reference_var, reference_mean = torch.var_mean(
        x.double(), dim=dim, keepdim=keepdim, unbiased=unbiased
    )
    mean, var, count = mean_var(
        x,
        dim=dim,
        keepdim=keepdim,
        unbiased=unbiased,
        same_shape_across_devices=same_shape,
    )
    assert count == x.shape[dim]
    assert mean.dtype == var.dtype == dtype
    torch.testing.assert_close(mean, reference_mean.to(dtype), rtol=0.01, atol=0.01)
    torch.testing.assert_close(var, reference_var.to(dtype), rtol=0.01, atol=0.01)
    (mean.float().sum() + var.float().sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("sync", [False, True])
def test_local_standard_deviation_and_epsilon(sync):
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mean, std, count = mean_std(x, unbiased=False, eps=0.25, sync=sync)
    torch.testing.assert_close(mean, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(std, torch.full((2,), 1.25**0.5))
    assert count == 2


@pytest.mark.parametrize("same_shape", [False, True])
def test_unbiased_single_observation_matches_torch_nan_contract(
    process_group, same_shape
):
    mean, variance, count = mean_var(
        torch.tensor([[2.0, 3.0]]), unbiased=True, same_shape_across_devices=same_shape
    )
    torch.testing.assert_close(mean, torch.tensor([2.0, 3.0]))
    assert torch.isnan(variance).all() and count == 1


@pytest.mark.parametrize("rank_deficient", [False, True])
def test_streaming_eigen_distributed_updates_match_local_statistics(
    process_group, rank_deficient
):
    from stable_pretraining.utils.online_topk import StreamingTopKEigen

    data = torch.arange(24.0, dtype=torch.float64).reshape(6, 4)
    if not rank_deficient:
        data = data + torch.eye(6, 4, dtype=torch.float64)
    sync = StreamingTopKEigen(4, 3, dtype=torch.float64, sync_distributed=True)
    local = StreamingTopKEigen(4, 3, dtype=torch.float64, sync_distributed=False)
    for batch in [data, data.flip(0) + 0.5]:
        sync(batch)
        local(batch)
        for key, value in local.state_dict().items():
            torch.testing.assert_close(sync.state_dict()[key], value)
    torch.testing.assert_close(sync.mean, torch.cat([data, data + 0.5]).mean(0))
    torch.testing.assert_close(sync.V.T @ sync.V, torch.eye(3, dtype=torch.float64))
    assert sync.n_samples == 12


def test_streaming_eigen_svd_fallback_preserves_covariance_subspace(monkeypatch):
    from stable_pretraining.utils.online_topk import StreamingTopKEigen

    x = torch.tensor(
        [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    )
    from unittest.mock import Mock

    monkeypatch.setattr(
        torch.linalg, "eigh", Mock(side_effect=RuntimeError("failed to converge"))
    )
    estimator = StreamingTopKEigen(3, 2, sync_distributed=False)
    values, vectors = estimator(x)
    torch.testing.assert_close(values, torch.tensor([2.0, 0.5]))
    torch.testing.assert_close(
        vectors @ vectors.T, torch.diag(torch.tensor([1.0, 1.0, 0.0]))
    )
