"""Two-process DDP integration test for AlignUniform uniformity."""

import math
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import pytest

from stable_pretraining.losses import AlignUniformLoss

pytestmark = pytest.mark.integration


def _align_uniform_ddp_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=Path(init_file).resolve().as_uri(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        generator = torch.Generator().manual_seed(7)
        x_i = torch.randn(world_size * 4, 6, generator=generator)
        x_j = torch.randn(world_size * 4, 6, generator=generator)
        torch.manual_seed(42)
        model = DistributedDataParallel(nn.Linear(6, 8, bias=False))
        reference = nn.Linear(6, 8, bias=False)
        reference.load_state_dict(model.module.state_dict())

        local = slice(rank * 4, (rank + 1) * 4)
        z_i, z_j = model(torch.cat((x_i[local], x_j[local]))).chunk(2)
        loss, alignment, uniformity = AlignUniformLoss().loss_and_terms(z_i, z_j)
        ref_i = F.normalize(reference(x_i), dim=-1)
        ref_j = F.normalize(reference(x_j), dim=-1)
        ref_alignment = (ref_i - ref_j).norm(dim=-1).square().mean()
        ref_uniformity = torch.stack(
            [
                torch.logsumexp(-2.0 * torch.pdist(view).square(), dim=0)
                - math.log(view.shape[0] * (view.shape[0] - 1) // 2)
                for view in (ref_i, ref_j)
            ]
        ).mean()
        local_alignment = (ref_i[local] - ref_j[local]).norm(dim=-1).square().mean()
        torch.testing.assert_close(alignment, local_alignment)
        torch.testing.assert_close(uniformity, ref_uniformity)
        torch.testing.assert_close(loss, local_alignment + ref_uniformity)

        loss.backward()
        (ref_alignment + ref_uniformity).backward()
        assert model.module.weight.grad is not None
        assert torch.isfinite(model.module.weight.grad).all()
        torch.testing.assert_close(
            model.module.weight.grad, reference.weight.grad, rtol=1e-5, atol=1e-6
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()
        with torch.no_grad():
            reference.weight -= 0.1 * reference.weight.grad
        torch.testing.assert_close(model.module.weight, reference.weight)
    finally:
        dist.destroy_process_group()


def test_align_uniform_loss_two_process_ddp(tmp_path):
    init_file = tmp_path / "align_uniform_ddp_init"
    mp.start_processes(
        _align_uniform_ddp_worker,
        args=(2, str(init_file)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
