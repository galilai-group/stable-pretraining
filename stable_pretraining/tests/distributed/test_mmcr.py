"""Global MMCR gradients agree with one-process training under CPU DDP."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from stable_pretraining.losses import MMCRLoss


def _worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        torch.manual_seed(23)
        images = torch.randn(6, 3, 5, dtype=torch.float64)
        for lmbda in (0.0, 0.2):
            encoder = nn.Linear(5, 4, bias=False).double()
            distributed = DistributedDataParallel(encoder)
            reference = nn.Linear(5, 4, bias=False).double()
            reference.load_state_dict(encoder.state_dict())
            loss = MMCRLoss(lmbda)(distributed(images.chunk(2)[rank]))
            full_loss = MMCRLoss(lmbda, gather_distributed=False)(reference(images))
            loss.backward()
            full_loss.backward()
            torch.testing.assert_close(loss, full_loss, rtol=1e-9, atol=1e-9)
            torch.testing.assert_close(
                encoder.weight.grad, reference.weight.grad, rtol=1e-8, atol=1e-8
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
@pytest.mark.distributed
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable"
)
def test_global_loss_and_ddp_parameter_gradients(tmp_path):
    try:
        dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")
    except RuntimeError as error:
        if "unsupported gloo device" in str(error):
            pytest.skip("This PyTorch build has no usable Gloo transport device")
        raise
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
