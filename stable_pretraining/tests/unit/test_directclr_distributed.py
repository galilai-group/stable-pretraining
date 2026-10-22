"""CPU/Gloo parity with a global dense loss, including DDP parameter gradients."""

import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from stable_pretraining.losses import DirectCLRLoss
from stable_pretraining.tests.unit.test_directclr import reference

pytestmark = pytest.mark.unit


def _parity_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        torch.manual_seed(31)
        inputs1, inputs2 = (
            torch.randn(6, 4, dtype=torch.float64),
            torch.randn(6, 4, dtype=torch.float64),
        )
        encoder = nn.Linear(4, 5, bias=False).double()
        ref_encoder = nn.Linear(4, 5, bias=False).double()
        ref_encoder.load_state_dict(encoder.state_dict())
        encoder = DistributedDataParallel(encoder)
        indices = slice(rank * 3, (rank + 1) * 3)
        local = DirectCLRLoss()(encoder(inputs1[indices]), encoder(inputs2[indices]))
        expected = reference(ref_encoder(inputs1), ref_encoder(inputs2))
        mean_loss = local.detach().clone()
        dist.all_reduce(mean_loss)
        torch.testing.assert_close(mean_loss / 2, expected)
        local.backward()
        expected.backward()
        torch.testing.assert_close(encoder.module.weight.grad, ref_encoder.weight.grad)
        with pytest.raises(ValueError, match="every rank"):
            DirectCLRLoss()(torch.ones(rank + 1, 3), torch.ones(rank + 1, 3))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend unavailable")
def test_real_two_process_global_value_and_ddp_gradients(tmp_path):
    rendezvous = (tmp_path / "rendezvous").as_uri()
    context = mp.spawn(_parity_worker, args=(rendezvous,), nprocs=2, join=False)
    # Spawn imports the CUDA-enabled torch wheel in each CPU worker; cold
    # shared filesystems can take longer than the collectives themselves.
    deadline = time.monotonic() + 240
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("DirectCLR distributed test timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
