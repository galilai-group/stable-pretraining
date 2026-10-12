"""FLIP's DDP-reduced gradients agree with a single global-batch objective."""

from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from stable_pretraining.tests.unit.test_flip import tiny_flip


def _compare_global_batch(rank, rendezvous):
    world_size = 2
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model = tiny_flip(mask_ratio=0)
    reference = deepcopy(model)
    images = torch.randn(4, 3, 32, 32)
    tokens = torch.randint(0, 32, (4, 6))

    image = reference.encode_image(images)
    text = reference.encode_text(tokens)
    logits = image @ text.T * reference.logit_scale.exp()
    labels = torch.arange(4)
    expected = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    expected.backward()

    ddp = DistributedDataParallel(model)
    local = slice(rank * 2, (rank + 1) * 2)
    loss = ddp({"image": images[local], "input_ids": tokens[local]})["loss"]
    loss.backward()
    mean_loss = loss.detach().clone()
    dist.all_reduce(mean_loss)
    torch.testing.assert_close(mean_loss / world_size, expected.detach())
    for (name, actual), (_, target) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert actual.grad is not None, name
        torch.testing.assert_close(
            actual.grad, target.grad, atol=2e-5, rtol=2e-4, msg=name
        )
    dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.regression
def test_flip_ddp_matches_global_loss_and_gradients(tmp_path):
    mp.spawn(
        _compare_global_batch, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2
    )
