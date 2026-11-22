"""LFR's DDP-reduced gradients agree with a single global-batch objective."""

from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from stable_pretraining.methods.lfr import _frozen
from stable_pretraining.tests.unit.test_lfr import tiny_lfr

WORLD_SIZE = 2


def _init(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=60),
    )
    torch.set_num_threads(1)


def _compare_global_batch(rank, rendezvous):
    _init(rank, rendezvous)
    torch.manual_seed(19)
    model = tiny_lfr()
    # BatchNorm statistics depend on the local batch, so compare in eval mode.
    model.eval()
    reference = deepcopy(model)
    images = torch.randn(8, 3, 32, 32)

    # A single process sees the whole batch. The process group is already
    # initialized, so evaluate the reference loss with world size one.
    embedding = reference.backbone(images)
    expected = 0
    for projector, predictor in zip(reference.projectors, reference.predictors):
        prediction = torch.nn.functional.normalize(predictor(embedding), dim=1)
        target = torch.nn.functional.normalize(projector(images), dim=1)
        similarity = prediction @ target.T
        diagonal = similarity.diagonal()
        expected = (
            expected
            + (1 - diagonal).square().sum()
            + reference.lfr_loss.lambd
            * (similarity.square().sum() - diagonal.square().sum())
        )
    expected = expected / len(reference.projectors)
    expected.backward()

    ddp = DistributedDataParallel(model, find_unused_parameters=True)
    local = slice(rank * 4, (rank + 1) * 4)
    loss = ddp({"image": images[local]}, stage="fit")["loss"]
    loss.backward()
    mean_loss = loss.detach().clone()
    dist.all_reduce(mean_loss)
    torch.testing.assert_close(mean_loss / WORLD_SIZE, expected.detach())
    for (name, actual), (_, target) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        if name.startswith("projectors"):
            assert actual.grad is None, name
            continue
        assert actual.grad is not None, name
        torch.testing.assert_close(
            actual.grad, target.grad, atol=2e-5, rtol=2e-4, msg=name
        )
    dist.destroy_process_group()


def _alternate_phases(rank, rendezvous):
    _init(rank, rendezvous)
    torch.manual_seed(23)
    model = tiny_lfr().train()
    ddp = DistributedDataParallel(model, find_unused_parameters=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for phase in ("encoder", "encoder", "predictor", "predictor", "encoder"):
        inactive = model.predictors if phase == "encoder" else model.backbone
        with _frozen(inactive):
            optimizer.zero_grad()
            ddp({"image": torch.randn(4, 3, 32, 32)}, stage="fit")["loss"].backward()
            optimizer.step()
        assert all(p.grad is None for p in inactive.parameters())
    # Replicas stay identical across phases.
    for parameter in model.parameters():
        gathered = [torch.empty_like(parameter) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered, parameter.detach())
        torch.testing.assert_close(gathered[0], gathered[1])
    dist.destroy_process_group()


def _validate_uneven_batches(rank, rendezvous):
    _init(rank, rendezvous)
    torch.manual_seed(29)
    model = tiny_lfr().eval()
    # Validation loaders keep the last partial batch, so ranks can disagree.
    images = torch.randn(3 + rank, 3, 32, 32)
    with torch.no_grad():
        loss = model({"image": images}, stage="validate")["loss"]
    assert torch.isfinite(loss)
    dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.regression
def test_lfr_ddp_matches_global_loss_and_gradients(tmp_path):
    mp.spawn(
        _compare_global_batch,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=WORLD_SIZE,
    )


@pytest.mark.distributed
@pytest.mark.regression
def test_lfr_ddp_alternates_frozen_halves_without_desync(tmp_path):
    mp.spawn(
        _alternate_phases,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=WORLD_SIZE,
    )


@pytest.mark.distributed
@pytest.mark.regression
def test_lfr_ddp_validation_tolerates_uneven_batches(tmp_path):
    mp.spawn(
        _validate_uneven_batches,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=WORLD_SIZE,
    )
