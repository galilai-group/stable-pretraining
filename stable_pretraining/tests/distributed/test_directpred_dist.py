"""Two-process (gloo) DDP tests for DirectPred.

The unit tests mock the cross-process gather. These spawn 2 CPU processes via
the shared harness and check that under real DDP, DirectPred trains exactly
like a single process on the full batch: the predictor statistics are those of
the projections of all ranks, and the loss and the DDP-averaged gradients match.
"""

from functools import partial

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.backbone import DirectPredictor, TeacherStudentWrapper
from stable_pretraining.losses import BYOLLoss
from stable_pretraining.tests.distributed import _dist_harness as H

N, D, RHO = 8, 6, 0.3  # samples per rank, projection dim, running-average weight


def _projection_batches(world_size, steps=3):
    """Full batches, identical on every rank; rank r's shard has mean 2 r."""
    g = torch.Generator().manual_seed(0)
    offset = 2.0 * torch.arange(world_size).repeat_interleave(N)[:, None]
    return [torch.randn(world_size * N, D, generator=g) + offset for _ in range(steps)]


def w_predictor_matches_full_batch(rank, world_size):
    """Each rank feeds its shard; every rank holds the full-batch estimate."""
    pred = DirectPredictor(D, rho=RHO)
    full = local = None
    for z in _projection_batches(world_size):
        shard = z.chunk(world_size)[rank]
        pred.update(shard)
        batch, batch_local = z.T @ z / z.size(0), shard.T @ shard / N
        full = batch if full is None else RHO * full + (1 - RHO) * batch
        local = batch_local if local is None else RHO * local + (1 - RHO) * batch_local
    torch.testing.assert_close(pred.corr, full)
    torch.testing.assert_close(pred.weight, pred.closed_form_weight(full))
    # A predictor fed only its own shard would be far off.
    assert (local - full).abs().max() > 0.5
    weights = [torch.empty_like(pred.weight) for _ in range(world_size)]
    dist.all_gather(weights, pred.weight)
    for w in weights[1:]:
        torch.testing.assert_close(w, weights[0], rtol=0, atol=0)


def _module():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        backbone = nn.Linear(10, 12)
        projector = nn.Sequential(nn.ReLU(), nn.Linear(12, D))  # no BatchNorm
    return spt.Module(
        backbone=TeacherStudentWrapper(backbone),
        projector=TeacherStudentWrapper(projector),
        predictor=DirectPredictor(D, rho=RHO),
        forward=forward.directpred,
        byol_loss=BYOLLoss(),
    )


def _image_batches(world_size, steps=2):
    """Two views per step, identical on every rank."""
    g = torch.Generator().manual_seed(1)
    return [
        [torch.randn(world_size * N, 10, generator=g) for _ in range(2)]
        for _ in range(steps)
    ]


def _steps(model, batches, call=None):
    """Training steps without optimizer updates; record what DDP must match."""
    call = call or model
    records = []
    for views in batches:
        model.zero_grad()
        loss = call({"views": [{"image": v} for v in views]}, stage="fit")["loss"]
        loss.backward()
        records.append(
            {
                "loss": loss.detach().clone(),
                "grads": {
                    name: p.grad.clone()
                    for name, p in model.named_parameters()
                    if p.grad is not None
                },
                "corr": model.predictor.corr.clone(),
                "weight": model.predictor.weight.clone(),
            }
        )
    return records


def w_training_matches_single_process(reference, rank, world_size):
    """DDP on shards == one process on the full batch, step after step."""
    model = _module()
    ddp = DistributedDataParallel(model)
    shards = [
        [v.chunk(world_size)[rank] for v in views]
        for views in _image_batches(world_size)
    ]
    for got, want in zip(_steps(model, shards, call=ddp), reference):
        dist.all_reduce(got["loss"])
        got["loss"] /= world_size  # mean of the shard losses
        assert got["grads"].keys() == want["grads"].keys()
        for key in ("loss", "corr", "weight"):
            torch.testing.assert_close(got[key], want[key], rtol=1e-5, atol=1e-6)
        for name, grad in want["grads"].items():
            torch.testing.assert_close(
                got["grads"][name], grad, rtol=1e-5, atol=1e-6, msg=name
            )


@pytest.mark.distributed
class TestDirectPredDDP:
    """DirectPred under real DDP matches a single process on the full batch."""

    def test_predictor_statistics_are_gathered_from_all_ranks(self):
        H.run_distributed(w_predictor_matches_full_batch, world_size=2, backend="gloo")

    def test_training_steps_match_a_single_process(self):
        # This process is not distributed: the reference sees the full batch.
        reference = _steps(_module(), _image_batches(2))
        # The second step uses the predictor set by the first.
        assert not torch.allclose(reference[0]["weight"], torch.eye(D))
        H.run_distributed(
            partial(w_training_matches_single_process, reference),
            world_size=2,
            backend="gloo",
        )
