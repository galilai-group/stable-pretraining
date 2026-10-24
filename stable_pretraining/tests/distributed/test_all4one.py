"""All4One's global contrast, local feature loss, DDP gradients, and queue."""

import os
import time
from copy import deepcopy
from datetime import timedelta
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from stable_pretraining.tests._all4one_helpers import tiny_all4one
from stable_pretraining.tests.unit.test_all4one_loss import _reference

pytestmark = [pytest.mark.distributed, pytest.mark.regression]


def _loss_inputs(output):
    return {
        name: output[key].chunk(2)
        for name, key in (
            ("projections", "projection"),
            ("momentum_projections", "momentum_projection"),
            ("predictions", "prediction"),
            ("centroids", "centroid"),
            ("shifted_centroids", "shifted_centroid"),
        )
    } | {"neighbours": output["neighbours"][:, 0].chunk(2)}


def _global_reference(outputs):
    local = [_loss_inputs(output) for output in outputs]

    def contrast(anchor_key, candidate_key, temperature):
        losses = []
        for view in range(2):
            anchors = torch.cat([row[anchor_key][view] for row in local])
            candidates = torch.cat([row[candidate_key][1 - view] for row in local])
            logits = F.normalize(anchors, dim=1) @ F.normalize(candidates, dim=1).T
            losses.append(
                F.cross_entropy(logits / temperature, torch.arange(len(anchors)))
            )
        return sum(losses) / 2

    neighbour = contrast("neighbours", "predictions", 0.2)
    centroid = contrast("centroids", "shifted_centroids", 0.1)
    # Averaging local RMS objectives differs from computing one global correlation.
    local_losses = [_reference(row) for row in local]
    features = {
        key: torch.stack([row[key] for row in local_losses]).mean()
        for key in ("feature_loss", "feature_diagonal", "feature_off_diagonal")
    }
    return {
        "loss": 0.5 * neighbour + 0.5 * centroid + features["feature_loss"],
        "neighbour_loss": neighbour,
        "centroid_loss": centroid,
        **features,
    }


def _compare_distributed(rank, rendezvous):
    torch.set_num_threads(1)
    torch.manual_seed(91)
    model = tiny_all4one()
    model.log = Mock()
    references = [deepcopy(model) for _ in range(2)]
    generator = torch.Generator().manual_seed(47)
    batches = [
        {"views": [{"image": torch.randn(3, 6, generator=generator)} for _ in range(2)]}
        for _ in range(2)
    ]
    # Before initializing the group, each reference uses ordinary local BatchNorm.
    outputs = [reference(batch) for reference, batch in zip(references, batches)]
    expected = _global_reference(outputs)
    expected["loss"].backward()
    gradients = {
        name: sum(
            dict(reference.named_parameters())[name].grad for reference in references
        )
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    initial_queue = model.queue.queue.out.clone()
    if rank == 1:
        model.queue.queue.out.zero_()
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        ddp = DistributedDataParallel(model)
        torch.testing.assert_close(model.queue.queue.out, initial_queue)
        actual = ddp(batches[rank])
        for name, target in expected.items():
            averaged = actual[name].detach().clone()
            dist.all_reduce(averaged)
            torch.testing.assert_close(
                averaged / 2, target.detach(), atol=2e-5, rtol=2e-4
            )
        actual["loss"].backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, name
                torch.testing.assert_close(
                    parameter.grad, gradients[name], atol=3e-5, rtol=3e-4, msg=name
                )
            else:
                assert parameter.grad is None, name

        expected_queue = initial_queue.clone()
        pointer = 0

        def append_reference(values):
            nonlocal pointer
            for value in F.normalize(values.detach(), dim=1):
                expected_queue[pointer] = value
                pointer = (pointer + 1) % len(expected_queue)

        append_reference(
            torch.cat([output["momentum_projection"][:3] for output in outputs])
        )
        torch.testing.assert_close(model.queue.queue.out, expected_queue)
        assert model.queue.queue.pointer.item() == pointer == 6
        # A second rank-ordered insertion crosses the queue boundary on both ranks.
        extra = torch.randn(6, 8, generator=generator)
        model.queue.enqueue(extra[rank * 3 : (rank + 1) * 3])
        append_reference(extra)
        torch.testing.assert_close(model.queue.queue.out, expected_queue)
        assert model.queue.queue.pointer.item() == pointer == 1
        buffers = [torch.empty_like(model.queue.queue.out) for _ in range(2)]
        dist.all_gather(buffers, model.queue.queue.out)
        torch.testing.assert_close(buffers[0], buffers[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_all4one_ddp_gradients_and_rank_ordered_queue(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Requires torch.distributed with the CPU Gloo backend")
    context = mp.spawn(
        _compare_distributed,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "All4One distributed workers did not finish within 180 seconds"
                )
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
