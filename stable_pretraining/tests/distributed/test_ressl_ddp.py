"""CPU distributed and Lightning DDP integration tests for ReSSL."""

import lightning as pl
import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.backbone.ressl import ReSSLTeacherStudentWrapper
from stable_pretraining.callbacks.queues import UnsortedQueue
from stable_pretraining.methods.ressl import gather_teacher_keys
from stable_pretraining.tests.distributed import _dist_harness as H
import subprocess
import sys


class _TinyEncoder(nn.Module):
    """BatchNorm-free encoder to keep the CPU tests fast."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(4, 8), nn.ReLU())

    def forward(self, images):
        return self.net(images)


class _TwoViews(Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, index):
        image = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
        image = image / 4 + index / 32
        label = torch.tensor(index % 2)
        return {
            "views": [
                {"image": image, "label": label},
                {"image": image + 0.1, "label": label},
            ]
        }


def _make_model(queue_size=7):
    backbone = ReSSLTeacherStudentWrapper(
        _TinyEncoder(),
        warm_init=True,
        base_ema_coefficient=0.9,
        final_ema_coefficient=0.9,
    )
    projector = ReSSLTeacherStudentWrapper(
        nn.Linear(8, 4),
        warm_init=True,
        base_ema_coefficient=0.9,
        final_ema_coefficient=0.9,
    )
    queue = UnsortedQueue(max_length=queue_size, shape=4)
    with torch.no_grad():
        queue.append(
            torch.nn.functional.normalize(
                torch.randn(queue_size, 4),
                dim=-1,
            )
        )
    return spt.Module(
        backbone=backbone,
        projector=projector,
        queue=queue,
        ressl_loss=spt.losses.ReSSLLoss(
            temperature_student=0.1,
            temperature_teacher=0.04,
        ),
        forward=forward.ressl,
        optim={"optimizer": {"type": "SGD", "lr": 0.01}},
    )


def _assert_queue_synced(queue, world_size):
    """Compare the real queue's storage, pointer, and flags on all ranks."""
    for name, buffer in queue.named_buffers():
        # Compare bytewise representations to support bool and integer buffers.
        values = buffer.detach().contiguous().reshape(-1)
        gathered = [torch.empty_like(values) for _ in range(world_size)]
        dist.all_gather(gathered, values)
        for other in gathered[1:]:
            torch.testing.assert_close(other, gathered[0], msg=f"queue.{name}")


def _worker_gather_queue(rank, world_size):
    """Validate rank ordering, real queue contents, and circular wraparound."""
    queue = UnsortedQueue(max_length=7, shape=4)
    reference = UnsortedQueue(max_length=7, shape=4)

    for step in range(5):
        local = torch.full((2, 4), float(10 * step + rank))
        gathered = gather_teacher_keys(local)
        expected_keys = torch.cat(
            [torch.full((2, 4), float(10 * step + r)) for r in range(world_size)]
        )
        torch.testing.assert_close(gathered, expected_keys)
        queue.append(gathered)
        reference.append(expected_keys)
        for name, expected in reference.named_buffers():
            torch.testing.assert_close(getattr(queue, name), expected)
        _assert_queue_synced(queue, world_size)

    assert bool(queue.filled)
    assert int(queue.pointer) == (5 * 2 * world_size) % queue.max_length

    # An entire global batch larger than capacity must retain only its tail.
    oversized = torch.full((8, 4), float(rank))
    gathered = gather_teacher_keys(oversized)
    queue.append(gathered)
    torch.testing.assert_close(queue.out, gathered[-queue.max_length :])
    assert int(queue.pointer) == 0
    _assert_queue_synced(queue, world_size)


def _worker_real_forward(rank, world_size):
    """Exercise the actual composable ReSSL forward and distributed queue."""
    torch.manual_seed(123)
    model = _make_model(queue_size=7)
    model.train()
    model.log = lambda *args, **kwargs: None  # no Trainer in harness workers
    _assert_queue_synced(model.queue, world_size)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.01,
    )
    initial_student = next(model.backbone.student.parameters()).detach().clone()

    for step in range(4):
        torch.manual_seed(1000 + rank * 100 + step)
        weak = torch.randn(2, 1, 2, 2)
        strong = weak + 0.1
        batch = {"views": [{"image": weak}, {"image": strong}]}

        optimizer.zero_grad()
        output = forward.ressl(model, batch, "train")
        loss = output["loss"]
        assert loss.ndim == 0 and bool(torch.isfinite(loss))
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(grad is not None for grad in grads)
        for grad in grads:
            if grad is not None:
                assert bool(torch.isfinite(grad).all())
                dist.all_reduce(grad)
                grad.div_(world_size)

        optimizer.step()
        model.backbone.update_teacher()
        model.projector.update_teacher()
        _assert_queue_synced(model.queue, world_size)

    assert not torch.equal(
        initial_student,
        next(model.backbone.student.parameters()).detach(),
    ), "Student parameters did not update"
    assert bool(model.queue.filled)

    snapshot = {name: value.clone() for name, value in model.queue.named_buffers()}
    model.eval()
    output = forward.ressl(
        model,
        {"image": torch.randn(2, 1, 2, 2)},
        "val",
    )
    assert "loss" not in output
    for name, value in model.queue.named_buffers():
        torch.testing.assert_close(value, snapshot[name])


class _CheckQueueCallback(pl.Callback):
    """Assert queue consistency inside *each* Lightning DDP rank."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        assert bool(torch.isfinite(outputs["loss"]))
        assert dist.is_initialized()
        assert dist.get_world_size() == 2
        _assert_queue_synced(pl_module.queue, dist.get_world_size())


@pytest.mark.unit
def test_ressl_gather_queue_wraparound_ddp():
    H.run_distributed(
        _worker_gather_queue,
        world_size=2,
        backend="gloo",
        timeout=180,
    )


@pytest.mark.unit
def test_ressl_forward_backward_queue_ddp():
    H.run_distributed(
        _worker_real_forward,
        world_size=2,
        backend="gloo",
        timeout=180,
    )


@pytest.mark.unit
def test_ressl_lightning_ddp(tmp_path):
    """Keep Lightning's distributed processes isolated from pytest."""
    assert not dist.is_initialized()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "stable_pretraining.tests.distributed._ressl_lightning_smoke",
            str(tmp_path),
        ],
        check=True,
        timeout=180,
    )

    assert not dist.is_initialized()
