"""Unit tests for ReSSL loss, EMA wrapper, queue, and distributed keys."""

import copy
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from stable_pretraining.backbone.ressl import ReSSLTeacherStudentWrapper
from stable_pretraining.callbacks.queues import UnsortedQueue
from stable_pretraining.losses import ReSSLLoss
from stable_pretraining.methods import ReSSL
from stable_pretraining.utils.ressl import gather_teacher_keys

pytestmark = pytest.mark.unit


def test_ressl_loss_is_finite_and_differentiable():
    torch.manual_seed(0)
    loss_fn = ReSSLLoss(temperature_student=0.1, temperature_teacher=0.04)
    student = F.normalize(torch.randn(4, 8), dim=-1).detach().requires_grad_()
    teacher = F.normalize(torch.randn(4, 8), dim=-1)
    queue = F.normalize(torch.randn(16, 8), dim=-1)

    loss = loss_fn(student, teacher, queue)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_ressl_loss_matches_manual_relational_cross_entropy():
    student = F.normalize(torch.tensor([[1.0, 2.0], [2.0, 1.0]]), dim=-1)
    teacher = F.normalize(torch.tensor([[2.0, 1.0], [1.0, 2.0]]), dim=-1)
    queue = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), dim=-1)
    loss_fn = ReSSLLoss(temperature_student=0.1, temperature_teacher=0.04)

    targets = (teacher @ queue.T / 0.04).softmax(dim=-1)
    log_probs = (student @ queue.T / 0.1).log_softmax(dim=-1)
    expected = -(targets * log_probs).sum(dim=-1).mean()
    torch.testing.assert_close(loss_fn(student, teacher, queue), expected)


def test_ressl_teacher_starts_equal_to_student():
    wrapper = ReSSLTeacherStudentWrapper(
        nn.Linear(4, 3),
        warm_init=True,
        base_ema_coefficient=0.9,
        final_ema_coefficient=0.9,
    )
    for student, teacher in zip(
        wrapper.student.parameters(), wrapper.teacher.parameters()
    ):
        torch.testing.assert_close(student, teacher)


def test_ressl_parameter_only_ema_does_not_copy_batchnorm_buffers():
    wrapper = ReSSLTeacherStudentWrapper(
        nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)),
        warm_init=True,
        base_ema_coefficient=0.9,
        final_ema_coefficient=0.9,
    )
    with torch.no_grad():
        for param in wrapper.student.parameters():
            param.fill_(1.0)
        for param in wrapper.teacher.parameters():
            param.zero_()
        wrapper.teacher[1].running_mean.fill_(7.0)
        wrapper.student[1].running_mean.fill_(3.0)

    # Assumes update_teacher() uses the configured coefficient when called.
    wrapper.update_teacher()
    for param in wrapper.teacher.parameters():
        torch.testing.assert_close(param, torch.full_like(param, 0.1))
    torch.testing.assert_close(wrapper.teacher[1].running_mean, torch.full((4,), 7.0))


def test_ressl_standalone_forward():
    class TinyEncoder(nn.Module):
        num_features = 8

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(4, 8),
            )

        def forward(self, x):
            return self.net(x)

    model = ReSSL(
        encoder_name=TinyEncoder(),
        queue_size=16,
    )
    model.train()

    weak = torch.randn(4, 1, 2, 2)
    strong = torch.randn(4, 1, 2, 2)

    output = model(weak, strong)

    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)

    output.loss.backward()
    assert any(p.grad is not None for p in model.backbone.student.parameters())


def test_ressl_queue_append_and_state_dict_roundtrip():
    queue = UnsortedQueue(max_length=8, shape=4)
    initial = F.normalize(torch.randn(8, 4), dim=-1)
    queue.append(initial)
    before = queue.get().detach().clone()
    keys = F.normalize(torch.randn(2, 4), dim=-1)
    queue.append(keys)
    assert queue.get().shape == (8, 4)
    assert not torch.equal(queue.get(), before)

    restored = UnsortedQueue(max_length=8, shape=4)
    restored.load_state_dict(copy.deepcopy(queue.state_dict()))
    torch.testing.assert_close(restored.get(), queue.get())
    for key, value in queue.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_gather_teacher_keys_without_distributed_process_group():
    keys = torch.randn(3, 4)
    torch.testing.assert_close(gather_teacher_keys(keys), keys)


def _distributed_queue_worker(rank: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        local = torch.tensor([[float(rank + 1), float(rank + 2)]])
        gathered = gather_teacher_keys(local)
        expected = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
        torch.testing.assert_close(gathered, expected)

        queue = UnsortedQueue(max_length=4, shape=2)
        queue.append(torch.zeros(4, 2))
        queue.append(gathered)
        contents = queue.get().detach().clone()
        other = contents.clone()
        dist.broadcast(other, src=0)
        torch.testing.assert_close(contents, other)

        pointer = queue.state_dict()["pointer"].clone()
        reference_pointer = pointer.clone()
        dist.broadcast(reference_pointer, src=0)
        torch.testing.assert_close(pointer, reference_pointer)
    finally:
        dist.destroy_process_group()


def test_ressl_loss_does_not_backprop_to_queue():
    loss_fn = ReSSLLoss()

    student = F.normalize(torch.randn(4, 8), dim=-1).detach().requires_grad_()

    teacher = F.normalize(torch.randn(4, 8), dim=-1)

    queue = F.normalize(torch.randn(16, 8), dim=-1).detach().requires_grad_()

    loss = loss_fn(student, teacher, queue)
    loss.backward()

    assert student.grad is not None
    assert queue.grad is None


@pytest.mark.skipif(
    not dist.is_available() or os.name == "nt",
    reason="Requires torch.distributed and POSIX file init",
)
def test_ressl_ddp_gather_and_synchronized_queue():
    # Use spawn, not fork, to match distributed test isolation.
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "dist_init")
        mp.spawn(_distributed_queue_worker, args=(init_file,), nprocs=2, join=True)
