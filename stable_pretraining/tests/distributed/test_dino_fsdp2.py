"""DINO async center update under FSDP2.

:meth:`DINOv1Loss.update_center` (``losses/dino.py``) starts a
:func:`torch.distributed.all_reduce` with ``async_op=True`` for the teacher
center, then waits for completion in :meth:`apply_center_update` on the
*next* training step. Concurrent collectives interleaved with FSDP's own
all-gather / reduce-scatter were a known FSDP1 hazard. FSDP2 uses the
standard collective stream model, so it should be hazard-free, but we
verify by running a few steps end-to-end and asserting:

- training does not deadlock,
- the loss stays finite,
- the center buffer's value is identical across ranks at the end (the
  async all-reduce did complete, and the same averaged tensor landed on
  every rank).

CUDA-only: NCCL is the realistic backend for this test, and DINO's center
update is a CUDA-path concern.
"""

import pytest
import torch
import torch.nn as nn

from stable_pretraining.tests.distributed.conftest import run_distributed


pytestmark = [pytest.mark.distributed, pytest.mark.gpu]


def _dino_no_hang_no_drift(rank: int, world_size: int) -> None:
    if not torch.cuda.is_available():
        return
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import init_device_mesh

    from stable_pretraining.losses.dino import DINOv1Loss

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    torch.manual_seed(7 + rank)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(16, 16)

        def forward(self, x):
            return self.lin(x).relu() + x

    class TinyVit(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(3)])
            self.proj = nn.Linear(16, 32)

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return self.proj(x)

    student = TinyVit().to(device)
    teacher = TinyVit().to(device)

    mesh = init_device_mesh("cuda", (world_size,))
    for blk in student.blocks:
        fully_shard(blk, mesh=mesh)
    fully_shard(student, mesh=mesh)
    for blk in teacher.blocks:
        fully_shard(blk, mesh=mesh)
    fully_shard(teacher, mesh=mesh)

    # DINOv1Loss in this codebase doesn't take ``out_dim`` (it's inferred
    # from the input shape) and its forward signature is
    # ``(student_logits[S,B,K], teacher_probs[T,B,K])`` where the teacher
    # probs come from ``softmax_center_teacher`` (which applies the center
    # internally). Centering uses an async all-reduce kicked off by
    # ``update_center`` and resolved by ``apply_center_update``.
    dino_loss = DINOv1Loss(temperature_student=0.1, center_momentum=0.9).to(device)
    optim = torch.optim.SGD(student.parameters(), lr=1e-3)

    teacher_temp = 0.04
    for step in range(5):
        # 2 views, batch 8, embed 32 — shape [V, B, K] expected by the loss.
        x = torch.randn(2, 8, 16, device=device)
        with torch.no_grad():
            t_out = teacher(x)  # [V, B, K=32]
        s_out = student(x)
        # Teacher probs include centering + softmax; this is also where the
        # async center update is resolved (``update_centers=True`` by default).
        teacher_probs = dino_loss.softmax_center_teacher(t_out, teacher_temp)
        loss = dino_loss(s_out, teacher_probs)
        # Kick off the async all-reduce for next step's centering.
        dino_loss.update_center(t_out)

        loss.backward()
        optim.step()
        optim.zero_grad()
        assert torch.isfinite(loss).item(), f"step {step}: non-finite loss"

    # After the run, drain the final pending center update so the async
    # all-reduce completes and ``self.center`` is in its final state.
    dino_loss.apply_center_update()

    # Center on every rank must match (single averaged value across ranks).
    center = dino_loss.center.detach().clone()
    gathered = [torch.zeros_like(center) for _ in range(world_size)]
    dist.all_gather(gathered, center)
    if rank == 0:
        for r, g in enumerate(gathered):
            assert torch.allclose(g, center, atol=0, rtol=0), (
                f"DINO center diverged across ranks: rank {r} differs from rank 0; "
                f"max abs diff = {(g - center).abs().max().item():.3e}"
            )


def test_dino_no_hang_no_center_drift():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires >=2 CUDA devices")
    run_distributed(_dino_no_hang_no_drift, world_size=2, backend="nccl")
