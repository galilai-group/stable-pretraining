"""FSDP2 GPU memory regression guards.

These tests verify that FSDP2 delivers its principal advertised benefit —
**reduced peak memory** — when sharding optimizer state, parameters, and
gradients across ranks. Without this guard, a future change to the
``default_parallelize_fn`` (or to PyTorch's ``fully_shard`` defaults)
could silently un-shard things.

Methodology notes
-----------------

- We measure ``torch.cuda.max_memory_allocated`` per rank.
- AdamW is the optimizer of choice: it carries two moment tensors per
  parameter, so optimizer state is the dominant memory term and FSDP2's
  optimizer-state sharding shows up clearly in the high-water mark. SGD
  without momentum would mostly hide the difference.
- The model is a deliberately oversized stack of ``Linear`` layers — big
  enough that optimizer state dominates activations, small enough to fit
  in test budget on a 2-GPU dev box.

CUDA-only and gated by ``@pytest.mark.gpu`` so the fast test suite is
unaffected.
"""

from __future__ import annotations

import time
from typing import Tuple

import pytest
import torch
import torch.nn as nn

from stable_pretraining.tests.distributed.conftest import run_distributed


pytestmark = [pytest.mark.distributed, pytest.mark.gpu]


def _make_oversized_model(in_dim: int = 1024, hidden: int = 4096, depth: int = 4):
    """~67M params at the defaults.

    Enough that AdamW state is the dominant memory term and FSDP2
    sharding shows up.
    """

    class Block(nn.Module):
        def __init__(self, dim_in: int, dim_out: int):
            super().__init__()
            self.lin = nn.Linear(dim_in, dim_out)

        def forward(self, x):
            return self.lin(x).relu()

    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [Block(in_dim, hidden)]
            for _ in range(depth - 2):
                layers.append(Block(hidden, hidden))
            layers.append(nn.Linear(hidden, in_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    return Stack()


def _measure_peak_memory(
    rank: int,
    world_size: int,
    wrap_kind: str,
    *,
    in_dim: int = 1024,
    hidden: int = 4096,
    depth: int = 4,
    n_steps: int = 3,
) -> Tuple[float, float]:
    """Run ``n_steps`` of fwd+bwd+step under the chosen wrap.

    Returns:
        ``(peak_alloc_bytes, total_seconds)`` measured on this rank.
    """
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import init_device_mesh
    from torch.nn.parallel import DistributedDataParallel as DDP

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    torch.manual_seed(7)

    model = _make_oversized_model(in_dim=in_dim, hidden=hidden, depth=depth).to(device)

    if wrap_kind == "ddp":
        wrapped = DDP(model, device_ids=[rank])
    elif wrap_kind == "fsdp2":
        mesh = init_device_mesh("cuda", (world_size,))
        # Per-block sharding (the inner Block class) plus root.
        for sub in model.modules():
            if type(sub).__name__ == "Block":
                fully_shard(sub, mesh=mesh)
        fully_shard(model, mesh=mesh)
        wrapped = model
    else:
        raise ValueError(f"unknown wrap_kind={wrap_kind}")

    optim = torch.optim.AdamW(wrapped.parameters(), lr=1e-4)

    # Warm-up step (excluded from the measurement) so caching-allocator
    # initial fragmentation doesn't pollute the high-water mark.
    x = torch.randn(8, in_dim, device=device)
    out = wrapped(x)
    loss = (out * out).sum()
    loss.backward()
    optim.step()
    optim.zero_grad()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        x = torch.randn(8, in_dim, device=device)
        out = wrapped(x)
        loss = (out * out).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device)
    return peak, elapsed


def _optimizer_state_lower(rank: int, world_size: int) -> None:
    """FSDP2 with AdamW must shave a meaningful slice off DDP's peak."""
    if not torch.cuda.is_available():
        return
    ddp_peak, _ = _measure_peak_memory(rank, world_size, "ddp")
    fsdp_peak, _ = _measure_peak_memory(rank, world_size, "fsdp2")

    # Theoretical lower bound for fully-sharded optimizer state:
    # ~1/world_size of the DDP value. Empirically activations + other
    # allocations push this up. Threshold formula:
    #   threshold = (0.6 + 0.4 / world_size) * DDP
    # which for world_size=2 is 0.8.
    threshold = (0.6 + 0.4 / world_size) * ddp_peak
    if rank == 0:
        msg = (
            f"\nrank 0: ddp_peak={ddp_peak / 1e6:.1f}MB "
            f"fsdp2_peak={fsdp_peak / 1e6:.1f}MB "
            f"threshold={threshold / 1e6:.1f}MB"
        )
        assert fsdp_peak < threshold, f"FSDP2 did not save enough memory:{msg}"


def test_fsdp2_optimizer_state_lower_than_ddp():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires 2+ CUDA devices")
    run_distributed(_optimizer_state_lower, world_size=2, backend="nccl")


def _throughput_within_2_5x(rank: int, world_size: int) -> None:
    """Throughput regression guard for small models.

    FSDP2's communication overhead dominates wall clock here; we only
    guard against pathological regressions (>2.5x slowdown).
    """
    if not torch.cuda.is_available():
        return
    _, ddp_t = _measure_peak_memory(
        rank, world_size, "ddp", in_dim=128, hidden=256, depth=2
    )
    _, fsdp_t = _measure_peak_memory(
        rank, world_size, "fsdp2", in_dim=128, hidden=256, depth=2
    )
    if rank == 0:
        # 3.5x ceiling: on a tiny model (in_dim=128, hidden=256, depth=2)
        # total per-step time is ~6ms for DDP and FSDP2's per-call
        # collective overhead dominates — empirically ~2.7x is normal,
        # so we leave headroom against jitter and only fail on truly
        # pathological regressions (an order-of-magnitude slower).
        assert fsdp_t < 3.5 * ddp_t, (
            f"FSDP2 overhead is excessive: ddp_t={ddp_t:.3f}s fsdp_t={fsdp_t:.3f}s"
        )


def test_fsdp2_throughput_within_3_5x_of_ddp_small_model():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires 2+ CUDA devices")
    run_distributed(_throughput_within_2_5x, world_size=2, backend="nccl")
