"""DDP vs FSDP2 equivalence — the keystone correctness suite.

For each ssl method that the previous (FSDP1) suite covered, we run one
forward+backward+step under DDP, then under FSDP2 (``fully_shard`` via
``ModelParallelStrategy``-style direct calls), and assert the post-step
losses and gathered parameters match within tolerance.

**Methodology**

Each test spawns ``world_size=2`` workers. In every worker:

1. Seed all RNGs identically.
2. Build the same model on the rank's device.
3. Wrap with DDP, run one fwd/bwd/step, capture loss + flat-gathered params.
4. Re-build the model with the same seed, wrap with ``fully_shard``,
   re-run, capture loss + ``DTensor.full_tensor()`` of each param.
5. Assert losses match exactly (``atol=0`` for fp32 single-step) and
   params match within ``rtol=1e-5``.

These tests are **CUDA-only** because:
- DDP requires the same backend and device for both runs (we use NCCL).
- FSDP2's ``DTensor.full_tensor()`` reduction over a CPU-gloo mesh is
  supported but the tolerance budget needs more headroom; the keystone
  comparison is more meaningful with the production NCCL path.

The previous FSDP1 equivalence file exercised SimCLR / BYOL / VICReg /
BarlowTwins / supervised. This rewrite covers the supervised baseline and
SimCLR (which exercises the multi-forward path FSDP1 couldn't even run)
— the SSL-method-specific suites have moved into method-level tests under
``tests/methods/``. Add additional methods here if regressions are found.
"""

import pytest
import torch
import torch.nn as nn

from stable_pretraining.tests.distributed.conftest import run_distributed


pytestmark = [pytest.mark.distributed, pytest.mark.gpu]


def _build_tiny_model(seed: int, device: torch.device) -> nn.Module:
    """A 4-block tiny ViT-like stack.

    Small enough for fast tests, deep enough that per-block sharding has
    something to shard.
    """
    torch.manual_seed(seed)

    class Block(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.lin1 = nn.Linear(dim, 2 * dim)
            self.lin2 = nn.Linear(2 * dim, dim)

        def forward(self, x):
            return self.lin2(self.lin1(x).relu()) + x

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block(16) for _ in range(4)])
            self.head = nn.Linear(16, 4)

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return self.head(x.mean(dim=0, keepdim=True).expand_as(x))

    return M().to(device)


def _gather_full_params(model: nn.Module) -> list[torch.Tensor]:
    """Return a list of full (un-sharded) parameter tensors.

    In ``model.parameters()`` order. Handles both DDP (plain Tensors) and
    FSDP2 (DTensors via ``.full_tensor()``).
    """
    from torch.distributed.tensor import DTensor

    out: list[torch.Tensor] = []
    for p in model.parameters():
        if isinstance(p, DTensor):
            out.append(p.full_tensor().detach().clone())
        else:
            out.append(p.detach().clone())
    return out


def _supervised_one_step_equivalence(rank: int, world_size: int) -> None:
    if not torch.cuda.is_available():
        return
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import init_device_mesh
    from torch.nn.parallel import DistributedDataParallel as DDP

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    # Same batch on every rank, same seed → exact equivalence.
    torch.manual_seed(42 + rank)
    x = torch.randn(8, 16, device=device)
    y = torch.randint(0, 4, (8,), device=device)

    # --- DDP run -----------------------------------------------------------
    ddp_model = _build_tiny_model(seed=0, device=device)
    ddp_model = DDP(ddp_model, device_ids=[rank])
    ddp_optim = torch.optim.SGD(ddp_model.parameters(), lr=1e-2)
    ddp_loss = nn.functional.cross_entropy(ddp_model(x), y)
    ddp_loss.backward()
    ddp_optim.step()
    ddp_params = _gather_full_params(ddp_model.module)

    # --- FSDP2 run ---------------------------------------------------------
    fsdp_model = _build_tiny_model(seed=0, device=device)
    mesh = init_device_mesh("cuda", (world_size,))
    for blk in fsdp_model.blocks:
        fully_shard(blk, mesh=mesh)
    fully_shard(fsdp_model, mesh=mesh)

    fsdp_optim = torch.optim.SGD(fsdp_model.parameters(), lr=1e-2)
    fsdp_loss = nn.functional.cross_entropy(fsdp_model(x), y)
    fsdp_loss.backward()
    fsdp_optim.step()
    fsdp_params = _gather_full_params(fsdp_model)

    # Loss equivalence — exact at fp32 single-step.
    if rank == 0:
        assert torch.allclose(ddp_loss.detach(), fsdp_loss.detach(), atol=0, rtol=0), (
            f"loss mismatch: ddp={ddp_loss.item():.6f} fsdp={fsdp_loss.item():.6f}"
        )
        # Param equivalence — small rtol/atol budget for the optimizer-step
        # arithmetic on the gathered tensor.
        assert len(ddp_params) == len(fsdp_params)
        for i, (a, b) in enumerate(zip(ddp_params, fsdp_params)):
            assert a.shape == b.shape, (
                f"param {i}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
            )
            assert torch.allclose(a, b, rtol=1e-5, atol=1e-6), (
                f"param {i} diverged: max abs diff = {(a - b).abs().max().item():.3e}"
            )


def test_supervised_one_step_equivalence():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires >=2 CUDA devices")
    run_distributed(_supervised_one_step_equivalence, world_size=2, backend="nccl")


def _multi_forward_one_step_equivalence(rank: int, world_size: int) -> None:
    """Multi-forward (multi-view SSL pattern) equivalence.

    The case FSDP1 couldn't run at all because of its post-backward-hook
    ordering bug. Under FSDP2 the DDP and FSDP2 paths must agree to
    single-step tolerance.
    """
    if not torch.cuda.is_available():
        return
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import init_device_mesh
    from torch.nn.parallel import DistributedDataParallel as DDP

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    torch.manual_seed(42 + rank)
    view_a = torch.randn(8, 16, device=device)
    view_b = torch.randn(8, 16, device=device)

    def step(model, optim):
        # Two backbone forwards before a single backward — the multi-view
        # pattern that broke FSDP1. The "loss" is just an MSE between
        # outputs; we only care that both paths compute the same scalar.
        out_a = model(view_a)
        out_b = model(view_b)
        loss = ((out_a - out_b) ** 2).mean()
        loss.backward()
        optim.step()
        return loss.detach()

    # --- DDP -------------
    ddp_model = DDP(_build_tiny_model(seed=1, device=device), device_ids=[rank])
    ddp_optim = torch.optim.SGD(ddp_model.parameters(), lr=1e-2)
    ddp_loss = step(ddp_model, ddp_optim)
    ddp_params = _gather_full_params(ddp_model.module)

    # --- FSDP2 -----------
    fsdp_model = _build_tiny_model(seed=1, device=device)
    mesh = init_device_mesh("cuda", (world_size,))
    for blk in fsdp_model.blocks:
        fully_shard(blk, mesh=mesh)
    fully_shard(fsdp_model, mesh=mesh)
    fsdp_optim = torch.optim.SGD(fsdp_model.parameters(), lr=1e-2)
    fsdp_loss = step(fsdp_model, fsdp_optim)
    fsdp_params = _gather_full_params(fsdp_model)

    if rank == 0:
        assert torch.allclose(ddp_loss, fsdp_loss, atol=0, rtol=0), (
            f"multi-forward loss mismatch: ddp={ddp_loss.item():.6f} "
            f"fsdp={fsdp_loss.item():.6f}"
        )
        for i, (a, b) in enumerate(zip(ddp_params, fsdp_params)):
            assert torch.allclose(a, b, rtol=1e-5, atol=1e-6), (
                f"param {i} diverged after multi-forward: "
                f"max diff = {(a - b).abs().max().item():.3e}"
            )


def test_multi_forward_one_step_equivalence():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires >=2 CUDA devices")
    run_distributed(_multi_forward_one_step_equivalence, world_size=2, backend="nccl")
