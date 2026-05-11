r"""FSDP2 checkpointing.

FSDP2 stores parameters as ``DTensor``\\ s. Sharded save/load uses
``torch.distributed.checkpoint`` (DCP), which is what Lightning's
``ModelParallelStrategy(save_distributed_checkpoint=True)`` uses under
the hood.

This test verifies the round trip directly via DCP:

1. Build a model, FSDP2-wrap it, take one optimizer step (so params and
   optimizer state are non-trivial).
2. Save the model state dict via DCP (one shard per rank → one DCP tree).
3. Build a fresh model, FSDP2-wrap it (different initial params), load
   the DCP tree, and assert every parameter matches the source after
   ``DTensor.full_tensor()`` reduction.

CUDA-only — DCP's distributed save/load path is meaningful with NCCL.
The previous (FSDP1) "DDP checkpoint into FSDP run fails clearly" test
was removed: modern PyTorch handles cross-format loading by transparently
converting on demand, so the assertion no longer holds.
"""

import pytest
import torch
import torch.nn as nn

from stable_pretraining.tests.distributed.conftest import run_distributed


pytestmark = [pytest.mark.distributed, pytest.mark.gpu]


def _build(seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(16, 16)

        def forward(self, x):
            return self.lin(x).relu() + x

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(3)])

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return x

    return M().to(device)


def _save_and_load_round_trip(rank: int, world_size: int) -> None:
    if not torch.cuda.is_available():
        return
    import tempfile
    from pathlib import Path

    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        set_model_state_dict,
    )
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import init_device_mesh

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    # Source model: train one step to make params non-trivial.
    src = _build(seed=0, device=device)
    mesh = init_device_mesh("cuda", (world_size,))
    for blk in src.blocks:
        fully_shard(blk, mesh=mesh)
    fully_shard(src, mesh=mesh)

    optim = torch.optim.SGD(src.parameters(), lr=1e-2)
    x = torch.randn(8, 16, device=device)
    src(x).pow(2).mean().backward()
    optim.step()

    # Capture the source full-tensor params for the equality assertion.
    src_full = [p.full_tensor().detach().clone() for p in src.parameters()]

    # Save via DCP into a shared directory (rank 0 creates it, all ranks
    # contribute their shards).
    tmp = tempfile.mkdtemp(prefix="fsdp2-ckpt-") if rank == 0 else None
    objs = [tmp]
    dist.broadcast_object_list(objs, src=0)
    ckpt_dir = Path(objs[0])

    src_state = get_model_state_dict(src)
    dcp.save(src_state, checkpoint_id=str(ckpt_dir))
    dist.barrier()

    # Destination model: different seed → different initial params.
    dst = _build(seed=999, device=device)
    for blk in dst.blocks:
        fully_shard(blk, mesh=mesh)
    fully_shard(dst, mesh=mesh)

    # Sanity: dst params currently differ from src.
    dst_full_pre = [p.full_tensor().detach().clone() for p in dst.parameters()]
    if rank == 0:
        any_diff = any(
            not torch.allclose(a, b, atol=0, rtol=0)
            for a, b in zip(src_full, dst_full_pre)
        )
        assert any_diff, "test setup bug: dst should differ from src before load"

    dst_state = get_model_state_dict(dst)
    dcp.load(dst_state, checkpoint_id=str(ckpt_dir))
    set_model_state_dict(dst, dst_state)
    dist.barrier()

    # Post-load: dst must match src exactly (after full_tensor reduction).
    dst_full_post = [p.full_tensor().detach().clone() for p in dst.parameters()]
    if rank == 0:
        for i, (a, b) in enumerate(zip(src_full, dst_full_post)):
            assert torch.allclose(a, b, atol=0, rtol=0), (
                f"param {i} did not round-trip through DCP: "
                f"max abs diff = {(a - b).abs().max().item():.3e}"
            )


def test_save_and_load_round_trip():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires >=2 CUDA devices")
    run_distributed(_save_and_load_round_trip, world_size=2, backend="nccl")
