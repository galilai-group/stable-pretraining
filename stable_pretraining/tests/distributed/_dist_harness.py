"""Reusable harness + worker bodies for FSDP2 distributed tests.

Worker functions live in this importable module (not in the ``test_*.py`` file)
so that ``torch.multiprocessing``'s *spawn* start method can pickle them by
qualified name in the child processes. Each worker takes ``(rank, world_size)``
and runs inside an initialized process group; assertions failing in any rank are
captured and re-raised on the parent by :func:`run_distributed`.

Both CPU (``gloo``) and GPU (``nccl``) backends are supported; the GPU workers
are only invoked by tests gated on ``>=2`` visible CUDA devices.
"""

import os
import socket
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


# ---------------------------------------------------------------------------
# Spawn harness
# ---------------------------------------------------------------------------


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker(rank, world_size, backend, port, fn, error_queue):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        if backend == "nccl":
            torch.cuda.set_device(rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        fn(rank, world_size)
        dist.barrier()
    except Exception:  # noqa: BLE001 - propagate full traceback to parent
        error_queue.put((rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_distributed(fn, world_size: int = 2, backend: str = "gloo", timeout: int = 120):
    """Spawn ``world_size`` procs, run ``fn(rank, world_size)`` in each, re-raise failures.

    Args:
        fn: A picklable module-level callable ``(rank, world_size) -> None``.
        world_size: Number of processes to spawn.
        backend: ``"gloo"`` (CPU) or ``"nccl"`` (GPU).
        timeout: Per-process join timeout in seconds.

    Raises:
        AssertionError: If any rank raised, with the rank-sorted tracebacks.
    """
    ctx = mp.get_context("spawn")
    error_queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(
            target=_worker,
            args=(rank, world_size, backend, port, fn, error_queue),
        )
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout)

    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join()

    if errors:
        errors.sort(key=lambda rt: rt[0])
        report = "\n\n".join(f"--- rank {r} ---\n{tb}" for r, tb in errors)
        raise AssertionError(f"distributed worker(s) failed:\n{report}")


# ---------------------------------------------------------------------------
# Tiny model factories (block-structured so block detection has something to do)
# ---------------------------------------------------------------------------


class _Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


class TinyBackbone(nn.Module):
    """A minimal transformer-ish backbone: a ``ModuleList`` of residual blocks."""

    def __init__(self, dim: int = 8, depth: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(depth)])
        self.embed_dim = dim

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _mesh(device_type: str, world_size: int):
    from torch.distributed.device_mesh import init_device_mesh

    # 1-D data-parallel mesh, matching the dim name our parallelize_fn slices.
    return init_device_mesh(
        device_type, (world_size,), mesh_dim_names=("data_parallel",)
    )


def _is_dtensor(t) -> bool:
    from torch.distributed.tensor import DTensor

    return isinstance(t, DTensor)


# ---------------------------------------------------------------------------
# CPU / gloo workers
# ---------------------------------------------------------------------------


def w_shard_subtree_blocks(rank, world_size):
    """`_shard_subtree` converts every block + the root to DTensor params."""
    from stable_pretraining.utils.fsdp2 import _shard_subtree

    backbone = TinyBackbone(dim=8, depth=3)
    mesh = _mesh("cpu", world_size)
    _shard_subtree(backbone, mesh, None)

    assert all(_is_dtensor(p) for p in backbone.parameters()), (
        "all backbone params should be DTensor after _shard_subtree"
    )
    # forward + backward + a manual SGD step must run on the sharded module
    x = torch.randn(4, 8)
    out = backbone(x)
    out.mean().backward()
    opt = torch.optim.SGD(backbone.parameters(), lr=0.1)
    opt.step()


def w_parallelize_trainable_children_only(rank, world_size):
    """default_parallelize_fn shards backbone/projector but NOT callback containers."""
    import stable_pretraining as spt
    from stable_pretraining.utils.fsdp2 import default_parallelize_fn

    module = spt.Module(
        forward=lambda self, batch, stage: batch,
        backbone=TinyBackbone(dim=8, depth=2),
        projector=nn.Linear(8, 4),
    )
    # A callback-owned module with plain params that must stay untouched.
    module.callbacks_modules["probe"] = nn.Linear(4, 3)

    mesh = _mesh("cpu", world_size)
    default_parallelize_fn(module, mesh)

    # backbone + projector params are sharded...
    assert all(_is_dtensor(p) for p in module.backbone.parameters())
    assert all(_is_dtensor(p) for p in module.projector.parameters())
    # ...but the probe (callback) params are left as plain tensors.
    assert all(
        not _is_dtensor(p) for p in module.callbacks_modules["probe"].parameters()
    ), "callback container params must NOT be sharded"


def w_teacher_student_ema(rank, world_size):
    """TeacherStudentWrapper shards both halves alignedly; EMA runs on DTensors."""
    from stable_pretraining.backbone.utils import TeacherStudentWrapper

    student = TinyBackbone(dim=8, depth=2)
    wrapper = TeacherStudentWrapper(student, base_ema_coefficient=0.5)

    mesh = _mesh("cpu", world_size)
    wrapper.fsdp_setup(mesh, None)  # also asserts aligned wrapping internally

    assert all(_is_dtensor(p) for p in wrapper.student.parameters())
    assert all(_is_dtensor(p) for p in wrapper.teacher.parameters())

    # Make the student differ from the teacher, then EMA-update toward it.
    with torch.no_grad():
        for p in wrapper.student.parameters():
            p.add_(1.0)
    t_before = [p.to_local().clone() for p in wrapper.teacher.parameters()]
    wrapper.train()
    wrapper.update_teacher()
    t_after = [p.to_local() for p in wrapper.teacher.parameters()]

    moved = any(not torch.allclose(a, b) for a, b in zip(t_before, t_after))
    assert moved, "teacher params should move toward student after EMA update"


def w_collectives_roundtrip(rank, world_size):
    """Document the deferred all_gather/all_reduce bug (currently FAILS).

    Used by an ``xfail`` test: when the collectives are fixed this will pass,
    flipping the xfail to xpass so the marker can be removed.
    """
    from stable_pretraining.utils import all_gather, all_reduce

    reduced = all_reduce(torch.tensor([float(rank)]))
    expected_sum = float(sum(range(world_size)))
    assert reduced.item() == expected_sum, (
        f"all_reduce should SUM to {expected_sum}, got {reduced.item()}"
    )

    gathered = torch.cat(all_gather(torch.tensor([float(rank)])), 0)
    assert gathered.tolist() == [float(r) for r in range(world_size)], (
        f"all_gather should concatenate all ranks, got {gathered.tolist()}"
    )


def w_assert_aligned_rejects_dtensor_mismatch(rank, world_size):
    """assert_aligned_wrapping raises when DTensor placements differ."""
    from torch.distributed.tensor import Shard, distribute_tensor

    from stable_pretraining.utils.fsdp2 import assert_aligned_wrapping

    mesh = _mesh("cpu", world_size)

    class _One(nn.Module):
        def __init__(self, placement):
            super().__init__()
            p = nn.Parameter(torch.randn(world_size * 2, world_size * 2))
            # replace with a DTensor sharded along the given dim
            self._p = distribute_tensor(p.detach(), mesh, [placement])

        def parameters(self, recurse=True):
            yield self._p

        def buffers(self, recurse=True):
            return iter(())

    student = _One(Shard(0))
    teacher_ok = _One(Shard(0))
    teacher_bad = _One(Shard(1))

    assert_aligned_wrapping(student, teacher_ok)  # matching → no raise

    raised = False
    try:
        assert_aligned_wrapping(student, teacher_bad)
    except RuntimeError:
        raised = True
    assert raised, "mismatched DTensor placements should raise"


# ---------------------------------------------------------------------------
# GPU / nccl workers (only run when >=2 CUDA devices are available)
# ---------------------------------------------------------------------------


def w_fsdp2_gpu_training_step(rank, world_size):
    """End-to-end FSDP2 sharding + a real optimizer step on GPU."""
    from stable_pretraining.utils.fsdp2 import _shard_subtree

    torch.manual_seed(0)
    backbone = TinyBackbone(dim=64, depth=4).cuda()
    mesh = _mesh("cuda", world_size)
    _shard_subtree(backbone, mesh, None)

    assert all(_is_dtensor(p) for p in backbone.parameters())

    opt = torch.optim.AdamW(backbone.parameters(), lr=1e-2)
    x = torch.randn(16, 64, device="cuda")
    before = backbone.blocks[0].fc1.weight.full_tensor().clone()
    loss = backbone(x).pow(2).mean()
    loss.backward()
    opt.step()
    after = backbone.blocks[0].fc1.weight.full_tensor()

    assert torch.isfinite(loss), "loss must be finite under FSDP2"
    assert not torch.allclose(before, after), "params must update after step"


def w_ddp_vs_fsdp2_equivalence(rank, world_size):
    """One SGD step under FSDP2 (split batch) == single-process full-batch step.

    With a mean-reduced loss, FSDP averages per-rank gradients, so sharding the
    model and splitting the global batch across ranks must reproduce the exact
    update a single process would get on the full batch. Verified to tight
    tolerance on the gathered (full) parameters.
    """
    from stable_pretraining.utils.fsdp2 import _shard_subtree

    dim, total_batch, lr = 32, 8 * world_size, 0.1

    # Deterministic global batch identical on every rank.
    gen = torch.Generator(device="cuda").manual_seed(1234)
    full_x = torch.randn(total_batch, dim, device="cuda", generator=gen)

    # --- reference: unsharded model, full batch, one SGD step ---
    torch.manual_seed(7)
    ref = TinyBackbone(dim=dim, depth=2).cuda()
    ref_sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    opt_ref = torch.optim.SGD(ref.parameters(), lr=lr)
    ref(full_x).pow(2).mean().backward()
    opt_ref.step()
    ref_after = {k: v.detach().clone() for k, v in ref.state_dict().items()}

    # --- FSDP2: same init, each rank takes its slice of the global batch ---
    torch.manual_seed(7)
    model = TinyBackbone(dim=dim, depth=2).cuda()
    model.load_state_dict(ref_sd)  # identical init to the reference
    mesh = _mesh("cuda", world_size)
    _shard_subtree(model, mesh, None)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    per = total_batch // world_size
    local_x = full_x[rank * per : (rank + 1) * per]
    # divide by world_size so the SUM of per-rank mean-losses == full-batch mean
    (model(local_x).pow(2).mean() / world_size).backward()
    opt.step()

    if rank == 0:
        for name, p in model.named_parameters():
            full = p.full_tensor()
            expected = ref_after[name]
            assert torch.allclose(full, expected, atol=1e-4, rtol=1e-4), (
                f"FSDP2 param '{name}' diverged from full-batch reference: "
                f"max|Δ|={(full - expected).abs().max().item():.3e}"
            )
