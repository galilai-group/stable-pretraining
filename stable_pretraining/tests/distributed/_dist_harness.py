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


class _PlainBackbone(nn.Module):
    """No ModuleList/Sequential — block detection finds nothing, only root shards."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.embed_dim = dim

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class _NestedBackbone(nn.Module):
    """A ModuleList of stages, each itself a ModuleList of blocks (nested units)."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.stages = nn.ModuleList(
            [nn.ModuleList([_Block(dim) for _ in range(2)]) for _ in range(2)]
        )
        self.embed_dim = dim

    def forward(self, x):
        for stage in self.stages:
            for blk in stage:
                x = blk(x)
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


def w_non_block_backbone(rank, world_size):
    """Edge case: backbone with no ModuleList/Sequential → only the root shards."""
    from stable_pretraining.utils.fsdp2 import _shard_subtree

    bb = _PlainBackbone(dim=8)
    _shard_subtree(bb, _mesh("cpu", world_size), None)
    assert all(_is_dtensor(p) for p in bb.parameters())
    bb(torch.randn(4, 8)).mean().backward()  # still trainable


def w_nested_modulelist(rank, world_size):
    """Edge case: nested ModuleLists shard deepest-first without double-wrapping."""
    from stable_pretraining.utils.fsdp2 import _shard_subtree

    bb = _NestedBackbone(dim=8)
    _shard_subtree(bb, _mesh("cpu", world_size), None)
    assert all(_is_dtensor(p) for p in bb.parameters())
    bb(torch.randn(4, 8)).mean().backward()


def w_zero_coeff_teacher_student(rank, world_size):
    """Edge case: ema_coefficient==0 → teacher IS student → shard once, no error."""
    from stable_pretraining.backbone.utils import TeacherStudentWrapper

    student = TinyBackbone(dim=8, depth=2)
    wrapper = TeacherStudentWrapper(
        student, base_ema_coefficient=0.0, final_ema_coefficient=0.0
    )
    assert wrapper.teacher is wrapper.student, "teacher should be the student object"
    wrapper.fsdp_setup(_mesh("cpu", world_size), None)  # must not double-shard/raise
    assert all(_is_dtensor(p) for p in wrapper.student.parameters())
    wrapper.forward_student(torch.randn(4, 8))


def w_ema_numerical_correctness(rank, world_size):
    """Edge case: teacher EMA value is exactly ema*teacher+(1-ema)*student on DTensors."""
    from stable_pretraining.backbone.utils import TeacherStudentWrapper

    torch.manual_seed(0)
    ema = 0.9
    wrapper = TeacherStudentWrapper(
        TinyBackbone(dim=8, depth=1), base_ema_coefficient=ema, warm_init=True
    )
    wrapper.fsdp_setup(_mesh("cpu", world_size), None)
    # After warm-init teacher==student; perturb student so the EMA is non-trivial.
    with torch.no_grad():
        for p in wrapper.student.parameters():
            p.add_(1.0)
    t0 = [p.to_local().clone() for p in wrapper.teacher.parameters()]
    s = [p.to_local().clone() for p in wrapper.student.parameters()]
    wrapper.train()
    wrapper.update_teacher()
    for before, sv, after in zip(t0, s, wrapper.teacher.parameters()):
        expected = ema * before + (1 - ema) * sv
        assert torch.allclose(after.to_local(), expected, atol=1e-5), "EMA value wrong"


def w_multiple_trainable_children(rank, world_size):
    """Edge case: backbone+projector+predictor all shard; callbacks stay plain."""
    import stable_pretraining as spt
    from stable_pretraining.utils.fsdp2 import default_parallelize_fn

    module = spt.Module(
        forward=lambda self, b, s: b,
        backbone=TinyBackbone(dim=8, depth=2),
        projector=nn.Linear(8, 8),
        predictor=nn.Linear(8, 8),
    )
    module.callbacks_modules["probe"] = nn.Linear(8, 3)
    default_parallelize_fn(module, _mesh("cpu", world_size))
    for name in ("backbone", "projector", "predictor"):
        assert all(_is_dtensor(p) for p in getattr(module, name).parameters()), name
    assert all(not _is_dtensor(p) for p in module.callbacks_modules.parameters())


def w_custom_parallelize_fn(rank, world_size):
    """Edge case: Module(parallelize_fn=...) is dispatched instead of the default."""
    import stable_pretraining as spt
    from stable_pretraining.utils.fsdp2 import _shard_subtree

    calls = {}

    def custom(module, device_mesh):
        calls["used"] = True
        _shard_subtree(module.backbone, device_mesh["data_parallel"], None)

    module = spt.Module(
        forward=lambda self, b, s: b,
        backbone=TinyBackbone(dim=8, depth=2),
        parallelize_fn=custom,
    )
    module._device_mesh = _mesh("cpu", world_size)
    module.configure_model()
    assert calls.get("used") is True, "custom parallelize_fn was not called"
    assert all(_is_dtensor(p) for p in module.backbone.parameters())


def w_no_shardable_children_warns(rank, world_size):
    """Edge case: a param-less module → warn + no-op, never raise."""
    import stable_pretraining as spt
    from stable_pretraining.utils.fsdp2 import default_parallelize_fn

    module = spt.Module(forward=lambda self, b, s: b, backbone=nn.Identity())
    default_parallelize_fn(module, _mesh("cpu", world_size))  # must not raise


def w_data_parallel_mesh_2d(rank, world_size):
    """Edge case: a 2-D (data_parallel × tensor_parallel=1) mesh shards over DP only."""
    from torch.distributed.device_mesh import init_device_mesh

    from stable_pretraining.utils.fsdp2 import _data_parallel_mesh, _shard_subtree

    mesh2d = init_device_mesh(
        "cpu", (world_size, 1), mesh_dim_names=("data_parallel", "tensor_parallel")
    )
    dp = _data_parallel_mesh(mesh2d)
    assert dp.size() == world_size, "should slice the data_parallel dim"
    bb = TinyBackbone(dim=8, depth=2)
    _shard_subtree(bb, dp, None)
    assert all(_is_dtensor(p) for p in bb.parameters())
    bb(torch.randn(4, 8)).mean().backward()


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


# ---------------------------------------------------------------------------
# Collective-op workers (utils.all_gather / all_reduce correctness)
#
# Regression for the bug where ``utils.all_gather`` / ``utils.all_reduce``
# discarded the functional collective's return value and just echoed the local
# input — a silent no-op under DDP that single-GPU CI never caught.
# ---------------------------------------------------------------------------


def w_all_gather_values(rank, world_size):
    """``all_gather`` returns one tensor per rank, in rank order."""
    from stable_pretraining.utils import all_gather

    out = all_gather(torch.tensor([float(rank)]))
    assert len(out) == world_size, (len(out), world_size)
    cat = torch.cat(out, 0)
    expected = torch.arange(world_size, dtype=torch.float)
    assert torch.allclose(cat, expected), (cat, expected)


def w_all_gather_grad(rank, world_size):
    """``all_gather`` is autograd-aware (gradients flow back to the source).

    Every rank computes the same ``sum(gathered**2)`` loss, so the local input
    appears in all ``world_size`` per-rank losses. The functional collective's
    backward reduce-scatters (sums) those gradients across ranks, giving
    ``d/dx_local == 2 * world_size * x_local``. The old no-op implementation
    didn't gather, so it produced ``2 * x_local`` — this assertion catches that.
    """
    from stable_pretraining.utils import all_gather

    x = torch.tensor([float(rank) + 1.0], requires_grad=True)
    out = torch.cat(all_gather(x), 0)
    out.pow(2).sum().backward()
    assert x.grad is not None
    expected = 2.0 * world_size * x.detach()
    assert torch.allclose(x.grad, expected), (x.grad, expected)


def w_all_reduce_sum(rank, world_size):
    """``all_reduce`` returns the SUM across ranks (default op)."""
    from stable_pretraining.utils import all_reduce

    out = all_reduce(torch.tensor([float(rank)]))
    assert out is not None
    expected = float(sum(range(world_size)))
    assert torch.allclose(out, torch.tensor([expected])), (out, expected)


def w_barlow_matches_single_proc(rank, world_size):
    """With identical data on every rank, Barlow loss == single-process loss.

    Validates the global-batch normalization: each rank divides by
    ``local_batch * world_size`` and the partial cross-correlation matrices are
    summed, recovering exactly the single-process result.
    """
    from stable_pretraining.losses import BarlowTwinsLoss
    from stable_pretraining.losses.utils import off_diagonal

    torch.manual_seed(0)  # identical data on every rank
    z_i = torch.randn(8, 16)
    z_j = torch.randn(8, 16)
    loss = BarlowTwinsLoss(lambd=5e-3)(z_i, z_j)

    def _norm(z):
        return (z - z.mean(0)) / (z.std(0) + 1e-5)

    c = _norm(z_i).T @ _norm(z_j) / z_i.size(0)
    ref = (torch.diagonal(c) - 1).pow(2).sum() + 5e-3 * off_diagonal(c).pow(2).sum()
    assert torch.allclose(loss, ref, atol=1e-5), (loss.item(), ref.item())


def w_contrastive_runs_under_ddp(rank, world_size):
    """NTXEnt (masked) and CLIP (unmasked) contrastive losses run under DDP.

    Regression: ``InfoNCELoss._compute`` builds no targets/mask itself — the
    callers size them to the LOCAL batch. If ``_compute`` all-gathered
    anchors/candidates, ``logits`` would grow by ``world_size`` and mismatch the
    local targets/mask, crashing ``masked_fill`` / ``cross_entropy``. These
    losses are intentionally local-batch objectives; each rank computes its own.
    """
    from stable_pretraining.losses import CLIPLoss, NTXEntLoss

    torch.manual_seed(rank)  # different data per rank, like real DDP
    z_i = torch.randn(4, 8)
    z_j = torch.randn(4, 8)
    ntx = NTXEntLoss(temperature=0.5)(z_i, z_j)
    clip = CLIPLoss(temperature=0.07)(z_i, z_j)
    assert ntx.ndim == 0 and torch.isfinite(ntx), ntx
    assert clip.ndim == 0 and torch.isfinite(clip), clip


def _ref_ntxent_global(z_i, z_j, temperature):
    """Single-process NT-Xent over a (global) batch — no gather, classic form."""
    import torch.nn.functional as F

    a = F.normalize(torch.cat([z_i, z_j], 0), dim=-1)
    logits = a @ a.T / temperature
    two_n = a.size(0)
    n = z_i.size(0)
    targets = torch.cat([torch.arange(n, two_n), torch.arange(n)])
    logits = logits.masked_fill(torch.eye(two_n, dtype=torch.bool), -torch.inf)
    return F.cross_entropy(logits, targets)


def w_ntxent_crossgpu_matches_global(rank, world_size):
    """Mean of per-rank DDP NT-Xent == single-process NT-Xent on the global batch.

    Each rank holds a disjoint, equal-sized slice of a shared global batch and
    contrasts its local anchors against the candidates gathered from all ranks.
    Because the slices are equal-sized, the mean of the per-rank cross-entropies
    equals the single-process cross-entropy over the whole batch — proving the
    gathered negatives + rank-offset targets/mask are wired correctly.
    """
    from stable_pretraining.losses import NTXEntLoss
    from stable_pretraining.utils import all_reduce

    torch.manual_seed(0)  # identical GLOBAL batch on every rank
    n, d = 4, 16
    z_i_global = torch.randn(n * world_size, d)
    z_j_global = torch.randn(n * world_size, d)
    z_i = z_i_global[rank * n : (rank + 1) * n]
    z_j = z_j_global[rank * n : (rank + 1) * n]

    local = NTXEntLoss(temperature=0.5)(z_i, z_j)
    avg = all_reduce(local.detach().clone()) / world_size  # SUM / W == mean

    ref = _ref_ntxent_global(z_i_global, z_j_global, 0.5)
    assert torch.allclose(avg, ref, atol=1e-5), (avg, ref)


def _ref_clip_global(feats_i, feats_j, temperature):
    """Single-process symmetric CLIP loss over a (global) batch — no gather."""
    import torch.nn.functional as F

    fi = F.normalize(feats_i, dim=-1)
    fj = F.normalize(feats_j, dim=-1)
    targets = torch.arange(fi.size(0))
    loss_i = F.cross_entropy(fi @ fj.T / temperature, targets)
    loss_j = F.cross_entropy(fj @ fi.T / temperature, targets)
    return 0.5 * (loss_i + loss_j)


def w_clip_crossgpu_matches_global(rank, world_size):
    """Mean of per-rank DDP CLIP loss == single-process CLIP on the global batch."""
    from stable_pretraining.losses import CLIPLoss
    from stable_pretraining.utils import all_reduce

    torch.manual_seed(0)  # identical GLOBAL batch on every rank
    b, d = 4, 16
    feats_i_global = torch.randn(b * world_size, d)
    feats_j_global = torch.randn(b * world_size, d)
    feats_i = feats_i_global[rank * b : (rank + 1) * b]
    feats_j = feats_j_global[rank * b : (rank + 1) * b]

    local = CLIPLoss(temperature=0.07)(feats_i, feats_j)
    avg = all_reduce(local.detach().clone()) / world_size

    ref = _ref_clip_global(feats_i_global, feats_j_global, 0.07)
    assert torch.allclose(avg, ref, atol=1e-5), (avg, ref)
