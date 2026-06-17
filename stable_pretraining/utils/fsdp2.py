"""FSDP2 (PyTorch ``fully_shard``) support for stable-pretraining.

This module wires PyTorch's per-parameter sharding API
(:func:`torch.distributed.fsdp.fully_shard`, a.k.a. *FSDP2*) into the library
through PyTorch Lightning's :class:`~lightning.pytorch.strategies.ModelParallelStrategy`.

**How a user asks for it**

Exactly like DDP — a single trainer-config switch::

    trainer:
      strategy: fsdp2          # registered below in the StrategyRegistry
      precision: bf16-mixed
      devices: auto

Nothing else is required: importing :mod:`stable_pretraining` registers the
``"fsdp2"`` strategy, and :meth:`stable_pretraining.Module.configure_model`
applies the sharding automatically. Advanced users can override *what* gets
sharded by passing ``Module(parallelize_fn=my_fn)`` (see
:func:`default_parallelize_fn` for the contract).

**Design notes (why this is small)**

- We shard the *trainable child subtrees* of the :class:`~stable_pretraining.Module`
  (``backbone``, ``projector``, ``predictor``, ``prototypes``, ...) rather than
  the ``Module`` root. Sharding per-subtree keeps every main parameter's
  gradient synchronized by FSDP, while leaving the callback/metric containers
  (``OnlineProbe`` & friends, which own *separate* optimizers over plain
  parameters) completely untouched — so there is no need to detach-and-reattach
  them around a root ``fully_shard`` call.
- Blocks are detected generically (elements of any :class:`torch.nn.ModuleList`
  / :class:`torch.nn.Sequential` that own parameters), so timm ViTs, torchvision
  ResNets, and custom transformer stacks all shard per-block with no model
  registry to maintain.
- Teacher/student EMA (DINO, BYOL, MoCo, ...) updates the teacher via a
  positional ``zip`` over ``teacher.parameters()`` / ``student.parameters()``.
  Under FSDP2 both become sharded :class:`~torch.distributed.tensor.DTensor`
  objects, so the two must be wrapped *identically*. :func:`assert_aligned_wrapping`
  enforces that, and :class:`~stable_pretraining.backbone.TeacherStudentWrapper`
  shards both halves through the same code path.
"""

from typing import Optional, Union

from loguru import logger as logging
from torch import nn

# FSDP2 + Lightning's model-parallel strategy both require torch>=2.4. The
# public ``fully_shard`` re-export moved to ``torch.distributed.fsdp`` in torch
# 2.6; on 2.4/2.5 it lives under ``torch.distributed._composable.fsdp``.
try:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
except ImportError:  # pragma: no cover - torch 2.4/2.5
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from lightning.pytorch.strategies import ModelParallelStrategy, StrategyRegistry

__all__ = [
    "StablePretrainingFSDP2",
    "default_parallelize_fn",
    "assert_aligned_wrapping",
    "is_fsdp_strategy",
    "describe_fsdp_strategy",
    "register_fsdp2_strategy",
    "make_fsdp2_strategy",
    "FSDP2_STRATEGY_NAME",
]

FSDP2_STRATEGY_NAME = "fsdp2"

# Children of the SPT ``Module`` that must NOT be sharded: callback/metric
# containers own their own optimizers over plain ``nn.Parameter``s and never
# participate in the main FSDP gradient sync.
_NON_SHARDED_CHILDREN = frozenset({"callbacks_modules", "callbacks_metrics", "metrics"})


# ---------------------------------------------------------------------------
# Strategy detection / introspection
# ---------------------------------------------------------------------------


def is_fsdp_strategy(strategy_or_trainer) -> bool:
    """Return ``True`` when FSDP2 (``ModelParallelStrategy``) is in effect.

    Accepts either a Lightning ``Trainer`` (its ``.strategy`` is inspected) or a
    strategy object directly, so callbacks can guard FSDP-specific behavior with
    a single call.

    Args:
        strategy_or_trainer: A ``pl.Trainer`` or a Lightning ``Strategy``.

    Returns:
        bool: ``True`` if the active strategy is a :class:`ModelParallelStrategy`
        (the base of our ``"fsdp2"`` strategy), else ``False``.
    """
    strategy = getattr(strategy_or_trainer, "strategy", strategy_or_trainer)
    return isinstance(strategy, ModelParallelStrategy)


def describe_fsdp_strategy(strategy_or_trainer) -> dict:
    """Summarize the active FSDP2 strategy for logging.

    Args:
        strategy_or_trainer: A ``pl.Trainer`` or a Lightning ``Strategy``.

    Returns:
        dict: ``{"fsdp2": False}`` when FSDP2 is not active; otherwise the
        resolved data/tensor-parallel sizes, world size, distributed-checkpoint
        flag, and whether a custom mixed-precision policy is attached.
    """
    strategy = getattr(strategy_or_trainer, "strategy", strategy_or_trainer)
    if not isinstance(strategy, ModelParallelStrategy):
        return {"fsdp2": False}
    return {
        "fsdp2": True,
        "data_parallel_size": getattr(strategy, "_data_parallel_size", None),
        "tensor_parallel_size": getattr(strategy, "_tensor_parallel_size", None),
        "world_size": getattr(strategy, "world_size", None),
        "save_distributed_checkpoint": getattr(
            strategy, "_save_distributed_checkpoint", None
        ),
        "mp_policy": getattr(strategy, "_spt_mp_policy", None) is not None,
    }


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------


def _has_params(module: nn.Module) -> bool:
    """True if ``module`` owns at least one parameter (recursively)."""
    return any(True for _ in module.parameters(recurse=True))


def _data_parallel_mesh(device_mesh):
    """Return the data-parallel (sharding) sub-mesh.

    ``ModelParallelStrategy`` always hands :meth:`configure_model` a 2-D mesh
    (``data_parallel`` × ``tensor_parallel``) even when tensor-parallel size is
    1. Passing the full 2-D mesh to ``fully_shard`` would silently reinterpret
    the second axis as an HSDP *replicate* axis, so we slice out the
    ``data_parallel`` dimension explicitly.
    """
    try:
        return device_mesh["data_parallel"]
    except (KeyError, RuntimeError, TypeError):
        # Already 1-D, or an unnamed mesh — use as-is.
        return device_mesh


def _shard_subtree(
    root: nn.Module,
    mesh,
    mp_policy: Optional[MixedPrecisionPolicy] = None,
) -> None:
    """Apply ``fully_shard`` to ``root``: every block first, then ``root`` last.

    "Blocks" are the parameter-owning elements of any :class:`torch.nn.ModuleList`
    or :class:`torch.nn.Sequential` reachable from ``root`` (transformer blocks,
    ResNet stages, ...). FSDP2 requires nested units to be created before their
    parent, so blocks are sharded deepest-first and ``root`` is sharded last.

    Args:
        root: The subtree to shard in place (e.g. a backbone or projector).
        mesh: The 1-D data-parallel device mesh to shard over.
        mp_policy: Optional FSDP2 mixed-precision policy. When ``None``, FSDP
            keeps parameters in their native dtype and relies on the Lightning
            precision plugin (e.g. ``bf16-mixed``) for autocast.
    """
    shard_kwargs = {"mesh": mesh}
    if mp_policy is not None:
        shard_kwargs["mp_policy"] = mp_policy

    name_of = {id(m): n for n, m in root.named_modules()}
    blocks, seen = [], set()
    for module in root.modules():
        if isinstance(module, (nn.ModuleList, nn.Sequential)):
            for block in module:
                if id(block) not in seen and _has_params(block):
                    seen.add(id(block))
                    blocks.append(block)
    # Deepest-first (more dots in the qualified name = deeper) so a nested
    # block is always wrapped before the block that contains it.
    blocks.sort(key=lambda b: name_of.get(id(b), "").count("."), reverse=True)

    for block in blocks:
        fully_shard(block, **shard_kwargs)
    fully_shard(root, **shard_kwargs)


def default_parallelize_fn(
    module: nn.Module,
    device_mesh,
    *,
    mp_policy: Optional[MixedPrecisionPolicy] = None,
) -> nn.Module:
    """Shard the trainable subtrees of an SPT :class:`~stable_pretraining.Module`.

    This is the default ``parallelize_fn`` dispatched from
    :meth:`stable_pretraining.Module.configure_model`. It shards each
    parameter-owning child of ``module`` **except** the callback/metric
    containers (which keep their own optimizers over plain parameters).

    A child exposing an ``fsdp_setup(mesh, mp_policy)`` method (notably
    :class:`~stable_pretraining.backbone.TeacherStudentWrapper`) is delegated to,
    so it can shard its internal sub-modules in an aligned way; every other
    child is sharded with :func:`_shard_subtree`.

    Custom ``parallelize_fn`` callables passed to ``Module(parallelize_fn=...)``
    only need the signature ``(module, device_mesh) -> None`` — the
    ``mp_policy`` keyword is filled in for the default by ``configure_model``.

    Args:
        module: The SPT ``Module`` to parallelize in place.
        device_mesh: The (2-D) device mesh provided by ``ModelParallelStrategy``.
        mp_policy: Optional FSDP2 mixed-precision policy.

    Returns:
        nn.Module: ``module`` (sharded in place), for convenience.
    """
    mesh = _data_parallel_mesh(device_mesh)

    sharded = []
    for name, child in module.named_children():
        if name in _NON_SHARDED_CHILDREN:
            continue
        if not _has_params(child):
            continue
        if hasattr(child, "fsdp_setup") and callable(child.fsdp_setup):
            child.fsdp_setup(mesh, mp_policy)
        else:
            _shard_subtree(child, mesh, mp_policy)
        sharded.append(name)

    if sharded:
        logging.info(f"  fsdp2: sharded trainable subtrees {sharded}")
    else:
        logging.warning(
            "! fsdp2: no parameter-owning child modules found to shard on "
            f"{type(module).__name__}; FSDP2 will have no effect. Pass a custom "
            "Module(parallelize_fn=...) if your trainable parameters live "
            "directly on the module."
        )
    return module


# ---------------------------------------------------------------------------
# Teacher/student alignment
# ---------------------------------------------------------------------------


def assert_aligned_wrapping(student: nn.Module, teacher: nn.Module) -> None:
    """Verify teacher and student are sharded identically for EMA correctness.

    The EMA update zips ``teacher.parameters()`` with ``student.parameters()``
    positionally and applies in-place ops. Under FSDP2 both sides are sharded
    ``DTensor`` objects, so the update is only correct when every paired tensor
    shares shape, dtype, and DTensor placement/mesh.
    This checks that and raises early (at ``configure_model`` time) with a clear
    message rather than letting the teacher silently drift after wrapping.

    Args:
        student: The student sub-module (already sharded).
        teacher: The teacher sub-module (already sharded).

    Raises:
        RuntimeError: If the parameter or buffer streams are misaligned.
    """
    from torch.distributed.tensor import DTensor

    def _check(kind, s_iter, t_iter):
        s_list, t_list = list(s_iter), list(t_iter)
        if len(s_list) != len(t_list):
            raise RuntimeError(
                f"FSDP2 teacher/student {kind} count mismatch "
                f"(student={len(s_list)}, teacher={len(t_list)}); teacher and "
                "student must have identical structure for the EMA update."
            )
        for i, (s, t) in enumerate(zip(s_list, t_list)):
            if s.shape != t.shape or s.dtype != t.dtype:
                raise RuntimeError(
                    f"FSDP2 teacher/student {kind}[{i}] mismatch: "
                    f"student=({tuple(s.shape)}, {s.dtype}) vs "
                    f"teacher=({tuple(t.shape)}, {t.dtype})."
                )
            s_dt, t_dt = isinstance(s, DTensor), isinstance(t, DTensor)
            if s_dt != t_dt:
                raise RuntimeError(
                    f"FSDP2 teacher/student {kind}[{i}] DTensor mismatch: one is "
                    "sharded and the other is not — they must be wrapped the same."
                )
            if s_dt and (
                s.placements != t.placements or s.device_mesh != t.device_mesh
            ):
                raise RuntimeError(
                    f"FSDP2 teacher/student {kind}[{i}] sharding mismatch: "
                    f"student placements={s.placements} mesh={s.device_mesh} vs "
                    f"teacher placements={t.placements} mesh={t.device_mesh}."
                )

    _check("parameter", student.parameters(), teacher.parameters())
    _check("buffer", student.buffers(), teacher.buffers())


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class StablePretrainingFSDP2(ModelParallelStrategy):
    """Pure-FSDP2 flavor of Lightning's :class:`ModelParallelStrategy`.

    Two small but important behaviors on top of the base strategy:

    1. **Sane "auto" sizing.** Lightning's ``"auto"`` resolves
       ``tensor_parallel_size = num_processes`` and ``data_parallel_size =
       num_nodes`` — i.e. *tensor* parallelism across the GPUs and no FSDP
       sharding at all. For ``strategy="fsdp2"`` we want the opposite: shard
       across every rank (``data_parallel_size = world_size``) with no tensor
       parallelism (``tensor_parallel_size = 1``). Explicit sizes are honored.
    2. **Mixed-precision policy passthrough.** An optional
       :class:`~torch.distributed.fsdp.MixedPrecisionPolicy` is stashed so
       :func:`default_parallelize_fn` can forward it to ``fully_shard``.
    """

    def __init__(
        self,
        *args,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._spt_mp_policy = mp_policy

    def setup_environment(self) -> None:
        """Resolve FSDP-first parallel sizes, then build the device mesh.

        Mirrors :meth:`ModelParallelStrategy.setup_environment` but flips the
        ``"auto"`` policy to pure data-parallel sharding (see the class
        docstring). Reimplemented (rather than calling ``super()``) because the
        base resolves ``"auto"`` inline before building the mesh.
        """
        from lightning.pytorch.strategies.model_parallel import _setup_device_mesh

        # Grandparent (ParallelStrategy) setup — matches the base's super() call.
        super(ModelParallelStrategy, self).setup_environment()
        self._setup_distributed()

        if self._tensor_parallel_size == "auto":
            self._tensor_parallel_size = 1
        if self._data_parallel_size == "auto":
            self._data_parallel_size = self.world_size // self._tensor_parallel_size

        self._device_mesh = _setup_device_mesh(
            self._data_parallel_size,
            self._tensor_parallel_size,
            self.world_size,
            self.root_device,
        )
        assert self.lightning_module is not None
        self.lightning_module._device_mesh = self._device_mesh


def register_fsdp2_strategy() -> None:
    """Register ``"fsdp2"`` in Lightning's ``StrategyRegistry`` (idempotent).

    Called from ``stable_pretraining``'s deferred init so that
    ``Trainer(strategy="fsdp2")`` works with no explicit import on the user's
    side.
    """
    if FSDP2_STRATEGY_NAME in StrategyRegistry.available_strategies():
        return
    StrategyRegistry.register(
        FSDP2_STRATEGY_NAME,
        StablePretrainingFSDP2,
        description=(
            "Pure FSDP2 (torch fully_shard) sharding across all ranks, wired "
            "through Lightning's ModelParallelStrategy."
        ),
    )


def make_fsdp2_strategy(
    *,
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    data_parallel_size: Union[str, int] = "auto",
    tensor_parallel_size: Union[str, int] = "auto",
    **kwargs,
) -> StablePretrainingFSDP2:
    """Build a :class:`StablePretrainingFSDP2` strategy object.

    Convenience for users who want to pass a custom mixed-precision policy or
    explicit parallel sizes (``Trainer(strategy=make_fsdp2_strategy(...))``).
    Most users should just use the registered string ``strategy="fsdp2"``.

    Args:
        mp_policy: Optional FSDP2 mixed-precision policy forwarded to
            ``fully_shard``.
        data_parallel_size: Sharding world size (``"auto"`` = all ranks).
        tensor_parallel_size: Tensor-parallel size (``"auto"`` = 1).
        **kwargs: Forwarded to :class:`ModelParallelStrategy`
            (e.g. ``save_distributed_checkpoint``, ``timeout``).

    Returns:
        StablePretrainingFSDP2: A ready-to-use strategy instance.
    """
    return StablePretrainingFSDP2(
        data_parallel_size=data_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        mp_policy=mp_policy,
        **kwargs,
    )
