"""FSDP2 Integration Helpers.

Wires :func:`torch.distributed.fsdp.fully_shard` (FSDP2) into Lightning by
registering a thin :class:`ModelParallelStrategy` subclass under the
``"fsdp2"`` :class:`StrategyRegistry` name.

After import, users can simply do:

```python
from stable_pretraining.utils.fsdp import StablePretrainingFSDP2

strategy = StablePretrainingFSDP2()
trainer = pl.Trainer(strategy=strategy)
```

and the rest is automatic.

Supports Blocks from timm, HuggingFace, stable-pretraining, and torchvision.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import torch
import torch.nn as nn
from loguru import logger

from lightning.pytorch.strategies import ModelParallelStrategy, StrategyRegistry
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from timm.models.vision_transformer import Block as TimmViTBlock
from torchvision.models.resnet import (
    BasicBlock as TorchvisionResNetBasicBlock,
    Bottleneck as TorchvisionResNetBottleneck,
)
from transformers.models.vit.modeling_vit import ViTLayer as HuggingFaceViTLayer
from stable_pretraining.backbone import TransformerBlock as SPTTransformerBlock


__all__ = [
    "default_parallelize_fn",
    "StablePretrainingFSDP2",
    "UnsupportedModelError",
    "assert_aligned_wrapping",
    "find_callback_containers",
    "RECOGNIZED_BLOCK_CLASSES",
    "is_fsdp_strategy",
    "describe_fsdp_strategy",
]


RECOGNIZED_BLOCK_CLASSES: set[type[nn.Module]] = {
    TimmViTBlock,
    HuggingFaceViTLayer,
    TorchvisionResNetBasicBlock,
    TorchvisionResNetBottleneck,
    SPTTransformerBlock,
}


class UnsupportedModelError(RuntimeError):
    """Raised at strategy setup when no recognized block class is found.

    Pass a custom ``parallelize_fn`` to ``spt.Module`` (or
    ``block_classes={YourBlock}`` directly to :func:`default_parallelize_fn`)
    for models outside :data:`RECOGNIZED_BLOCK_CLASSES`. The error fires at
    setup, not at training time, so misconfiguration surfaces immediately.
    """


def find_callback_containers(model: nn.Module) -> list[nn.Module]:
    """Return the ``callbacks_modules`` / ``callbacks_metrics`` containers on ``model``."""
    return [
        sub
        for name, sub in model.named_modules()
        if name.endswith("callbacks_modules") or name.endswith("callbacks_metrics")
    ]


def _find_container_attachment(model: nn.Module, container: nn.Module):
    """Return ``(parent, attr_name)`` such that ``getattr(parent, attr_name) is container``."""
    for parent in model.modules():
        for attr_name, child in parent._modules.items():
            if child is container:
                return parent, attr_name
    raise RuntimeError(f"Could not locate container {container!r} in model tree.")


def default_parallelize_fn(
    model: nn.Module,
    device_mesh,
    *,
    block_classes: Optional[Iterable[type[nn.Module]]] = None,
    mp_policy: Optional[MixedPrecisionPolicy] = None,
) -> nn.Module:
    r"""Apply FSDP2 sharding: per-block (auto-detected) + root.

    Decides *what* gets sharded: blocks in :data:`RECOGNIZED_BLOCK_CLASSES`
    (or the explicit ``block_classes``) are sharded per-instance, the root
    is sharded last, and callback containers are detached around the root
    sweep so they stay plain ``nn.Module``\ s.

    ``mp_policy`` controls *how* sharded units cast dtypes. For standard
    mixed precision use ``Trainer(precision="bf16-mixed")``; pass a
    :class:`MixedPrecisionPolicy` here only for non-default policies (e.g.
    ``reduce_dtype`` ≠ ``param_dtype``). See ``docs/source/fsdp.rst``.
    """
    # ModelParallelStrategy always passes a 2-D mesh, even at TP=1.
    shard_mesh = device_mesh["data_parallel"] if device_mesh.ndim > 1 else device_mesh

    if block_classes is None:
        present_types = {type(m) for m in model.modules()}
        block_classes = present_types & RECOGNIZED_BLOCK_CLASSES
        if not block_classes:
            raise UnsupportedModelError(
                f"No recognized block class found in {type(model).__name__}. "
                f"Recognized: {sorted(c.__name__ for c in RECOGNIZED_BLOCK_CLASSES)}. "
                f"Pass a custom ``parallelize_fn`` or ``block_classes=`` kwarg."
            )
    block_classes = tuple(block_classes)
    logger.info(
        f"default_parallelize_fn: per-block fully_shard over "
        f"{[c.__name__ for c in block_classes]}"
    )

    shard_kwargs = {"mesh": shard_mesh}
    if mp_policy is not None:
        shard_kwargs["mp_policy"] = mp_policy

    # nested units must be created before the parent
    n_blocks = 0
    for sub in model.modules():
        if sub is not model and isinstance(sub, block_classes):
            fully_shard(sub, **shard_kwargs)
            n_blocks += 1

    # Detach callback containers around the root sweep. Usually a no-op
    # (callbacks register modules *after* configure_model), but populated
    # containers would have their params claimed as DTensors by the root
    # sweep, breaking each callback's standalone optimizer
    detached: list[tuple[nn.Module, str, nn.Module]] = []
    for c in find_callback_containers(model):
        parent, attr_name = _find_container_attachment(model, c)
        del parent._modules[
            attr_name
        ]  # direct mutation: avoids __setattr__ side-effects
        detached.append((parent, attr_name, c))

    try:
        fully_shard(model, **shard_kwargs)
    finally:
        for parent, attr_name, c in reversed(detached):
            parent._modules[attr_name] = c

    logger.info(
        f"default_parallelize_fn: sharded {n_blocks + 1} module(s) "
        f"(per-block + root); detached {len(detached)} callback container(s)"
    )
    return model


def assert_aligned_wrapping(student: nn.Module, teacher: nn.Module) -> None:
    """Assert ``student`` and ``teacher`` have identical FSDP shard layouts.

    Required by :class:`TeacherStudentWrapper.update_teacher`'s in-place EMA
    via ``zip(teacher.parameters(), student.parameters())``. For DTensor
    params the check covers shape + dtype + ``placements`` + ``device_mesh`` —
    same-shape-different-placement is a silent-corruption hazard.
    """
    s_params = list(student.parameters())
    t_params = list(teacher.parameters())
    if len(s_params) != len(t_params):
        raise AssertionError(
            f"FSDP wrapping mismatch: student has {len(s_params)} parameters, "
            f"teacher has {len(t_params)}."
        )
    for i, (sp, tp) in enumerate(zip(s_params, t_params)):
        if sp.shape != tp.shape:
            raise AssertionError(
                f"FSDP wrapping mismatch at parameter {i}: student shape "
                f"{tuple(sp.shape)} vs teacher shape {tuple(tp.shape)}"
            )
        if sp.dtype != tp.dtype:
            raise AssertionError(
                f"FSDP wrapping mismatch at parameter {i}: student dtype "
                f"{sp.dtype} vs teacher dtype {tp.dtype}"
            )
        if isinstance(sp, DTensor) or isinstance(tp, DTensor):
            if not (isinstance(sp, DTensor) and isinstance(tp, DTensor)):
                raise AssertionError(
                    f"FSDP wrapping mismatch at parameter {i}: one side is a "
                    f"DTensor and the other is a plain Tensor."
                )
            if sp.placements != tp.placements:
                raise AssertionError(
                    f"FSDP wrapping mismatch at parameter {i}: student "
                    f"placements {sp.placements} vs teacher placements "
                    f"{tp.placements}."
                )
            if sp.device_mesh != tp.device_mesh:
                raise AssertionError(
                    f"FSDP wrapping mismatch at parameter {i}: student and "
                    f"teacher use different device meshes."
                )

    s_bufs = list(student.buffers())
    t_bufs = list(teacher.buffers())
    if len(s_bufs) != len(t_bufs):
        raise AssertionError(
            f"FSDP wrapping mismatch: student has {len(s_bufs)} buffers, "
            f"teacher has {len(t_bufs)}"
        )
    for i, (sb, tb) in enumerate(zip(s_bufs, t_bufs)):
        if sb.shape != tb.shape:
            raise AssertionError(
                f"FSDP wrapping mismatch at buffer {i}: student shape "
                f"{tuple(sb.shape)} vs teacher shape {tuple(tb.shape)}"
            )
        if sb.dtype != tb.dtype:
            raise AssertionError(
                f"FSDP wrapping mismatch at buffer {i}: student dtype "
                f"{sb.dtype} vs teacher dtype {tb.dtype}"
            )


class StablePretrainingFSDP2(ModelParallelStrategy):
    """:class:`ModelParallelStrategy` with auto-computed ``data_parallel_size``.

    Lightning's ``"auto"`` resolves ``data_parallel_size`` to ``num_nodes``
    (= 1 on single-node multi-GPU), which would fail the
    ``data_parallel_size * tensor_parallel_size == world_size`` check. This
    subclass reads ``LOCAL_WORLD_SIZE`` (set by ``torchrun``) or
    ``torch.cuda.device_count()`` when the user leaves it as ``"auto"``.

    Registered under ``"fsdp2"`` in :class:`StrategyRegistry`, so
    ``Trainer(strategy="fsdp2")`` is valid. Sharding is dispatched by
    :meth:`stable_pretraining.Module.configure_model` to the
    ``parallelize_fn`` callable passed via ``Module(parallelize_fn=...)``
    (defaults to :func:`default_parallelize_fn`).

    Optional ``mp_policy`` is stashed as ``_spt_mp_policy``;
    :func:`default_parallelize_fn` reads it via ``trainer.strategy`` and
    forwards it to ``fully_shard``. Custom ``parallelize_fn`` callables are
    free to ignore it.
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
        if self._data_parallel_size == "auto":
            local_world = os.environ.get("LOCAL_WORLD_SIZE")
            if local_world is not None:
                self._data_parallel_size = int(local_world)
                source = "LOCAL_WORLD_SIZE"
            else:
                self._data_parallel_size = max(1, torch.cuda.device_count())
                source = "torch.cuda.device_count()"
            logger.info(
                f"StablePretrainingFSDP2: data_parallel_size="
                f"{self._data_parallel_size} (inferred from {source})"
            )
        # Lightning resolves tensor_parallel_size="auto" to num_processes,
        # which combined with our DP default trips the DP*TP == world_size
        # check. Default to pure-FSDP so users wanting TP must pass it explicitly
        if self._tensor_parallel_size == "auto":
            self._tensor_parallel_size = 1
        super().setup_environment()


StrategyRegistry.register(
    "fsdp2",
    StablePretrainingFSDP2,
    description=(
        "FSDP2 (fully_shard via ModelParallelStrategy) with auto data_parallel_size. "
        "Sharding is dispatched by stable_pretraining.Module.configure_model."
    ),
)


def is_fsdp_strategy(strategy_or_trainer) -> bool:
    """Return True if the argument is (or wraps) an FSDP2 strategy."""
    strat = getattr(strategy_or_trainer, "strategy", strategy_or_trainer)
    return isinstance(strat, ModelParallelStrategy)


def describe_fsdp_strategy(strategy_or_trainer) -> dict:
    """Return a serializable summary of the FSDP2 strategy's relevant settings."""
    if not is_fsdp_strategy(strategy_or_trainer):
        return {"is_fsdp": False}
    strat = getattr(strategy_or_trainer, "strategy", strategy_or_trainer)
    mp_policy = getattr(strat, "_spt_mp_policy", None)
    return {
        "is_fsdp": True,
        "subclass": type(strat).__name__,
        "data_parallel_size": getattr(strat, "_data_parallel_size", None),
        "tensor_parallel_size": getattr(strat, "_tensor_parallel_size", None),
        "save_distributed_checkpoint": getattr(
            strat, "_save_distributed_checkpoint", None
        ),
        "mp_policy": type(mp_policy).__name__ if mp_policy is not None else None,
    }
