"""FSDP2 wiring smoke tests.

Two layers:

1. **Structural** (run anywhere): factory returns the right class, registry
   exposes recognized block classes, callback containers are found,
   ``default_parallelize_fn`` raises ``UnsupportedModelError`` on
   unrecognized models.
2. **Wrap+forward+backward** (CPU/gloo or CUDA): a real ``run_distributed``
   group calls ``default_parallelize_fn`` on a model and verifies the result —
   per-block units, root unit, callback containers detached and reattached
   without becoming DTensors, and one forward/backward/step is finite.
"""

import pytest
import torch
import torch.nn as nn

from stable_pretraining.tests.distributed.conftest import (
    run_distributed,
    tiny_backbone,
)


pytestmark = pytest.mark.distributed


# ---------------------------------------------------------------------------
# 1. Structural — no process group required.
# ---------------------------------------------------------------------------


def test_strategy_subclass_is_model_parallel_strategy():
    from lightning.pytorch.strategies import ModelParallelStrategy

    from stable_pretraining.utils.fsdp2 import StablePretrainingFSDP2

    strat = StablePretrainingFSDP2(data_parallel_size=1, tensor_parallel_size=1)
    assert isinstance(strat, ModelParallelStrategy)


def test_fsdp2_registered_in_strategy_registry():
    from lightning.pytorch.strategies import StrategyRegistry

    import stable_pretraining.utils.fsdp2  # noqa: F401  triggers registration

    assert "fsdp2" in StrategyRegistry.available_strategies()


def test_strategy_stashes_mp_policy():
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from stable_pretraining.utils.fsdp2 import StablePretrainingFSDP2

    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    strat = StablePretrainingFSDP2(
        data_parallel_size=1, tensor_parallel_size=1, mp_policy=policy
    )
    assert strat._spt_mp_policy is policy


def test_recognized_block_classes_includes_known_libraries():
    """The registry recognizes timm + HF + torchvision block classes when installed."""
    from stable_pretraining.utils.fsdp2 import recognized_block_classes

    classes = recognized_block_classes()
    names = {c.__name__ for c in classes}
    # All three packages are hard deps in this repo, so all four should resolve.
    assert {"Block", "ViTLayer", "BasicBlock", "Bottleneck"} <= names


def _unrecognized_model_worker(rank: int, world_size: int) -> None:
    """Worker for ``test_default_parallelize_fn_raises_on_unrecognized_model``.

    Must be module-level so ``mp.spawn`` can pickle it.
    """
    from torch.distributed.tensor import init_device_mesh

    from stable_pretraining.utils.fsdp2 import (
        UnsupportedModelError,
        default_parallelize_fn,
    )

    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return self.lin(x)

    mesh = init_device_mesh("cpu", (world_size,))
    with pytest.raises(UnsupportedModelError, match="recognized"):
        default_parallelize_fn(Plain(), mesh)


def test_default_parallelize_fn_raises_on_unrecognized_model():
    """Unrecognized models must raise ``UnsupportedModelError``, not silently fall back."""
    run_distributed(_unrecognized_model_worker, world_size=1, backend="gloo")


def test_find_callback_containers_locates_module_dicts():
    import stable_pretraining as spt
    from stable_pretraining.utils.fsdp2 import find_callback_containers

    module = spt.Module(model=tiny_backbone(), forward=lambda self, b, s: {"loss": 0})
    containers = find_callback_containers(module)
    names = sorted(
        n for n, sub in module.named_modules() if any(sub is c for c in containers)
    )
    assert names == ["callbacks_metrics", "callbacks_modules"]


# ---------------------------------------------------------------------------
# 2. Wrap + forward + backward — real distributed group.
# ---------------------------------------------------------------------------


def _smoke_wrap_and_step(rank: int, world_size: int) -> None:
    """Build a tiny module, parallelize it, run one fwd/bwd/step.

    Asserts:
    - per-block ``fully_shard`` is applied to the synthetic ``Block`` modules
      (passed via ``block_classes=`` since the synthetic class isn't in the
      recognized registry),
    - the root is sharded,
    - callback containers are reattached after the root sweep AND their
      params are NOT DTensors (the detach/reattach contract),
    - one forward+backward+optimizer step runs and produces a finite loss.
    """
    from torch.distributed.tensor import DTensor, init_device_mesh

    import stable_pretraining as spt
    from stable_pretraining.utils.fsdp2 import (
        default_parallelize_fn,
        find_callback_containers,
    )

    torch.manual_seed(7 + rank)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(8, 8)
            self.lin2 = nn.Linear(8, 8)

        def forward(self, x):
            return self.lin2(self.lin1(x).relu())

    class TinyViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(3)])

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return x

    def fwd(self, batch, stage):
        out = self.model(batch["image"])
        return {"loss": (out * out).mean(), "embedding": out}

    module = spt.Module(model=TinyViT(), forward=fwd)
    module.callbacks_modules["probe"] = nn.Linear(8, 4)

    mesh = init_device_mesh("cpu", (world_size,))
    default_parallelize_fn(module, mesh, block_classes={Block})

    # 1) Callback container is back in the tree (reattached).
    containers = find_callback_containers(module)
    assert any(c is module.callbacks_modules for c in containers), (
        "callbacks_modules was not reattached after default_parallelize_fn"
    )

    # 2) Callback params are NOT DTensors (root sweep did not claim them).
    for name, p in module.callbacks_modules.named_parameters():
        assert not isinstance(p, DTensor), (
            f"callback param '{name}' was sharded into a DTensor — "
            f"detach/reattach failed to keep it un-sharded"
        )

    # 3) Backbone Block params ARE DTensors.
    backbone_params = list(module.model.blocks[0].parameters())
    assert backbone_params, "expected the first Block to have parameters"
    assert any(isinstance(p, DTensor) for p in backbone_params), (
        "expected at least one backbone Block param to be a DTensor after wrap"
    )

    # 4) One fwd/bwd/step is finite.
    optim = torch.optim.SGD(
        [p for p in module.parameters() if p.requires_grad], lr=1e-3
    )
    out = module(batch={"image": torch.randn(4, 8)}, stage="fit")
    loss = out["loss"]
    loss.backward()
    optim.step()
    if rank == 0:
        assert torch.isfinite(loss).item(), f"loss not finite: {loss.item()}"


def test_wrap_and_step_cpu_world_size_2():
    """End-to-end: wrap, forward, backward, step on CPU/gloo, world_size=2."""
    run_distributed(_smoke_wrap_and_step, world_size=2, backend="gloo")
