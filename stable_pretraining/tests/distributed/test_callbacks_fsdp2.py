r"""Callback compatibility under FSDP2.

The :class:`stable_pretraining.callbacks.utils.TrainableCallback` system
monkey-patches :meth:`Module.configure_model` (and ``configure_optimizers``)
to register callback-owned modules into ``module.callbacks_modules``. Each
callback runs its own optimizer over its own parameters; sharding those
params under the main FSDP2 unit would put them out of reach of that
optimizer.

The :func:`stable_pretraining.utils.fsdp2.default_parallelize_fn` excludes
callback containers from sharding via **detach-around-root**: it pops the
``callbacks_modules`` and ``callbacks_metrics`` ``ModuleDict``\\ s out of the
module tree before ``fully_shard(model)`` and reattaches them after, so
the root sweep doesn't claim them as DTensors. Skipping in the per-block
loop alone is *not* sufficient — the root sweep would happily turn them
into DTensors regardless.

These tests verify the three load-bearing properties of that exclusion:

1. The containers are reattached (the model tree is shape-preserved).
2. Their parameters remain plain ``nn.Parameter``\\ s, not DTensors.
3. A per-callback optimizer over those parameters runs end-to-end and
   produces finite, non-zero gradients.
"""

import pytest
import torch
import torch.nn as nn

from stable_pretraining.tests.distributed.conftest import run_distributed


pytestmark = pytest.mark.distributed


# ---------------------------------------------------------------------------
# 1. Detach/reattach correctness — the keystone test.
# ---------------------------------------------------------------------------


def _detach_reattach(rank: int, world_size: int) -> None:
    """After parallelize_fn runs, callback containers must satisfy three checks.

    - still be reachable via ``module.callbacks_modules`` (reattached),
    - still be in ``model.named_modules()`` output,
    - their parameters must NOT be DTensors.
    """
    from torch.distributed.tensor import DTensor, init_device_mesh

    import stable_pretraining as spt
    from stable_pretraining.utils.fsdp2 import default_parallelize_fn

    torch.manual_seed(0)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return self.lin(x)

    class TinyVit(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(2)])
            self.embed_dim = 8

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return x

    def fwd(self, batch, stage):
        return {"loss": self.model(batch["image"]).pow(2).mean()}

    module = spt.Module(model=TinyVit(), forward=fwd)
    # Simulate two callbacks registering modules.
    module.callbacks_modules["probe"] = nn.Linear(8, 4)
    module.callbacks_modules["queue"] = nn.Linear(8, 8)

    mesh = init_device_mesh("cpu", (world_size,))
    # Pass ``block_classes={Block}`` because the synthetic ``Block`` isn't
    # in the recognized registry; the kwarg is the documented escape hatch.
    default_parallelize_fn(module, mesh, block_classes={Block})

    # 1) Reattachment.
    assert "callbacks_modules" in module._modules, (
        "callbacks_modules was not reattached after parallelize_fn"
    )
    assert "callbacks_metrics" in module._modules, (
        "callbacks_metrics was not reattached after parallelize_fn"
    )
    assert module.callbacks_modules["probe"].weight.shape == torch.Size([4, 8]), (
        "probe Linear shape changed during detach/reattach"
    )

    # 2) Callback params are NOT DTensors.
    for name, p in module.callbacks_modules.named_parameters():
        assert not isinstance(p, DTensor), (
            f"callback param '{name}' was claimed by root fully_shard "
            f"(became a DTensor) — detach/reattach is broken"
        )

    # 3) Backbone params ARE DTensors (per-block + root claim them).
    backbone_params = list(module.model.blocks[0].parameters())
    assert backbone_params
    assert any(isinstance(p, DTensor) for p in backbone_params), (
        "expected at least one backbone Block param to be a DTensor"
    )


def test_detach_reattach_cpu_world_size_2():
    run_distributed(_detach_reattach, world_size=2, backend="gloo")


# ---------------------------------------------------------------------------
# 2. Per-callback optimizer steps cleanly under FSDP2.
# ---------------------------------------------------------------------------


def _per_callback_optimizer_steps(rank: int, world_size: int) -> None:
    """Per-callback optimizers must run cleanly under FSDP2.

    - construct without raising,
    - receive non-zero gradients on backward,
    - step the param without raising (no DTensor / Tensor mixing).
    """
    from torch.distributed.tensor import init_device_mesh

    import stable_pretraining as spt
    from stable_pretraining.utils.fsdp2 import default_parallelize_fn

    torch.manual_seed(0)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return self.lin(x)

    class TinyVit(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(2)])

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return x

    def fwd(self, batch, stage):
        emb = self.model(batch["image"])
        # Probe head sees a regular tensor (FSDP2 unsharded the backbone).
        probe_logits = self.callbacks_modules["probe"](emb)
        probe_loss = nn.functional.cross_entropy(probe_logits, batch["label"])
        ssl_loss = emb.pow(2).mean()
        return {"loss": ssl_loss + probe_loss, "embedding": emb}

    module = spt.Module(model=TinyVit(), forward=fwd)
    module.callbacks_modules["probe"] = nn.Linear(8, 4)

    mesh = init_device_mesh("cpu", (world_size,))
    default_parallelize_fn(module, mesh, block_classes={Block})

    # Build the per-callback optimizer the way TrainableCallback does.
    probe_params = list(module.callbacks_modules.parameters())
    probe_optim = torch.optim.SGD(probe_params, lr=1e-2)

    batch = {
        "image": torch.randn(4, 8),
        "label": torch.randint(0, 4, (4,)),
    }
    out = module(batch=batch, stage="fit")
    out["loss"].backward()

    # Probe params received gradients.
    for name, p in module.callbacks_modules.named_parameters():
        assert p.grad is not None, f"probe param '{name}' got no gradient"
        assert torch.isfinite(p.grad).all(), f"probe param '{name}' has non-finite grad"
        # And the gradient is non-trivial.
        assert p.grad.abs().sum() > 0, f"probe param '{name}' has zero gradient"

    probe_optim.step()
    probe_optim.zero_grad()


def test_per_callback_optimizer_cpu_world_size_2():
    run_distributed(_per_callback_optimizer_steps, world_size=2, backend="gloo")
