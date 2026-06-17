r"""Tests for FSDP2 (``strategy="fsdp2"``) support.

Three tiers:

* **Non-distributed** unit checks of the pure helpers (no process group).
* **CPU / gloo** multi-process checks of the real sharding mechanics via the
  :func:`_dist_harness.run_distributed` spawn harness — these run anywhere,
  including CI, and validate wrapping/EMA/alignment without a GPU.
* **GPU / nccl** checks gated on ``>=2`` visible CUDA devices: a real FSDP2
  training step and a DDP-vs-FSDP2 numerical-equivalence proof. Run these on a
  GPU node, e.g.::

      srun --partition=spot --gres=gpu:2 --pty \\
          python -m pytest stable_pretraining/tests/distributed/test_fsdp2.py -m gpu -v
"""

import pytest
import torch
from torch import nn

from stable_pretraining.tests.distributed import _dist_harness as H

_HAS_2_GPU = torch.cuda.is_available() and torch.cuda.device_count() >= 2
_gpu = pytest.mark.skipif(not _HAS_2_GPU, reason="needs >=2 CUDA devices")


# ---------------------------------------------------------------------------
# Non-distributed: pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHelpers:
    """Non-distributed checks of the pure FSDP2 helper functions."""

    def test_is_fsdp_strategy(self):
        from stable_pretraining.utils.fsdp2 import (
            is_fsdp_strategy,
            make_fsdp2_strategy,
        )

        assert is_fsdp_strategy(make_fsdp2_strategy()) is True
        assert is_fsdp_strategy("ddp") is False
        assert is_fsdp_strategy(None) is False

    def test_describe_non_fsdp(self):
        from stable_pretraining.utils.fsdp2 import describe_fsdp_strategy

        assert describe_fsdp_strategy("ddp") == {"fsdp2": False}

    def test_strategy_registered(self):
        import stable_pretraining as spt  # noqa: F401 - triggers deferred init
        from lightning.pytorch.strategies import StrategyRegistry

        # access a heavy attr to force deferred init / registration
        _ = spt.Manager
        assert "fsdp2" in StrategyRegistry.available_strategies()

    def test_assert_aligned_wrapping_plain_modules(self):
        from stable_pretraining.utils.fsdp2 import assert_aligned_wrapping

        a = nn.Linear(4, 4)
        b = nn.Linear(4, 4)
        assert_aligned_wrapping(a, b)  # same structure → no raise

        with pytest.raises(RuntimeError, match="count mismatch"):
            assert_aligned_wrapping(a, nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)))

        with pytest.raises(RuntimeError, match="mismatch"):
            assert_aligned_wrapping(a, nn.Linear(4, 8))

    def test_configure_model_noop_without_mesh(self):
        import stable_pretraining as spt

        m = spt.Module(
            forward=lambda self, batch, stage: batch,
            backbone=H.TinyBackbone(dim=8, depth=2),
        )
        m.configure_model()  # device_mesh is None → no-op, must not raise
        assert all(not H._is_dtensor(p) for p in m.backbone.parameters())

    def test_named_parameters_accepts_remove_duplicate(self):
        import stable_pretraining as spt

        m = spt.Module(
            forward=lambda self, batch, stage: batch, backbone=nn.Linear(4, 4)
        )
        # FSDP2's wrap path calls this; must not raise.
        params = list(m.named_parameters(remove_duplicate=False))
        assert len(params) > 0


# ---------------------------------------------------------------------------
# CPU / gloo: real sharding mechanics
# ---------------------------------------------------------------------------


@pytest.mark.distributed
class TestCPUSharding:
    """Multi-process (CPU/gloo) checks of the real FSDP2 sharding mechanics."""

    def test_shard_subtree_blocks(self):
        H.run_distributed(H.w_shard_subtree_blocks, world_size=2, backend="gloo")

    def test_parallelize_trainable_children_only(self):
        H.run_distributed(
            H.w_parallelize_trainable_children_only, world_size=2, backend="gloo"
        )

    def test_teacher_student_ema(self):
        H.run_distributed(H.w_teacher_student_ema, world_size=2, backend="gloo")

    def test_assert_aligned_rejects_dtensor_mismatch(self):
        H.run_distributed(
            H.w_assert_aligned_rejects_dtensor_mismatch, world_size=2, backend="gloo"
        )

    @pytest.mark.xfail(
        reason="Deferred bug: utils.all_gather/all_reduce discard the functional "
        "collective's return (no-op under distributed). See project memory / "
        "the separate collectives fix. Remove this marker once fixed.",
        strict=True,
    )
    def test_collectives_roundtrip_currently_broken(self):
        H.run_distributed(H.w_collectives_roundtrip, world_size=2, backend="gloo")


# ---------------------------------------------------------------------------
# GPU / nccl: real training + equivalence (>=2 GPUs)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.distributed
@_gpu
class TestGPUFSDP2:
    """Real GPU (nccl, >=2 devices) FSDP2 training + DDP-equivalence checks."""

    def test_fsdp2_training_step(self):
        H.run_distributed(
            H.w_fsdp2_gpu_training_step, world_size=2, backend="nccl", timeout=180
        )

    def test_ddp_vs_fsdp2_equivalence(self):
        H.run_distributed(
            H.w_ddp_vs_fsdp2_equivalence, world_size=2, backend="nccl", timeout=180
        )
