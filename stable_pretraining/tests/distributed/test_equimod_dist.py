"""Two-process (gloo, CPU) tests for EquiMod under DDP."""

import pytest

from stable_pretraining.tests.distributed import _dist_harness as H


@pytest.mark.distributed
class TestEquiModDistributed:
    """EquiMod trains under DDP with the documented loss and batch-norm semantics."""

    def test_ddp_step_keeps_ranks_in_sync(self):
        H.run_distributed(
            H.w_equimod_ddp_step_keeps_ranks_in_sync, world_size=2, backend="gloo"
        )

    def test_both_losses_use_negatives_gathered_from_all_ranks(self):
        H.run_distributed(
            H.w_equimod_losses_crossgpu_match_global, world_size=2, backend="gloo"
        )

    def test_param_norm_uses_local_batch_statistics(self):
        H.run_distributed(
            H.w_equimod_param_norm_uses_local_batch_statistics,
            world_size=2,
            backend="gloo",
        )
