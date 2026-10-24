"""Two-process (gloo, CPU) tests for EquiMod under DDP."""

import pytest

from stable_pretraining.tests.distributed import _dist_harness as H


@pytest.mark.distributed
class TestEquiModDistributed:
    """EquiMod trains under DDP and its gathered equivariance loss is correct."""

    def test_ddp_step_keeps_ranks_in_sync(self):
        H.run_distributed(
            H.w_equimod_ddp_step_keeps_ranks_in_sync, world_size=2, backend="gloo"
        )

    def test_equivariance_loss_cross_rank_negatives_match_global(self):
        H.run_distributed(
            H.w_equimod_equi_loss_crossgpu_matches_global, world_size=2, backend="gloo"
        )
