"""Two-process (gloo, CPU) tests for MSF under DDP."""

import pytest

from stable_pretraining.tests.distributed import _dist_harness as H


@pytest.mark.distributed
class TestMSFDistributed:
    """MSF trains under DDP and keeps every rank in sync."""

    def test_ddp_step_keeps_ranks_in_sync(self):
        H.run_distributed(
            H.w_msf_ddp_step_keeps_ranks_in_sync, world_size=2, backend="gloo"
        )
