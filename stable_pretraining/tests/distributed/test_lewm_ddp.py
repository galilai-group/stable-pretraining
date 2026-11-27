"""Two-process distributed training test for LeWM."""

import pytest

from stable_pretraining.tests.distributed import _dist_harness as H


@pytest.mark.unit
def test_lewm_two_process_ddp():
    H.run_distributed(
        H.w_lewm_ddp_training,
        world_size=2,
        backend="gloo",
        timeout=180,
    )
