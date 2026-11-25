"""Two-process (gloo) tests of where LeHami's SIGReg statistics are computed.

SIGReg in :class:`~stable_pretraining.losses.LeHamiLoss` is computed across the
batch at each frame. These tests pin down, on two CPU processes, which batch
that is: the local one when the loss is built before the process group (as
under Lightning and ``spt.Manager``), the global one when built after.
"""

import pytest

from stable_pretraining.tests.distributed import _dist_harness as H
from stable_pretraining.tests.distributed import _lehami_workers as W


@pytest.mark.distributed
class TestLeHamiSIGRegDDP:
    """SIGReg is per GPU or global depending on when the loss is built."""

    def test_per_gpu_when_built_before_process_group(self):
        H.run_distributed(
            W.w_sigreg_is_per_gpu_when_built_before_init, world_size=2, backend="gloo"
        )

    def test_global_when_built_after_process_group(self):
        H.run_distributed(
            W.w_sigreg_is_global_when_built_after_init, world_size=2, backend="gloo"
        )
