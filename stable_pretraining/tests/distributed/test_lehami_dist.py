"""Two-process (gloo) tests of where LeHami's SIGReg statistics are computed.

SIGReg in :class:`~stable_pretraining.losses.LeHamiLoss` is computed across the
batch at each frame. On two CPU processes, these pin down that it tests each
rank's local batch when the loss is built before the process group, as under
Lightning and ``spt.Manager``.
"""

import pytest

from stable_pretraining.tests.distributed import _dist_harness as H
from stable_pretraining.tests.distributed import _lehami_workers as W


@pytest.mark.distributed
class TestLeHamiSIGRegDDP:
    """SIGReg tests the local batch on each rank."""

    def test_per_gpu_when_built_before_process_group(self):
        H.run_distributed(
            W.w_sigreg_is_per_gpu_when_built_before_init, world_size=2, backend="gloo"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="EppsPulley (methods/lejepa.py) ignores the result of the functional "
        "all_reduce, so a loss built after init_process_group scales the local "
        "statistic by world_size instead of averaging across ranks.",
    )
    def test_global_when_built_after_process_group(self):
        H.run_distributed(
            W.w_sigreg_is_global_when_built_after_init, world_size=2, backend="gloo"
        )
