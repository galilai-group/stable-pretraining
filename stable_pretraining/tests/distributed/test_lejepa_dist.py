"""Two-process (gloo) tests that LeJEPA's SIGReg statistics are global under DDP.

Regression for ``EppsPulley`` discarding the result of the functional
``all_reduce``, which returns a new tensor: each rank kept its local
statistic and then scaled it by ``world_size``.
"""

import pytest

from stable_pretraining.tests.distributed import _dist_harness as H
from stable_pretraining.tests.distributed import _lejepa_workers as W


@pytest.mark.distributed
class TestEppsPulleyDDP:
    """The Epps-Pulley statistic averages its moments across ranks."""

    def test_epps_pulley_matches_single_process_global_batch(self):
        H.run_distributed(
            W.w_epps_pulley_tests_the_global_batch, world_size=2, backend="gloo"
        )

    def test_sliced_epps_pulley_matches_single_process_global_batch(self):
        H.run_distributed(
            W.w_sliced_epps_pulley_tests_the_global_batch, world_size=2, backend="gloo"
        )
