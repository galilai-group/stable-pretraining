"""Two-process (gloo) test of the batch statistics JEM gathers across ranks."""

import pytest

from stable_pretraining.tests.distributed import _dist_harness as H


@pytest.mark.distributed
def test_jem_global_regularizer_matches_global():
    H.run_distributed(
        H.w_jem_global_regularizer_matches_global, world_size=2, backend="gloo"
    )
