"""Two-process CPU DDP integration test for the public FroSSL method."""

import sys

import pytest
import torch

from stable_pretraining.tests.distributed import _dist_harness as H


@pytest.mark.distributed
@pytest.mark.integration
@pytest.mark.skipif(
    sys.platform != "linux" or not torch.distributed.is_gloo_available(),
    reason="requires Linux and the Gloo distributed backend",
)
def test_frossl_two_process_ddp_step() -> None:
    """Check rank-local FroSSL optimization and synchronized DDP weights."""
    H.run_distributed(H.w_frossl_ddp_step, world_size=2, backend="gloo")
