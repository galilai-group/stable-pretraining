"""Two-process CPU/Gloo and GPU/NCCL checks for LpJEPA's regularizer."""

from functools import partial

import pytest
import torch
import torch.distributed as dist

from stable_pretraining.tests.distributed import _dist_harness as H

pytestmark = pytest.mark.distributed


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "gloo",
            marks=pytest.mark.skipif(
                not dist.is_gloo_available(), reason="Gloo unavailable"
            ),
        ),
        pytest.param(
            "nccl",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not dist.is_nccl_available() or torch.cuda.device_count() < 2,
                    reason="Requires NCCL and at least 2 CUDA devices",
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize("p", [0.75, 1.0, 2.0])
def test_rdmreg_matches_global_loss_gradients_and_sampling(p, backend):
    H.run_distributed(
        partial(H.w_rdmreg_matches_global, p=p), world_size=2, backend=backend
    )
