"""Real CPU DDP loss and parameter-gradient checks for ADNCE."""

import time
from pathlib import Path

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from stable_pretraining.tests.distributed._adnce_harness import worker

pytestmark = [
    pytest.mark.integration,
    pytest.mark.distributed,
    pytest.mark.skipif(
        not dist.is_available() or not dist.is_gloo_available(),
        reason="PyTorch Gloo distributed backend is unavailable",
    ),
]


@pytest.mark.parametrize("scenario", ["equivalence", "input_mismatch"])
def test_adnce_two_processes(tmp_path: Path, scenario: str) -> None:
    """Match global gradients or reject incompatible inputs on all ranks."""
    context = mp.spawn(
        worker,
        args=(2, (tmp_path / "rendezvous").as_uri(), scenario),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=max(0.0, deadline - time.monotonic())):
            if time.monotonic() >= deadline:
                pytest.fail(f"ADNCE two-process {scenario} check timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
