"""Real-process regressions for failure reporting in the distributed harness."""

import os
import time

import pytest
import torch.multiprocessing as mp

from stable_pretraining.tests.distributed import _dist_harness as H


def _exit_abruptly(rank, world_size):
    os._exit(7)


def _wait_indefinitely(rank, world_size):
    time.sleep(60)


def _raise_exception(rank, world_size):
    raise ValueError("worker exception sentinel")


@pytest.mark.distributed
class TestDistributedHarness:
    """A missing error-queue message must never turn a failed worker into a pass."""

    def test_abrupt_worker_exit_fails(self):
        with pytest.raises(
            AssertionError, match=r"rank 0.*\nWorker exited with code 7"
        ):
            H.run_distributed(_exit_abruptly, world_size=1, timeout=30)

    def test_worker_timeout_fails_and_cleans_up(self):
        previous_children = {child.pid for child in mp.active_children()}
        with pytest.raises(AssertionError, match=r"rank 0.*\nWorker timed out"):
            H.run_distributed(_wait_indefinitely, world_size=1, timeout=0.25)
        assert {child.pid for child in mp.active_children()} <= previous_children

    def test_worker_exception_keeps_traceback(self):
        with pytest.raises(
            AssertionError, match="ValueError: worker exception sentinel"
        ):
            H.run_distributed(_raise_exception, world_size=1, timeout=30)
