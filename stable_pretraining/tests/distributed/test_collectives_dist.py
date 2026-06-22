"""Multi-process (gloo) correctness tests for ``utils.all_gather`` / ``all_reduce``.

These spawn 2 CPU processes via the shared spawn harness and assert the
collectives actually gather/reduce across ranks — the behavior that a
single-GPU unit run cannot exercise. Regression for the bug where the wrappers
discarded the functional collective's return value and silently echoed the
local input.
"""

import pytest

from stable_pretraining.tests.distributed import _dist_harness as H


@pytest.mark.distributed
class TestCollectives:
    """all_gather / all_reduce must move data across ranks, with autograd."""

    def test_all_gather_gathers_all_ranks(self):
        H.run_distributed(H.w_all_gather_values, world_size=2, backend="gloo")

    def test_all_gather_is_autograd_aware(self):
        H.run_distributed(H.w_all_gather_grad, world_size=2, backend="gloo")

    def test_all_reduce_sums(self):
        H.run_distributed(H.w_all_reduce_sum, world_size=2, backend="gloo")

    def test_barlow_twins_matches_single_process(self):
        H.run_distributed(H.w_barlow_matches_single_proc, world_size=2, backend="gloo")

    def test_contrastive_losses_run_under_ddp(self):
        H.run_distributed(H.w_contrastive_runs_under_ddp, world_size=2, backend="gloo")

    def test_ntxent_cross_gpu_negatives_match_global(self):
        H.run_distributed(
            H.w_ntxent_crossgpu_matches_global, world_size=2, backend="gloo"
        )

    def test_clip_cross_gpu_negatives_match_global(self):
        H.run_distributed(
            H.w_clip_crossgpu_matches_global, world_size=2, backend="gloo"
        )
