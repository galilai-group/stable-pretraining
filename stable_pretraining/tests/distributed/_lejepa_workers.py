"""Worker bodies for the LeJEPA SIGReg DDP tests (importable so spawn can pickle them).

Each worker runs inside an initialized gloo process group (see
:func:`._dist_harness.run_distributed`). Every rank builds the same global
batch and takes its own equal slice.
"""

from unittest import mock

import torch

N, D = 8, 6


def _single_process(module_cls, **kwargs):
    with mock.patch("torch.distributed.is_initialized", return_value=False):
        return module_cls(**kwargs)


def _slices(rank: int, world_size: int, dim: int):
    torch.manual_seed(0)  # identical global batch on every rank
    x_global = torch.randn(N * world_size, dim)
    x_local = x_global[rank * N : (rank + 1) * N].clone().requires_grad_(True)
    return x_global, x_local


def w_epps_pulley_tests_the_global_batch(rank, world_size):
    """Under DDP, each rank's Epps-Pulley statistic is that of the global batch."""
    from stable_pretraining.methods.lejepa import EppsPulley

    x_global, x_local = _slices(rank, world_size, 4)
    value = EppsPulley()(x_local)
    expected = _single_process(EppsPulley)(x_global)
    assert torch.allclose(value, expected, rtol=1e-5, atol=1e-6), (value, expected)
    value.sum().backward()
    assert torch.isfinite(x_local.grad).all()


def w_sliced_epps_pulley_tests_the_global_batch(rank, world_size):
    """Same for the sliced test, whose projections are shared across ranks."""
    from stable_pretraining.methods.lejepa import SlicedEppsPulley

    x_global, x_local = _slices(rank, world_size, D)
    value = SlicedEppsPulley(num_slices=16)(x_local)
    expected = _single_process(SlicedEppsPulley, num_slices=16)(x_global)
    assert torch.allclose(value, expected, rtol=1e-5, atol=1e-6), (value, expected)
    value.backward()
    assert torch.isfinite(x_local.grad).all()
