"""Worker bodies for the LeHami DDP tests (importable so spawn can pickle them).

Each worker runs inside an initialized gloo process group (see
:func:`._dist_harness.run_distributed`). Every rank builds the same global
batch, takes its own equal slice, and compares its SIGReg value with
single-process references.
"""

from unittest import mock

import torch

B, F, D = 4, 3, 5


def _loss(before_init: bool):
    from stable_pretraining.losses import LeHamiLoss

    if not before_init:
        return LeHamiLoss(n_slices=8)
    # Lightning and Manager build the model before the process group exists.
    with mock.patch("torch.distributed.is_initialized", return_value=False):
        return LeHamiLoss(n_slices=8)


def _sigreg(loss_fn, q_reg: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(q_reg.shape[0], 1, D)
    return loss_fn(zeros, zeros, zeros, zeros, q_reg)[3]


def _batches(rank: int, world_size: int):
    torch.manual_seed(0)  # identical global batch on every rank
    q_global = torch.randn(B * world_size, F, D)
    q_local = q_global[rank * B : (rank + 1) * B].clone().requires_grad_(True)
    return q_global, q_local


def w_sigreg_is_per_gpu_when_built_before_init(rank, world_size):
    """The Lightning / Manager order: SIGReg sees only the local batch."""
    q_global, q_local = _batches(rank, world_size)
    value = _sigreg(_loss(before_init=True), q_local)
    local_ref = _sigreg(_loss(before_init=True), q_local.detach())
    global_ref = _sigreg(_loss(before_init=True), q_global)
    assert torch.allclose(value, local_ref, rtol=1e-5, atol=1e-6), (value, local_ref)
    assert not torch.allclose(value, global_ref, rtol=1e-3), (value, global_ref)
    value.backward()
    assert torch.isfinite(q_local.grad).all()


def w_sigreg_is_global_when_built_after_init(rank, world_size):
    """Built inside the process group, SIGReg should test the global batch.

    Currently fails: ``EppsPulley`` drops the result of the functional
    ``all_reduce``, so each rank keeps its local statistic, scaled by the
    world size. Used by a strict ``xfail`` test.
    """
    q_global, q_local = _batches(rank, world_size)
    value = _sigreg(_loss(before_init=False), q_local)
    global_ref = _sigreg(_loss(before_init=True), q_global)
    assert torch.allclose(value, global_ref, rtol=1e-5, atol=1e-6), (value, global_ref)
    value.backward()
    assert torch.isfinite(q_local.grad).all()
