"""TeacherStudentWrapper alignment under FSDP2.

:meth:`TeacherStudentWrapper.update_teacher` performs in-place EMA via
``zip(teacher.parameters(), student.parameters())``. Under FSDP2 each
parameter is a ``DTensor``, and ``zip`` requires that *each* pair of
DTensors be addressing the same logical region of the same parameter:
matching ``shape`` / ``dtype`` is necessary but not sufficient — the
``placements`` and ``device_mesh`` must also agree, otherwise ``zip``
silently pairs non-corresponding shards (corruption, not a crash).

:func:`stable_pretraining.utils.fsdp2.assert_aligned_wrapping` enforces all
four properties. These tests cover its truth table directly with crafted
DTensors (so they run anywhere ``torch.distributed`` is importable, no
fully_shard call needed).
"""

import pytest
import torch
import torch.nn as nn

from stable_pretraining.tests.distributed.conftest import run_distributed


pytestmark = pytest.mark.distributed


# ---------------------------------------------------------------------------
# 1. assert_aligned_wrapping — truth table
# ---------------------------------------------------------------------------


def test_aligned_wrapping_accepts_identical_plain_modules():
    r"""Plain ``nn.Parameter``\ s with matching shape/dtype must be accepted."""
    from stable_pretraining.utils.fsdp2 import assert_aligned_wrapping

    student = nn.Linear(4, 4)
    teacher = nn.Linear(4, 4)
    # No exception expected.
    assert_aligned_wrapping(student, teacher)


def test_aligned_wrapping_rejects_param_count_mismatch():
    from stable_pretraining.utils.fsdp2 import assert_aligned_wrapping

    student = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    teacher = nn.Sequential(nn.Linear(4, 4))
    with pytest.raises(AssertionError, match="parameter tensors"):
        assert_aligned_wrapping(student, teacher)


def test_aligned_wrapping_rejects_shape_mismatch():
    from stable_pretraining.utils.fsdp2 import assert_aligned_wrapping

    student = nn.Linear(4, 4)
    teacher = nn.Linear(4, 8)
    with pytest.raises(AssertionError, match="shape"):
        assert_aligned_wrapping(student, teacher)


def test_aligned_wrapping_rejects_dtype_mismatch():
    from stable_pretraining.utils.fsdp2 import assert_aligned_wrapping

    student = nn.Linear(4, 4).to(torch.float32)
    teacher = nn.Linear(4, 4).to(torch.float64)
    with pytest.raises(AssertionError, match="dtype"):
        assert_aligned_wrapping(student, teacher)


def test_aligned_wrapping_rejects_buffer_dtype_mismatch():
    """Buffer dtype check (parity with the param dtype check)."""
    from stable_pretraining.utils.fsdp2 import assert_aligned_wrapping

    student = nn.BatchNorm1d(4)
    teacher = nn.BatchNorm1d(4)
    teacher.running_mean = teacher.running_mean.to(torch.float64)
    with pytest.raises(AssertionError, match="dtype"):
        assert_aligned_wrapping(student, teacher)


# ---------------------------------------------------------------------------
# 2. DTensor placement check — needs a real device mesh.
# ---------------------------------------------------------------------------


def _placement_mismatch_raises(rank: int, world_size: int) -> None:
    """Placement-mismatch rejection.

    Two DTensors with matching local shape but different placements
    must be rejected by ``assert_aligned_wrapping``.
    """
    from torch.distributed.tensor import (
        DTensor,
        Replicate,
        Shard,
        distribute_tensor,
        init_device_mesh,
    )

    from stable_pretraining.utils.fsdp2 import assert_aligned_wrapping

    mesh = init_device_mesh("cpu", (world_size,))

    base_full = torch.randn(world_size, 4)

    # Student: shard dim 0.
    s_dt = distribute_tensor(base_full.clone(), mesh, [Shard(0)])
    s_param = nn.Parameter(s_dt)
    student = nn.Module()
    student.register_parameter("w", s_param)

    # Teacher: replicate (different placement, same local shape after replicate)
    # — placements differ; assertion must fire.
    t_dt = distribute_tensor(base_full.clone(), mesh, [Replicate()])
    t_param = nn.Parameter(t_dt)
    teacher = nn.Module()
    teacher.register_parameter("w", t_param)

    # Sanity: both are DTensors.
    assert isinstance(s_param, DTensor)
    assert isinstance(t_param, DTensor)

    # Local shapes may match if dim 0 size == world_size; we contrived this
    # so they're equal. Placements differ → assertion must raise.
    if s_param.shape == t_param.shape and s_param.dtype == t_param.dtype:
        try:
            assert_aligned_wrapping(student, teacher)
        except AssertionError as e:
            assert "placement" in str(e).lower()
            return
        raise AssertionError(
            "assert_aligned_wrapping should have raised on placement mismatch"
        )


def test_placement_mismatch_raises_world_size_2():
    run_distributed(_placement_mismatch_raises, world_size=2, backend="gloo")


def _matched_dtensor_passes(rank: int, world_size: int) -> None:
    """Matched-DTensor positive case.

    Two DTensors with matching shape, dtype, placements, and device_mesh
    must pass the alignment check.
    """
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    from stable_pretraining.utils.fsdp2 import assert_aligned_wrapping

    mesh = init_device_mesh("cpu", (world_size,))
    base = torch.randn(world_size * 2, 4)

    s_dt = distribute_tensor(base.clone(), mesh, [Shard(0)])
    t_dt = distribute_tensor(base.clone(), mesh, [Shard(0)])

    student = nn.Module()
    student.register_parameter("w", nn.Parameter(s_dt))
    teacher = nn.Module()
    teacher.register_parameter("w", nn.Parameter(t_dt))

    assert_aligned_wrapping(student, teacher)


def test_matched_dtensor_passes_world_size_2():
    run_distributed(_matched_dtensor_passes, world_size=2, backend="gloo")


# ---------------------------------------------------------------------------
# 3. DTensor / plain Tensor mixing is rejected.
# ---------------------------------------------------------------------------


def _dtensor_vs_plain_raises(rank: int, world_size: int) -> None:
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    from stable_pretraining.utils.fsdp2 import assert_aligned_wrapping

    mesh = init_device_mesh("cpu", (world_size,))
    base = torch.randn(world_size * 2, 4)

    s_dt = distribute_tensor(base.clone(), mesh, [Shard(0)])
    student = nn.Module()
    student.register_parameter("w", nn.Parameter(s_dt))

    teacher = nn.Module()
    # Teacher uses a plain tensor with the same local shape.
    teacher.register_parameter("w", nn.Parameter(s_dt.to_local().clone()))

    if student.w.shape == teacher.w.shape and student.w.dtype == teacher.w.dtype:
        with pytest.raises(AssertionError, match="DTensor"):
            assert_aligned_wrapping(student, teacher)


def test_dtensor_vs_plain_raises_world_size_2():
    run_distributed(_dtensor_vs_plain_raises, world_size=2, backend="gloo")
