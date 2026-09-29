"""Real CPU device meshes verify sharding topology before GPU training starts."""

import os

import lightning as pl
import pytest
import torch
import torch.distributed as dist
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch.accelerators import CPUAccelerator
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from stable_pretraining.utils.fsdp2 import (
    StablePretrainingFSDP2,
    assert_aligned_wrapping,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def process_group(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith(("SLURM_", "TORCHELASTIC_", "MASTER_")) or key in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
        ):
            monkeypatch.delenv(key)
    assert not dist.is_initialized()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'mesh-rendezvous'}",
        rank=0,
        world_size=1,
    )
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mismatch", ["plain", "placement"])
def test_teacher_student_alignment_rejects_different_tensor_layouts(
    process_group, mismatch
):
    student, teacher = nn.Module(), nn.Module()
    value = torch.ones(2, 2)
    student.weight = nn.Parameter(
        distribute_tensor(value, process_group, [Replicate()])
    )
    teacher.weight = nn.Parameter(
        value.clone()
        if mismatch == "plain"
        else distribute_tensor(value, process_group, [Shard(0)])
    )
    with pytest.raises(RuntimeError, match="(DTensor mismatch|sharding mismatch)"):
        assert_aligned_wrapping(student, teacher)
    teacher.weight = nn.Parameter(
        distribute_tensor(value, process_group, [Replicate()])
    )
    assert_aligned_wrapping(student, teacher)


@pytest.mark.parametrize("size", ["auto", 1])
def test_fsdp_strategy_builds_data_parallel_mesh_and_attaches_to_module(
    process_group, size
):
    strategy = StablePretrainingFSDP2(
        data_parallel_size=size, tensor_parallel_size=size, process_group_backend="gloo"
    )
    strategy.parallel_devices = [torch.device("cpu")]
    strategy.cluster_environment = LightningEnvironment()
    strategy.accelerator = CPUAccelerator()
    module = pl.LightningModule()
    strategy.connect(module)
    strategy.setup_environment()
    assert strategy.device_mesh.shape == (1, 1)
    assert strategy.device_mesh.mesh_dim_names == ("data_parallel", "tensor_parallel")
    assert module._device_mesh is strategy.device_mesh
    assert strategy._data_parallel_size == strategy._tensor_parallel_size == 1
