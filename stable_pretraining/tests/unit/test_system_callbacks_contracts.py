"""Hardware counters degrade independently; sharding selection preserves callback ownership."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from stable_pretraining.callbacks.hardware_monitor import HardwareMonitor
from stable_pretraining.utils import fsdp2

pytestmark = pytest.mark.unit


@pytest.fixture
def nvml(monkeypatch):
    api = SimpleNamespace(
        nvmlInit=Mock(),
        nvmlShutdown=Mock(),
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(
            gpu=20 + 40 * handle
        ),
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(
            used=(handle + 1) * 1024**3, total=4 * 1024**3
        ),
        nvmlDeviceGetTemperature=lambda handle, _: 40 + 20 * handle,
        nvmlDeviceGetPowerUsage=lambda handle: 100_000 + 50_000 * handle,
    )
    monkeypatch.setitem(sys.modules, "pynvml", api)
    return api


@pytest.mark.parametrize("per_gpu", [False, True])
@pytest.mark.parametrize("failure", [None, "optional", "core"])
def test_gpu_metrics_aggregate_available_devices_only(
    nvml, monkeypatch, per_gpu, failure
):
    monitor = HardwareMonitor(
        log_cpu=False, log_ram=False, log_disk=False, log_net=False, log_per_gpu=per_gpu
    )
    monitor._init_capabilities()
    if failure == "optional":
        monkeypatch.setattr(
            nvml,
            "nvmlDeviceGetTemperature",
            Mock(side_effect=RuntimeError("permission")),
        )
        monkeypatch.setattr(
            nvml,
            "nvmlDeviceGetPowerUsage",
            Mock(side_effect=RuntimeError("permission")),
        )
    elif failure == "core":
        monkeypatch.setattr(
            nvml,
            "nvmlDeviceGetUtilizationRates",
            Mock(side_effect=[SimpleNamespace(gpu=20), RuntimeError("lost GPU")]),
        )
    result = monitor._sample()
    assert result["hardware/gpu_avg_util_pct"] == (20 if failure == "core" else 40)
    assert result["hardware/gpu_total_mem_used_gb"] == (1 if failure == "core" else 3)
    assert ("hardware/gpu0_util_pct" in result) is per_gpu
    if failure == "optional":
        assert "hardware/gpu_avg_temp_c" not in result
        assert "hardware/gpu_total_power_w" not in result
    else:
        assert result["hardware/gpu_avg_temp_c"] == (40 if failure == "core" else 50)
        assert result["hardware/gpu_total_power_w"] == (
            100 if failure == "core" else 250
        )


def test_gpu_metrics_survive_missing_host_monitor(nvml, monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    monitor = HardwareMonitor()
    monitor._init_capabilities()
    assert monitor._psutil is None
    assert monitor._sample()["hardware/gpu_avg_util_pct"] == 40


@pytest.mark.parametrize("shutdown_fails", [False, True])
def test_poll_loop_recovers_and_always_releases_nvml(nvml, monkeypatch, shutdown_fails):
    monitor = HardwareMonitor()
    monkeypatch.setattr(monitor._stop, "is_set", Mock(side_effect=[False, False, True]))
    monkeypatch.setattr(monitor._stop, "wait", Mock())
    monkeypatch.setattr(
        monitor,
        "_sample",
        Mock(side_effect=[OSError("temporary"), {"hardware/cpu_percent": 50.0}]),
    )
    if shutdown_fails:
        nvml.nvmlShutdown.side_effect = RuntimeError("shutdown")
    monitor._poll_loop()
    assert monitor._latest == {"hardware/cpu_percent": 50.0}
    nvml.nvmlShutdown.assert_called_once_with()


@pytest.mark.parametrize(
    "rank,rank_zero,enabled,expected",
    [
        (0, True, True, True),
        (1, True, True, False),
        (1, False, True, True),
        (0, True, False, False),
    ],
)
def test_validation_logging_respects_rank_and_opt_in(
    rank, rank_zero, enabled, expected
):
    monitor = HardwareMonitor(log_on_validation=enabled, rank_zero_only=rank_zero)
    monitor._latest = {"hardware/cpu_percent": 5.0}
    module = SimpleNamespace(log_dict=Mock())
    monitor.on_validation_batch_end(
        SimpleNamespace(global_rank=rank), module, None, None, 0
    )
    assert module.log_dict.called is expected


@pytest.mark.parametrize("mixed", [False, True])
def test_sharding_wraps_deepest_blocks_once_before_parent(monkeypatch, mixed):
    shared = nn.Linear(3, 3)
    inner = nn.Sequential(shared, nn.ReLU())
    root = nn.Sequential(inner, shared, nn.ModuleList([nn.Linear(3, 3)]))
    shard = Mock()
    monkeypatch.setattr(fsdp2, "fully_shard", shard)
    policy = fsdp2.MixedPrecisionPolicy(param_dtype=torch.bfloat16) if mixed else None
    mesh = object()
    fsdp2._shard_subtree(root, mesh, policy)
    wrapped = [call.args[0] for call in shard.call_args_list]
    assert wrapped[-1] is root
    assert wrapped.index(shared) < wrapped.index(inner)
    assert wrapped.count(shared) == 1
    assert root[2] not in wrapped and root[2][0] in wrapped
    for call in shard.call_args_list:
        assert call.kwargs["mesh"] is mesh
        assert ("mp_policy" in call.kwargs) is mixed


def test_default_sharding_excludes_callback_parameters_and_delegates_ema(monkeypatch):
    module = nn.Module()
    module.backbone = nn.Linear(3, 3)
    module.teacher_student = nn.Linear(3, 3)
    module.teacher_student.fsdp_setup = Mock()
    module.callbacks_modules = nn.ModuleDict({"probe": nn.Linear(3, 2)})
    module.metrics = nn.ModuleDict()
    module.identity = nn.Identity()
    shard = Mock()
    monkeypatch.setattr(fsdp2, "fully_shard", shard)
    mesh = object()
    assert fsdp2.default_parallelize_fn(module, {"data_parallel": mesh}) is module
    shard.assert_called_once_with(module.backbone, mesh=mesh)
    module.teacher_student.fsdp_setup.assert_called_once_with(mesh, None)
    assert fsdp2.default_parallelize_fn(nn.Identity(), None) is not None


@pytest.mark.parametrize("mesh", [None, {}, []])
def test_unnamed_mesh_is_preserved(mesh):
    assert fsdp2._data_parallel_mesh(mesh) is mesh
