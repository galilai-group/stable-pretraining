"""JAX recovery and optional state do not disturb trainable parameters."""

from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from stable_pretraining.jax import manager, trainer, utils
from stable_pretraining.jax.backbone.teacher_student import TeacherStudentWrapper
from stable_pretraining.jax.callbacks.probe import OnlineProbe
from stable_pretraining.jax.callbacks.teacher_student import TeacherStudentCallback
from stable_pretraining.jax.losses.utils import off_diagonal

pytestmark = [pytest.mark.unit, pytest.mark.jax]


class _Network(nnx.Module):
    def __init__(self):
        self.layer = nnx.Linear(2, 2, rngs=nnx.Rngs(7))
        self.counter = nnx.Variable(jnp.array(4.0))

    def __call__(self, x):
        return self.layer(x)


def test_teacher_default_forward_and_nonparameter_state_are_preserved():
    wrapper = TeacherStudentWrapper(_Network())
    x = jnp.ones((2, 2))
    old = np.asarray(wrapper(x))
    wrapper.student.layer.kernel[...] += 2
    wrapper.student.counter[...] = 99
    wrapper.update_teacher(0.0)
    assert float(wrapper.teacher.counter[...]) == 4
    np.testing.assert_allclose(wrapper(x), wrapper.student(x))
    assert not np.allclose(old, wrapper(x))
    container = SimpleNamespace(network=wrapper)
    callback = TeacherStudentCallback(update_schedule=False)
    callback.on_train_epoch_start(
        SimpleNamespace(current_epoch=5, max_epochs=5), container
    )
    assert wrapper.ema == wrapper.base_ema_coefficient


def test_probe_skips_missing_labels_without_updating_weights():
    probe = OnlineProbe("probe", nnx.Linear(2, 2, rngs=nnx.Rngs(1)))
    before = np.asarray(probe.probe.kernel[...]).copy()
    runtime = SimpleNamespace(log=Mock())
    probe.on_train_batch_end(runtime, None, {"embedding": jnp.ones((2, 2))}, {}, 0)
    probe.on_validation_batch_end(runtime, None, {}, {}, 0)
    probe.on_validation_epoch_end(runtime, None)
    np.testing.assert_array_equal(probe.probe.kernel[...], before)
    runtime.log.assert_not_called()


def test_disabled_checkpointing_skips_save_and_registry(monkeypatch):
    runtime = manager.Manager.__new__(manager.Manager)
    runtime.ckpt_path = None
    runtime.run_dir = None
    runtime.checkpoint()
    runtime._inject_registry_logger()
    assert not hasattr(runtime, "trainer")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "not-an-int")
    assert not manager._is_slurm_requeue()


def test_requeue_reports_unavailable_scheduler_command(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    launch = Mock(side_effect=FileNotFoundError("scontrol"))
    monkeypatch.setattr(manager.subprocess, "run", launch)
    manager.Manager.__new__(manager.Manager).requeue()
    launch.assert_called_once_with(["scontrol", "requeue", "123"], check=False)


def test_single_device_parallel_fallback_and_nested_batch_metadata(monkeypatch):
    loggers = [object(), object()]
    runtime = trainer.Trainer(logger=tuple(loggers))
    assert runtime.loggers == loggers
    monkeypatch.setattr(jax, "device_count", lambda: 1)
    runtime._setup_data_parallel(_Network())
    assert runtime._n_devices == 1 and runtime._batch_sharding is None
    result = trainer._batch_to_jax({"image": np.ones(2), "metadata": ("name", 3)})
    assert result["metadata"] == ("name", 3)
    np.testing.assert_array_equal(result["image"], np.ones(2))


def test_weight_copy_rejects_mismatched_structure_and_copies_nested_linear():
    target = _Network()
    with pytest.raises(ValueError, match="count mismatch"):
        utils.copy_torch_params_(target, torch.nn.Identity())
    source = torch.nn.Linear(2, 2)
    utils.copy_torch_params_(target, source)
    np.testing.assert_allclose(
        target.layer.kernel[...], source.weight.detach().numpy().T
    )
    with pytest.raises(ValueError, match="square"):
        off_diagonal(jnp.ones((2, 3)))
