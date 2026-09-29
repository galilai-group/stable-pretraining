"""Optional callbacks stay bounded and respect hook and process ownership."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from stable_pretraining.callbacks import env_info, factories
from stable_pretraining.callbacks.earlystop import EpochMilestones
from stable_pretraining.callbacks.hardware_monitor import HardwareMonitor
from stable_pretraining.callbacks.hp_metric import HPMetricLogger
from stable_pretraining.callbacks.pca_visualizer import PCATokenVisualizer
from stable_pretraining.callbacks.trainer_info import ModuleSummary
from stable_pretraining._config import get_config

pytestmark = pytest.mark.unit


def test_hparam_metric_tracks_best_value_across_epochs():
    callback = HPMetricLogger("loss")
    module = SimpleNamespace()
    for value, expected in [(3.0, 3.0), (2.0, 2.0), (4.0, 2.0)]:
        callback.on_validation_epoch_end(
            SimpleNamespace(callback_metrics={"loss": torch.tensor(value)}), module
        )
        assert module.hp_metric.item() == expected


def test_milestones_only_fire_in_selected_hook_at_selected_epoch():
    with pytest.raises(ValueError, match="can't both be None"):
        EpochMilestones({1: 0.5})
    trainer = SimpleNamespace(
        current_epoch=0,
        callback_metrics={"loss": 0.8},
        sanity_checking=False,
        should_stop=False,
    )
    callback = EpochMilestones({1: 0.5}, monitor="loss")
    callback.on_train_epoch_end(trainer, None)
    callback.on_validation_epoch_end(trainer, None)
    assert not trainer.should_stop
    callback.after_validation = False
    trainer.current_epoch = 1
    callback.on_validation_epoch_end(trainer, None)
    assert not trainer.should_stop


def test_pca_visualizer_convolutional_features_and_stage_gates(monkeypatch):
    callback = PCATokenVisualizer("pca", "features", log_on=("train", "test"))
    trainer = SimpleNamespace(current_epoch=0, sanity_checking=True, global_rank=0)
    assert not callback._should_fire(trainer, 0)
    trainer.sanity_checking = False
    assert not callback._should_fire(trainer, 1)
    render = Mock()
    monkeypatch.setattr(callback, "_render_and_log", render)
    batch = {"image": torch.rand(2, 3, 8, 8), "features": torch.rand(2, 8, 2, 2)}
    callback.on_train_batch_end(trainer, None, {}, batch, 1)
    render.assert_not_called()
    callback.on_train_batch_end(trainer, None, {}, batch, 0)
    assert render.call_args.args[-1] == "train"
    callback.on_test_batch_end(trainer, None, {}, batch, 0)
    assert render.call_args.args[-1] == "test"
    with pytest.raises(ValueError, match="features must be"):
        callback.on_test_batch_end(
            trainer, None, {}, dict(batch, features=torch.ones(2, 8)), 0
        )


def test_module_summary_does_not_materialize_lazy_state(monkeypatch):
    from stable_pretraining.callbacks import trainer_info

    messages = []
    monkeypatch.setattr(trainer_info.logging, "info", messages.append)
    model = nn.Sequential(nn.LazyLinear(4), nn.LazyBatchNorm1d())
    ModuleSummary().setup(None, model, "fit")
    assert model[0].has_uninitialized_params() and model[1].has_uninitialized_params()
    assert any("Uninitialized parameters" in value for value in messages)


def test_hardware_setup_is_idempotent(monkeypatch):
    callback = HardwareMonitor()
    thread = Mock()
    thread.is_alive.return_value = True
    callback._thread = thread
    callback.setup(SimpleNamespace(global_rank=0), None, "fit")
    thread.start.assert_not_called()
    assert callback._thread is thread


def test_environment_teardown_wait_is_bounded(monkeypatch):
    callback = env_info.EnvironmentDumpCallback()
    callback._dump_thread = Mock()
    callback._dump_thread.is_alive.return_value = True
    warning = Mock()
    monkeypatch.setattr(env_info.logger, "warning", warning)
    callback.teardown(None, None, "fit")
    callback._dump_thread.join.assert_called_once_with(timeout=30)
    assert "did not complete" in warning.call_args.args[0]


def test_environment_versioning_stops_after_one_thousand_collisions(
    tmp_path, monkeypatch
):
    callback = env_info.EnvironmentDumpCallback()
    monkeypatch.setattr(Path, "exists", lambda _: True)
    result = callback._get_versioned_path(tmp_path / "environment.json")
    assert result.parent == tmp_path and result.suffix == ".json"
    assert result.name.startswith("environment_") and "_v" not in result.name


def test_environment_failure_is_logged_and_missing_git_is_optional(
    tmp_path, monkeypatch
):
    callback = env_info.EnvironmentDumpCallback()
    monkeypatch.setattr(env_info.subprocess, "check_output", lambda *a, **kw: "")
    assert callback._get_git_info() is None
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert callback._get_slurm_info()["SLURM_JOB_ID"] == "123"
    error = Mock()
    monkeypatch.setattr(env_info.logger, "error", error)
    monkeypatch.setattr(
        callback, "_get_python_info", Mock(side_effect=OSError("unavailable"))
    )
    callback._dump_environment(tmp_path)
    assert "unavailable" in error.call_args.args[0]


@pytest.mark.parametrize("style", ["rich", "auto"])
def test_progress_bar_uses_rich_when_explicit_or_interactive(monkeypatch, style):
    monkeypatch.setattr(get_config(), "_progress_bar", style)
    monkeypatch.setattr(factories.os, "isatty", lambda _: True)
    callback = factories._make_progress_bar()
    assert isinstance(callback, factories.RichProgressBar)
