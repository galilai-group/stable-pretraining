"""Regression tests for callback clocks, deferred logging, and progress output."""

from types import SimpleNamespace
from unittest.mock import Mock

import lightning as pl
import pytest
import torch
from torch import nn

from stable_pretraining.callbacks import factories, registry
from stable_pretraining.callbacks.teacher_student import TeacherStudentCallback
from stable_pretraining.callbacks.wd_schedule import WeightDecayUpdater

pytestmark = pytest.mark.unit


class _Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher = nn.Linear(1, 1)
        self.update_teacher = Mock()
        self.update_ema_coefficient = Mock()
        self._mark_updated = Mock()
        self.ema_coefficient = 0.9


@pytest.mark.parametrize("after_backward", [False, True])
@pytest.mark.parametrize("frequency", [1, 2, 3])
def test_teacher_updates_at_exact_frequency_without_repeating_accumulation_steps(
    after_backward, frequency
):
    wrapper = _Teacher()
    model = nn.Sequential(wrapper)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = SimpleNamespace(
        global_step=0, current_epoch=2, max_epochs=10, optimizers=[optimizer]
    )
    cb = TeacherStudentCallback(frequency, after_backward, verbose=False)
    cb.on_fit_start(trainer, model)
    cb.on_train_batch_end(trainer, model, {}, {}, 0)
    assert wrapper.update_teacher.call_count == 0
    for step in range(1, 7):
        cb.on_before_optimizer_step(trainer, model, optimizer)
        trainer.global_step = step
        cb.on_train_batch_end(trainer, model, {}, {}, step)
        cb.on_train_batch_end(trainer, model, {}, {}, step)
        assert wrapper.update_teacher.call_count == step // frequency
    assert wrapper._mark_updated.call_count == 6 // frequency
    wrapper.update_ema_coefficient.assert_called_with(2, 10)
    cb.on_fit_start(trainer, nn.Linear(2, 2))
    assert not cb._wrapper_found
    cb._update_all_wrappers(trainer, model)
    assert wrapper.update_teacher.call_count == 6 // frequency


@pytest.mark.parametrize("frequency", [0, -1, 1.5])
def test_teacher_rejects_invalid_update_frequency(frequency):
    with pytest.raises(ValueError, match="update_frequency"):
        TeacherStudentCallback(frequency)


@pytest.mark.parametrize(
    "schedule,midpoint",
    [("constant", 0.1), ("linear", 0.0625), ("cosine", 0.0625), ("exponential", 0.05)],
)
def test_weight_decay_uses_training_step_clock_and_round_trips_state(
    schedule, midpoint, monkeypatch
):
    p, q = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
    opt = torch.optim.SGD(
        [{"params": [p], "weight_decay": 0.7}, {"params": [q], "weight_decay": 0.8}],
        lr=0.1,
    )
    extra = torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=0.1)
    model = SimpleNamespace(optimizers=lambda **kwargs: [opt, extra])
    trainer = SimpleNamespace(
        global_step=5, estimated_stepping_batches=10, accumulate_grad_batches=2
    )
    cb = WeightDecayUpdater(
        schedule, 0.1, 0.025, param_group_indices=[1], opt_idx=0, verbose=True
    )
    log = Mock()
    monkeypatch.setattr("stable_pretraining.callbacks.wd_schedule._spt_log", log)
    cb.on_fit_start(trainer, model)
    cb.on_before_optimizer_step(trainer, model, extra)
    assert extra.param_groups[0]["weight_decay"] == 0
    cb.on_before_optimizer_step(trainer, model, opt)
    assert cb.total_steps == 10
    assert opt.param_groups[0]["weight_decay"] == 0.7
    assert opt.param_groups[1]["weight_decay"] == pytest.approx(midpoint)
    log.assert_called_once()
    restored = WeightDecayUpdater()
    restored.load_state_dict(cb.state_dict())
    assert restored.state_dict() == cb.state_dict()
    assert restored._compute_weight_decay(10) == pytest.approx(
        0.1 if schedule == "constant" else 0.025
    )


def test_weight_decay_supports_single_optimizer_and_unfiltered_groups():
    opt = torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=0.1)
    model = SimpleNamespace(optimizers=lambda **kwargs: opt)
    trainer = SimpleNamespace(global_step=0, estimated_stepping_batches=2)
    cb = WeightDecayUpdater("constant", 0.3, opt_idx=0, verbose=False)
    cb.on_fit_start(trainer, model)
    cb.on_before_optimizer_step(trainer, model, opt)
    assert opt.param_groups[0]["weight_decay"] == 0.3
    cb.schedule_type = "unsupported"
    with pytest.raises(ValueError, match="Unknown schedule_type"):
        cb._compute_weight_decay(1)


@pytest.fixture
def isolated_registry(monkeypatch):
    for name in ("_MODULE_REGISTRY", "_METRIC_BUFFER", "_DICT_BUFFER", "_IN_STEP"):
        monkeypatch.setattr(registry, name, {})


def test_logging_buffers_flushes_once_and_preserves_options(isolated_registry):
    model = Mock()
    cb = registry.ModuleRegistryCallback("test")
    registry._flush_buffer("absent")
    with pytest.warns(UserWarning, match="no module"):
        registry.log("a", 1, module_name="test")
    with pytest.warns(UserWarning, match="no module"):
        registry.log_dict({"a": 1}, module_name="test")
    cb.setup(None, model, "fit")
    assert registry.get_module("test") is model
    with pytest.warns(UserWarning, match="buffered"):
        registry.log("a", 2, module_name="test", on_epoch=True)
    with pytest.warns(UserWarning, match="buffered"):
        registry.log_dict({"b": 3}, module_name="test", sync_dist=True)
    assert not model.log.called
    cb.on_train_batch_start(None, model, {}, 0)
    model.log.assert_called_once_with("a", 2, on_epoch=True)
    model.log_dict.assert_called_once_with({"b": 3}, sync_dist=True)
    cb.on_train_batch_start(None, model, {}, 1)
    assert model.log.call_count == 1
    registry.log_dict({"c": 4}, module_name="test")
    model.log_dict.assert_called_with({"c": 4})
    cb.on_train_epoch_end(None, model)
    with pytest.warns(UserWarning, match="buffered"):
        registry.log("d", 5, module_name="test")
    with pytest.warns(UserWarning, match="1 buffered metric"):
        cb.teardown(None, model, "fit")
    assert registry.get_module("test") is None


def test_failed_buffer_flush_warns_and_drops_metrics(isolated_registry):
    model = Mock()
    model.log.side_effect = RuntimeError("closed logger")
    model.log_dict.side_effect = RuntimeError("closed logger")
    cb = registry.ModuleRegistryCallback("test")
    cb.setup(None, model, "fit")
    with pytest.warns(UserWarning, match="buffered"):
        registry.log("a", 1, module_name="test")
    with pytest.warns(UserWarning, match="buffered"):
        registry.log_dict({"b": 2}, module_name="test")
    with pytest.warns(UserWarning, match="Failed to flush") as captured:
        registry._flush_buffer("test")
    assert len(captured) == 2
    registry._flush_buffer("test")
    assert model.log.call_count == model.log_dict.call_count == 1
    cb.teardown(None, model, "fit")


@pytest.mark.parametrize("failure_point", ["start", "step", "batch_end"])
def test_failed_training_cleans_global_logging_before_next_run(
    isolated_registry, tmp_path, failure_point
):
    class Model(pl.LightningModule):
        def __init__(self, fail=False):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.fail = fail

        def training_step(self, batch, batch_idx):
            if self.fail and failure_point == "step":
                raise RuntimeError("interrupted run")
            return self.weight.square()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    class Interrupt(pl.Callback):
        def on_train_start(self, trainer, pl_module):
            if failure_point == "start":
                registry.log("pending", 1)
                registry.log_dict({"pending_dict": 2})
                raise RuntimeError("interrupted run")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if failure_point == "batch_end":
                raise RuntimeError("interrupted run")

    callback = registry.ModuleRegistryCallback()
    options = dict(
        accelerator="cpu",
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    data = torch.utils.data.DataLoader(torch.ones(2, 1), batch_size=1)
    trainer = pl.Trainer(callbacks=[callback, Interrupt()], **options)
    with pytest.raises(RuntimeError, match="interrupted run"):
        trainer.fit(Model(fail=True), data)
    for name in ("_MODULE_REGISTRY", "_METRIC_BUFFER", "_DICT_BUFFER", "_IN_STEP"):
        assert not getattr(registry, name)
    with pytest.warns(UserWarning, match="no module registered"):
        registry.log("after_failure", 3)
    callback.teardown(trainer, trainer.lightning_module, "fit")
    next_model = Model()
    pl.Trainer(callbacks=[callback], **options).fit(next_model, data)
    assert next_model.weight.item() == pytest.approx(0.8)
    assert registry.get_module() is None


@pytest.mark.parametrize("start", [None, 10.0])
def test_plain_progress_reports_metrics_on_interval(capsys, monkeypatch, start):
    bar = factories.PrintProgressBar(log_every_n_steps=2)
    bar._epoch_start = start
    monkeypatch.setattr(factories.time, "time", lambda: 12.0)
    trainer = SimpleNamespace(
        num_training_batches=8,
        current_epoch=1,
        max_epochs=None,
        progress_bar_metrics={"loss": 0.25},
    )
    bar.on_train_batch_end(trainer, None, None, None, 0)
    assert not capsys.readouterr().out
    bar.on_train_batch_end(trainer, None, None, None, 1)
    output = capsys.readouterr().out
    assert "step 2/8" in output and "loss: 0.25" in output


def test_rich_progress_teardown_tolerates_already_stopped_live_stack(monkeypatch):
    monkeypatch.setattr(
        factories._RichProgressBar, "_stop_progress", Mock(side_effect=IndexError)
    )
    factories.RichProgressBar()._stop_progress()
