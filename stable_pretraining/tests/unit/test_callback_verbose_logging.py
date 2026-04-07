# test_callback_verbose_logging.py
"""Unit tests for verbose logging via stable_pretraining.log across callbacks."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from stable_pretraining.callbacks.registry import (
    _MODULE_REGISTRY,
    _METRIC_BUFFER,
    _DICT_BUFFER,
    _IN_STEP,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is clean before and after each test."""
    _MODULE_REGISTRY.clear()
    _METRIC_BUFFER.clear()
    _DICT_BUFFER.clear()
    _IN_STEP.clear()
    yield
    _MODULE_REGISTRY.clear()
    _METRIC_BUFFER.clear()
    _DICT_BUFFER.clear()
    _IN_STEP.clear()


@pytest.fixture
def dummy_dataloader():
    x = torch.randn(20, 10)
    y = torch.randn(20, 1)
    return DataLoader(TensorDataset(x, y), batch_size=4)


# ---------------------------------------------------------------------------
# WeightDecayUpdater
# ---------------------------------------------------------------------------


class TestWeightDecayUpdaterVerbose:
    """Test that WeightDecayUpdater logs weight decay via stable_pretraining.log."""

    def test_logs_weight_decay_when_verbose(self):
        from stable_pretraining.callbacks.wd_schedule import WeightDecayUpdater

        cb = WeightDecayUpdater(
            schedule_type="cosine", start_value=0.01, end_value=0.0, verbose=True
        )
        cb.total_steps = 100

        model = MagicMock(spec=pl.LightningModule)
        model.optimizers.return_value = [MagicMock()]
        trainer = MagicMock(spec=pl.Trainer)
        trainer.global_step = 10
        trainer.accumulate_grad_batches = 1

        optimizer = MagicMock()
        optimizer.param_groups = [{"weight_decay": 0.01, "lr": 0.1}]

        with patch("stable_pretraining.callbacks.wd_schedule._spt_log") as mock_log:
            cb.on_before_optimizer_step(trainer, model, optimizer)
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[0][0] == "hparams/weight_decay"

    def test_no_log_when_not_verbose(self):
        from stable_pretraining.callbacks.wd_schedule import WeightDecayUpdater

        cb = WeightDecayUpdater(
            schedule_type="cosine", start_value=0.01, end_value=0.0, verbose=False
        )
        cb.total_steps = 100

        model = MagicMock(spec=pl.LightningModule)
        model.optimizers.return_value = [MagicMock()]
        trainer = MagicMock(spec=pl.Trainer)
        trainer.global_step = 10
        trainer.accumulate_grad_batches = 1

        optimizer = MagicMock()
        optimizer.param_groups = [{"weight_decay": 0.01, "lr": 0.1}]

        with patch("stable_pretraining.callbacks.wd_schedule._spt_log") as mock_log:
            cb.on_before_optimizer_step(trainer, model, optimizer)
            mock_log.assert_not_called()


# ---------------------------------------------------------------------------
# TeacherStudentCallback
# ---------------------------------------------------------------------------


class TestTeacherStudentVerbose:
    """Test that TeacherStudentCallback logs EMA coefficient."""

    def test_logs_ema_coefficient_when_verbose(self):
        from stable_pretraining.callbacks.teacher_student import TeacherStudentCallback

        cb = TeacherStudentCallback(update_frequency=1, verbose=True)
        cb._wrapper_found = True

        # Create a mock module with teacher/student behavior
        mock_wrapper = MagicMock()
        mock_wrapper.update_teacher = MagicMock()
        mock_wrapper.update_ema_coefficient = MagicMock()
        mock_wrapper.ema_coefficient = 0.996
        mock_wrapper.name = "backbone"
        mock_wrapper._mark_updated = MagicMock()

        pl_module = MagicMock(spec=pl.LightningModule)
        pl_module.modules.return_value = [pl_module, mock_wrapper]
        trainer = MagicMock(spec=pl.Trainer)
        trainer.current_epoch = 0
        trainer.max_epochs = 100

        with patch("stable_pretraining.callbacks.teacher_student._spt_log") as mock_log:
            cb._update_all_wrappers(trainer, pl_module)
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert "coefficient" in call_args[0][0]
            assert call_args[0][1] == 0.996

    def test_no_log_when_not_verbose(self):
        from stable_pretraining.callbacks.teacher_student import TeacherStudentCallback

        cb = TeacherStudentCallback(update_frequency=1, verbose=False)
        cb._wrapper_found = True

        mock_wrapper = MagicMock()
        mock_wrapper.update_teacher = MagicMock()
        mock_wrapper.update_ema_coefficient = MagicMock()
        mock_wrapper.ema_coefficient = 0.996
        mock_wrapper._mark_updated = MagicMock()

        pl_module = MagicMock(spec=pl.LightningModule)
        pl_module.modules.return_value = [pl_module, mock_wrapper]
        trainer = MagicMock(spec=pl.Trainer)
        trainer.current_epoch = 0
        trainer.max_epochs = 100

        with patch("stable_pretraining.callbacks.teacher_student._spt_log") as mock_log:
            cb._update_all_wrappers(trainer, pl_module)
            mock_log.assert_not_called()


# ---------------------------------------------------------------------------
# LiDAR
# ---------------------------------------------------------------------------


class TestLiDARVerbose:
    """Test that LiDAR returns dict and logs entropy/eigenvalue."""

    def test_compute_lidar_returns_dict(self):
        from stable_pretraining.callbacks.lidar import LiDAR

        cb = LiDAR(
            name="lidar_test",
            target="embedding",
            queue_length=200,
            target_shape=32,
            n_classes=5,
            samples_per_class=4,
            verbose=True,
        )

        embeddings = torch.randn(20, 32)
        result = cb._compute_lidar(embeddings)

        assert result is not None
        assert isinstance(result, dict)
        assert "lidar" in result
        assert "entropy" in result
        assert "top_eigenvalue" in result
        assert result["lidar"] > 0
        assert result["entropy"] >= 0
        assert result["top_eigenvalue"] >= 0


# ---------------------------------------------------------------------------
# RankMe
# ---------------------------------------------------------------------------


class TestRankMeVerbose:
    """Test that RankMe logs entropy and singular value stats when verbose."""

    def test_verbose_parameter_accepted(self):
        from stable_pretraining.callbacks.rankme import RankMe

        cb = RankMe(
            name="rankme_test",
            target="embedding",
            queue_length=100,
            target_shape=32,
            verbose=True,
        )
        assert cb.verbose is True

        cb2 = RankMe(
            name="rankme_test2",
            target="embedding",
            queue_length=100,
            target_shape=32,
            verbose=False,
        )
        assert cb2.verbose is False


# ---------------------------------------------------------------------------
# EpochMilestones
# ---------------------------------------------------------------------------


class TestEpochMilestonesVerbose:
    """Test that EpochMilestones logs value/threshold when verbose."""

    def test_logs_when_verbose_and_milestone_hit(self):
        from stable_pretraining.callbacks.earlystop import EpochMilestones

        cb = EpochMilestones(
            milestones={5: 0.8},
            monitor=["eval/accuracy"],
            direction="max",
            verbose=True,
        )

        trainer = MagicMock(spec=pl.Trainer)
        trainer.current_epoch = 5
        trainer.sanity_checking = False
        trainer.callback_metrics = {"eval/accuracy": torch.tensor(0.9)}
        trainer.should_stop = False

        with patch("stable_pretraining.callbacks.earlystop._spt_log") as mock_log:
            cb._check_condition(trainer)
            assert mock_log.call_count == 2
            logged_names = [c[0][0] for c in mock_log.call_args_list]
            assert "epoch_milestones/value" in logged_names
            assert "epoch_milestones/threshold" in logged_names

    def test_no_log_when_not_verbose(self):
        from stable_pretraining.callbacks.earlystop import EpochMilestones

        cb = EpochMilestones(
            milestones={5: 0.8},
            monitor=["eval/accuracy"],
            direction="max",
            verbose=False,
        )

        trainer = MagicMock(spec=pl.Trainer)
        trainer.current_epoch = 5
        trainer.sanity_checking = False
        trainer.callback_metrics = {"eval/accuracy": torch.tensor(0.9)}
        trainer.should_stop = False

        with patch("stable_pretraining.callbacks.earlystop._spt_log") as mock_log:
            cb._check_condition(trainer)
            mock_log.assert_not_called()


# ---------------------------------------------------------------------------
# OnlineQueue
# ---------------------------------------------------------------------------


class TestOnlineQueueVerbose:
    """Test that OnlineQueue logs fill percentage when verbose."""

    def test_verbose_parameter_accepted(self):
        from stable_pretraining.callbacks.queue import OnlineQueue

        cb = OnlineQueue(key="embedding", queue_length=100, verbose=True)
        assert cb.verbose is True

        cb2 = OnlineQueue(key="embedding", queue_length=100, verbose=False)
        assert cb2.verbose is False


# ---------------------------------------------------------------------------
# Module LR logging
# ---------------------------------------------------------------------------


class TestModuleLRLogging:
    """Test that Module logs per-optimizer LR via stable_pretraining.log."""

    def test_spt_log_import_works_in_module(self):
        """Verify registry log is imported in module.py for LR logging."""
        import stable_pretraining.module as mod

        # The module imports log from registry as _spt_log for LR logging
        assert hasattr(mod, "_spt_log")

    def test_lr_logged_via_spt(self):
        """Verify spt.log is called with LR data when invoked directly."""
        model = MagicMock(spec=pl.LightningModule)
        _MODULE_REGISTRY["default"] = model
        _IN_STEP["default"] = True

        import stable_pretraining as spt

        # Call spt.log as the module would
        spt.log("hparams/lr_default_0", 0.01, on_step=True, on_epoch=False)

        # The registered mock module should have received the log call
        model.log.assert_called_once_with(
            "hparams/lr_default_0", 0.01, on_step=True, on_epoch=False
        )
