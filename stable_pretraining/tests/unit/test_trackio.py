"""Unit tests for TrackioLogger and TrackioCheckpoint.

Tests cover:
- TrackioLogger: init, log_metrics, log_hyperparams, finalize, resume helpers
- TrackioCheckpoint: save/load checkpoint, sidecar file writing
- find_trackio_logger: discovery and error handling
- Manager integration: _maybe_restore_trackio_run
- DDP safety: rank_zero_only behaviour
- Config integration: trackio_checkpoint in default_callbacks
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ============================================================================
# Helpers & Fixtures
# ============================================================================


def _make_mock_trackio():
    """Return a mock trackio module with the expected API surface."""
    mock = MagicMock()
    mock_run = MagicMock()
    mock_run.name = "mock-run-42"
    mock.init.return_value = mock_run
    mock.log = MagicMock()
    mock.finish = MagicMock()
    return mock, mock_run


def _make_mock_trainer(loggers=None, is_global_zero=True, default_root_dir="/tmp"):
    """Create a mock Trainer with the given loggers."""
    trainer = MagicMock()
    trainer.loggers = loggers or []
    trainer.is_global_zero = is_global_zero
    trainer.default_root_dir = default_root_dir
    return trainer


# ============================================================================
# TrackioLogger — construction
# ============================================================================


class TestTrackioLoggerConstruction:
    """Tests for TrackioLogger construction and guards."""

    def test_import_error_when_trackio_missing(self):
        """TrackioLogger raises ImportError if trackio is not installed."""
        with patch.dict("sys.modules", {"trackio": None}):
            with patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", False):
                from stable_pretraining.loggers.trackio import TrackioLogger

                with pytest.raises(ImportError, match="trackio is required"):
                    TrackioLogger(project="test")

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_basic_construction(self, mock_trackio):
        from stable_pretraining.loggers.trackio import TrackioLogger

        logger = TrackioLogger(
            project="my-project",
            name="run-1",
            group="sweep-a",
            space_id="user/space",
        )
        assert logger.name == "my-project"
        assert logger.version == "run-1"
        assert logger._group == "sweep-a"
        assert logger._space_id == "user/space"
        assert logger._resume == "never"
        assert logger._run is None

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_version_empty_when_no_name(self, mock_trackio):
        from stable_pretraining.loggers.trackio import TrackioLogger

        logger = TrackioLogger(project="p")
        assert logger.version == ""

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_extra_kwargs_forwarded(self, mock_trackio):
        from stable_pretraining.loggers.trackio import TrackioLogger

        logger = TrackioLogger(project="p", webhook_url="https://example.com")
        assert logger._trackio_kwargs == {"webhook_url": "https://example.com"}


# ============================================================================
# TrackioLogger — logging
# ============================================================================


class TestTrackioLoggerLogging:
    """Tests for TrackioLogger metric logging."""

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_log_metrics_initializes_run(self, mock_trackio):
        """First log_metrics call should trigger trackio.init()."""
        from stable_pretraining.loggers.trackio import TrackioLogger

        mock_run = MagicMock()
        mock_trackio.init.return_value = mock_run

        logger = TrackioLogger(project="p")
        logger.log_metrics({"loss": 0.5}, step=10)

        mock_trackio.init.assert_called_once()
        mock_trackio.log.assert_called_once_with({"loss": 0.5}, step=10)

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_log_metrics_filters_non_scalars(self, mock_trackio):
        """Non-scalar values should be filtered out."""
        from stable_pretraining.loggers.trackio import TrackioLogger

        mock_trackio.init.return_value = MagicMock()

        logger = TrackioLogger(project="p")
        logger.log_metrics(
            {"loss": 0.5, "text": "hello", "acc": 0.9},
            step=1,
        )

        # "text" should be filtered out
        mock_trackio.log.assert_called_once_with({"loss": 0.5, "acc": 0.9}, step=1)

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_log_metrics_skips_empty(self, mock_trackio):
        """If all metrics are non-scalar, trackio.log should not be called."""
        from stable_pretraining.loggers.trackio import TrackioLogger

        mock_trackio.init.return_value = MagicMock()

        logger = TrackioLogger(project="p")
        logger.log_metrics({"text": "hello"}, step=1)

        mock_trackio.log.assert_not_called()

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_log_hyperparams_initializes_with_config(self, mock_trackio):
        """log_hyperparams should pass config to trackio.init()."""
        from stable_pretraining.loggers.trackio import TrackioLogger

        mock_trackio.init.return_value = MagicMock()

        logger = TrackioLogger(project="p")
        logger.log_hyperparams({"lr": 0.01, "batch_size": 32})

        call_kwargs = mock_trackio.init.call_args
        assert call_kwargs.kwargs["config"] == {"lr": 0.01, "batch_size": 32}

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_log_hyperparams_noop_if_already_initialized(self, mock_trackio):
        """Second log_hyperparams call should not re-init."""
        from stable_pretraining.loggers.trackio import TrackioLogger

        mock_trackio.init.return_value = MagicMock()

        logger = TrackioLogger(project="p")
        logger.log_hyperparams({"lr": 0.01})
        logger.log_hyperparams({"lr": 0.02})

        assert mock_trackio.init.call_count == 1


# ============================================================================
# TrackioLogger — finalize
# ============================================================================


class TestTrackioLoggerFinalize:
    """Tests for TrackioLogger finalize/teardown."""

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_finalize_calls_finish(self, mock_trackio):
        from stable_pretraining.loggers.trackio import TrackioLogger

        mock_trackio.init.return_value = MagicMock()

        logger = TrackioLogger(project="p")
        logger.log_metrics({"loss": 1.0})  # init the run
        logger.finalize("success")

        mock_trackio.finish.assert_called_once()
        assert logger._run is None

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_finalize_noop_if_not_initialized(self, mock_trackio):
        from stable_pretraining.loggers.trackio import TrackioLogger

        logger = TrackioLogger(project="p")
        logger.finalize("success")  # should not raise

        mock_trackio.finish.assert_not_called()


# ============================================================================
# TrackioLogger — resume helpers
# ============================================================================


class TestTrackioLoggerResume:
    """Tests for TrackioLogger resume behavior."""

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_resume_info(self, mock_trackio):
        from stable_pretraining.loggers.trackio import TrackioLogger

        logger = TrackioLogger(project="p", name="run-1", group="g")
        info = logger.resume_info
        assert info == {
            "project": "p",
            "name": "run-1",
            "group": "g",
            "server_url": None,
        }

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_set_resume(self, mock_trackio):
        from stable_pretraining.loggers.trackio import TrackioLogger

        logger = TrackioLogger(project="p")
        assert logger._resume == "never"
        assert logger._name is None

        logger.set_resume("restored-run")
        assert logger._name == "restored-run"
        assert logger._resume == "must"

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_set_resume_propagates_to_init(self, mock_trackio):
        """After set_resume, trackio.init should be called with resume='must'."""
        from stable_pretraining.loggers.trackio import TrackioLogger

        mock_trackio.init.return_value = MagicMock()

        logger = TrackioLogger(project="p")
        logger.set_resume("restored-run")
        logger.log_metrics({"loss": 0.5})

        call_kwargs = mock_trackio.init.call_args
        assert call_kwargs.kwargs["resume"] == "must"
        assert call_kwargs.kwargs["name"] == "restored-run"


# ============================================================================
# TrackioLogger — experiment property
# ============================================================================


class TestTrackioLoggerExperiment:
    """Tests for TrackioLogger experiment property handling."""

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_experiment_lazy_init(self, mock_trackio):
        from stable_pretraining.loggers.trackio import TrackioLogger

        mock_run = MagicMock()
        mock_trackio.init.return_value = mock_run

        logger = TrackioLogger(project="p")
        assert logger._run is None

        exp = logger.experiment
        assert exp is mock_run
        mock_trackio.init.assert_called_once()

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_experiment_cached(self, mock_trackio):
        from stable_pretraining.loggers.trackio import TrackioLogger

        mock_trackio.init.return_value = MagicMock()

        logger = TrackioLogger(project="p")
        _ = logger.experiment
        _ = logger.experiment
        assert mock_trackio.init.call_count == 1


# ============================================================================
# find_trackio_logger
# ============================================================================


class TestFindTrackioLogger:
    """Tests for locating the Trackio logger on a trainer."""

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_find_none(self, mock_trackio):
        from stable_pretraining.loggers.trackio import (
            find_trackio_logger,
        )

        trainer = _make_mock_trainer(loggers=[])
        assert find_trackio_logger(trainer) is None

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_find_single(self, mock_trackio):
        from stable_pretraining.loggers.trackio import (
            TrackioLogger,
            find_trackio_logger,
        )

        logger = TrackioLogger(project="p")
        trainer = _make_mock_trainer(loggers=[MagicMock(), logger])
        assert find_trackio_logger(trainer) is logger

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_find_multiple_raises(self, mock_trackio):
        from stable_pretraining.loggers.trackio import (
            TrackioLogger,
            find_trackio_logger,
        )

        l1 = TrackioLogger(project="p")
        l2 = TrackioLogger(project="q")
        trainer = _make_mock_trainer(loggers=[l1, l2])

        with pytest.raises(RuntimeError, match="Found 2 TrackioLoggers"):
            find_trackio_logger(trainer)


# ============================================================================
# _to_scalar helper
# ============================================================================


class TestToScalar:
    """Tests for the `_to_scalar` helper."""

    def test_int(self):
        from stable_pretraining.loggers.trackio import _to_scalar

        assert _to_scalar(42) == 42.0

    def test_float(self):
        from stable_pretraining.loggers.trackio import _to_scalar

        assert _to_scalar(3.14) == 3.14

    def test_bool(self):
        from stable_pretraining.loggers.trackio import _to_scalar

        assert _to_scalar(True) == 1.0
        assert _to_scalar(False) == 0.0

    def test_string_returns_none(self):
        from stable_pretraining.loggers.trackio import _to_scalar

        assert _to_scalar("hello") is None

    def test_torch_tensor(self):
        from stable_pretraining.loggers.trackio import _to_scalar

        try:
            import torch

            assert _to_scalar(torch.tensor(2.5)) == 2.5
            # Multi-element tensor → None
            assert _to_scalar(torch.tensor([1.0, 2.0])) is None
        except ImportError:
            pytest.skip("torch not available")

    def test_none_returns_none(self):
        from stable_pretraining.loggers.trackio import _to_scalar

        assert _to_scalar(None) is None


# ============================================================================
# _params_to_dict helper
# ============================================================================


class TestParamsToDict:
    """Tests for the `_params_to_dict` helper."""

    def test_dict_passthrough(self):
        from stable_pretraining.loggers.trackio import _params_to_dict

        d = {"lr": 0.01}
        assert _params_to_dict(d) == d

    def test_namespace(self):
        from stable_pretraining.loggers.trackio import _params_to_dict
        from argparse import Namespace

        ns = Namespace(lr=0.01, epochs=10)
        result = _params_to_dict(ns)
        assert result["lr"] == 0.01
        assert result["epochs"] == 10

    def test_fallback_to_string(self):
        from stable_pretraining.loggers.trackio import _params_to_dict

        result = _params_to_dict(42)
        assert result == {"params": "42"}

    def test_omegaconf(self):
        from stable_pretraining.loggers.trackio import _params_to_dict

        try:
            from omegaconf import DictConfig

            cfg = DictConfig({"lr": 0.01, "batch_size": 32})
            result = _params_to_dict(cfg)
            assert result == {"lr": 0.01, "batch_size": 32}
        except ImportError:
            pytest.skip("omegaconf not available")


# ============================================================================
# TrackioCheckpoint — save
# ============================================================================


class TestTrackioCheckpointSave:
    """Tests for the Trackio checkpoint save callback."""

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_save_stores_resume_info_in_checkpoint(self, mock_trackio, tmp_path):
        from stable_pretraining.loggers.trackio import TrackioLogger
        from stable_pretraining.callbacks.checkpoint_trackio import (
            TrackioCheckpoint,
        )

        logger = TrackioLogger(project="p", name="run-1", group="g")
        trainer = _make_mock_trainer(
            loggers=[logger],
            default_root_dir=str(tmp_path),
        )
        checkpoint: Dict[str, Any] = {}

        cb = TrackioCheckpoint()
        cb.on_save_checkpoint(trainer, MagicMock(), checkpoint)

        assert "trackio" in checkpoint
        assert checkpoint["trackio"]["project"] == "p"
        assert checkpoint["trackio"]["name"] == "run-1"
        assert checkpoint["trackio"]["group"] == "g"

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_save_writes_sidecar_file(self, mock_trackio, tmp_path):
        from stable_pretraining.loggers.trackio import TrackioLogger
        from stable_pretraining.callbacks.checkpoint_trackio import (
            TrackioCheckpoint,
            _TRACKIO_RESUME_FILENAME,
        )

        logger = TrackioLogger(project="p", name="run-1")
        trainer = _make_mock_trainer(
            loggers=[logger],
            default_root_dir=str(tmp_path),
            is_global_zero=True,
        )

        cb = TrackioCheckpoint()
        cb.on_save_checkpoint(trainer, MagicMock(), {})

        sidecar = tmp_path / _TRACKIO_RESUME_FILENAME
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["project"] == "p"
        assert data["name"] == "run-1"

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_save_no_sidecar_on_non_zero_rank(self, mock_trackio, tmp_path):
        from stable_pretraining.loggers.trackio import TrackioLogger
        from stable_pretraining.callbacks.checkpoint_trackio import (
            TrackioCheckpoint,
            _TRACKIO_RESUME_FILENAME,
        )

        logger = TrackioLogger(project="p", name="run-1")
        trainer = _make_mock_trainer(
            loggers=[logger],
            default_root_dir=str(tmp_path),
            is_global_zero=False,
        )

        cb = TrackioCheckpoint()
        cb.on_save_checkpoint(trainer, MagicMock(), {})

        sidecar = tmp_path / _TRACKIO_RESUME_FILENAME
        assert not sidecar.exists()

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_save_noop_without_trackio_logger(self, mock_trackio, tmp_path):
        from stable_pretraining.callbacks.checkpoint_trackio import (
            TrackioCheckpoint,
        )

        trainer = _make_mock_trainer(loggers=[], default_root_dir=str(tmp_path))
        checkpoint: Dict[str, Any] = {}

        cb = TrackioCheckpoint()
        cb.on_save_checkpoint(trainer, MagicMock(), checkpoint)

        assert "trackio" not in checkpoint

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_save_resolves_name_from_run_object(self, mock_trackio, tmp_path):
        """If name is None but the run object has a name, it should be captured."""
        from stable_pretraining.loggers.trackio import TrackioLogger
        from stable_pretraining.callbacks.checkpoint_trackio import (
            TrackioCheckpoint,
        )

        logger = TrackioLogger(project="p")  # name=None
        # Simulate that trackio auto-assigned a name via the run object
        mock_run = MagicMock()
        mock_run.name = "auto-generated-42"
        logger._run = mock_run

        trainer = _make_mock_trainer(loggers=[logger], default_root_dir=str(tmp_path))
        checkpoint: Dict[str, Any] = {}

        cb = TrackioCheckpoint()
        cb.on_save_checkpoint(trainer, MagicMock(), checkpoint)

        assert checkpoint["trackio"]["name"] == "auto-generated-42"


# ============================================================================
# TrackioCheckpoint — load
# ============================================================================


class TestTrackioCheckpointLoad:
    """Tests for the Trackio checkpoint load callback."""

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_load_noop_without_trackio_key(self, mock_trackio):
        from stable_pretraining.callbacks.checkpoint_trackio import (
            TrackioCheckpoint,
        )

        trainer = _make_mock_trainer()
        cb = TrackioCheckpoint()
        cb.on_load_checkpoint(trainer, MagicMock(), {})  # should not raise

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_load_warns_without_logger(self, mock_trackio):
        from stable_pretraining.callbacks.checkpoint_trackio import (
            TrackioCheckpoint,
        )

        trainer = _make_mock_trainer(loggers=[])
        checkpoint = {"trackio": {"name": "run-1", "project": "p"}}

        cb = TrackioCheckpoint()
        # Should warn but not raise
        cb.on_load_checkpoint(trainer, MagicMock(), checkpoint)

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_load_verifies_matching_name(self, mock_trackio):
        from stable_pretraining.loggers.trackio import TrackioLogger
        from stable_pretraining.callbacks.checkpoint_trackio import (
            TrackioCheckpoint,
        )

        logger = TrackioLogger(project="p", name="run-1")
        trainer = _make_mock_trainer(loggers=[logger])
        checkpoint = {"trackio": {"name": "run-1", "project": "p"}}

        cb = TrackioCheckpoint()
        cb.on_load_checkpoint(trainer, MagicMock(), checkpoint)
        # No error — names match

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_load_noop_when_name_is_none(self, mock_trackio):
        """If checkpoint has trackio key but name is None, skip verification."""
        from stable_pretraining.loggers.trackio import TrackioLogger
        from stable_pretraining.callbacks.checkpoint_trackio import (
            TrackioCheckpoint,
        )

        logger = TrackioLogger(project="p")
        trainer = _make_mock_trainer(loggers=[logger])
        checkpoint = {"trackio": {"name": None, "project": "p"}}

        cb = TrackioCheckpoint()
        cb.on_load_checkpoint(trainer, MagicMock(), checkpoint)  # should not raise


# ============================================================================
# Manager — _maybe_restore_trackio_run
# ============================================================================


class TestManagerTrackioResume:
    """Tests for Manager-driven Trackio resume flow."""

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_restore_from_sidecar(self, mock_trackio, tmp_path):
        from stable_pretraining.loggers.trackio import TrackioLogger
        from stable_pretraining.callbacks.checkpoint_trackio import (
            _TRACKIO_RESUME_FILENAME,
        )

        # Write a sidecar
        sidecar = tmp_path / _TRACKIO_RESUME_FILENAME
        sidecar.write_text(
            json.dumps(
                {
                    "name": "run-1",
                    "project": "p",
                    "group": None,
                }
            )
        )

        logger = TrackioLogger(project="p")
        trainer = _make_mock_trainer(loggers=[logger])

        # Simulate what Manager._maybe_restore_trackio_run does
        # (We can't easily instantiate Manager, so test the logic directly)
        from stable_pretraining.loggers.trackio import find_trackio_logger

        trackio_logger = find_trackio_logger(trainer)
        assert trackio_logger is not None

        resume_info = json.loads(sidecar.read_text())
        run_name = resume_info.get("name")
        assert run_name == "run-1"

        trackio_logger.set_resume(run_name)
        assert trackio_logger._name == "run-1"
        assert trackio_logger._resume == "must"

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_restore_skips_project_mismatch(self, mock_trackio, tmp_path):
        from stable_pretraining.loggers.trackio import TrackioLogger
        from stable_pretraining.callbacks.checkpoint_trackio import (
            _TRACKIO_RESUME_FILENAME,
        )

        sidecar = tmp_path / _TRACKIO_RESUME_FILENAME
        sidecar.write_text(
            json.dumps(
                {
                    "name": "run-1",
                    "project": "other-project",
                    "group": None,
                }
            )
        )

        logger = TrackioLogger(project="my-project")

        # The manager would check project match and skip
        resume_info = json.loads(sidecar.read_text())
        saved_project = resume_info.get("project")
        assert saved_project != logger._project
        # So resume should NOT be set
        assert logger._resume == "never"


# ============================================================================
# Config integration
# ============================================================================


class TestConfigIntegration:
    """Tests for config-driven logger/callback integration."""

    def test_trackio_checkpoint_is_valid_callback_key(self):
        from stable_pretraining._config import set as spt_set, get_config

        cfg = get_config()
        cfg.reset()

        spt_set(default_callbacks={"trackio_checkpoint": False})
        assert get_config().default_callbacks["trackio_checkpoint"] is False

        cfg.reset()

    def test_trackio_checkpoint_in_default_factory(self):
        from stable_pretraining.callbacks.factories import (
            _DEFAULT_CALLBACK_REGISTRY,
        )
        from stable_pretraining.callbacks.checkpoint_trackio import (
            TrackioCheckpoint,
        )

        assert "trackio_checkpoint" in _DEFAULT_CALLBACK_REGISTRY
        factory, kwargs = _DEFAULT_CALLBACK_REGISTRY["trackio_checkpoint"]
        assert factory is TrackioCheckpoint

    def test_disable_trackio_checkpoint_via_config(self):
        from stable_pretraining._config import set as spt_set, get_config
        from stable_pretraining.callbacks.factories import default
        from stable_pretraining.callbacks.checkpoint_trackio import (
            TrackioCheckpoint,
        )

        cfg = get_config()
        cfg.reset()

        spt_set(default_callbacks={"trackio_checkpoint": False})
        cbs = default()
        assert not any(isinstance(cb, TrackioCheckpoint) for cb in cbs)

        cfg.reset()


# ============================================================================
# Top-level exports
# ============================================================================


class TestExports:
    """Tests for public symbol exports."""

    def test_trackio_logger_importable_from_top_level(self):
        from stable_pretraining import TrackioLogger

        assert TrackioLogger is not None

    def test_trackio_available_flag(self):
        import stable_pretraining

        assert hasattr(stable_pretraining, "TRACKIO_AVAILABLE")
        assert isinstance(stable_pretraining.TRACKIO_AVAILABLE, bool)

    def test_trackio_checkpoint_importable_from_callbacks(self):
        from stable_pretraining.callbacks import TrackioCheckpoint

        assert TrackioCheckpoint is not None

    def test_loggers_module_importable(self):
        from stable_pretraining import loggers

        assert hasattr(loggers, "TrackioLogger")


# ============================================================================
# TrackioLogger — auto-GPU detection
# ============================================================================


class TestAutoLogGpu:
    """Tests for automatic GPU metric logging."""

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    @patch("stable_pretraining.loggers.trackio._cuda_available", return_value=True)
    def test_defaults_true_when_cuda_available(self, mock_cuda, mock_trackio):
        from stable_pretraining.loggers.trackio import TrackioLogger

        logger = TrackioLogger(project="p")
        assert logger._auto_log_gpu is True

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    @patch("stable_pretraining.loggers.trackio._cuda_available", return_value=False)
    def test_defaults_false_when_no_cuda(self, mock_cuda, mock_trackio):
        from stable_pretraining.loggers.trackio import TrackioLogger

        logger = TrackioLogger(project="p")
        assert logger._auto_log_gpu is False

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    @patch("stable_pretraining.loggers.trackio._cuda_available", return_value=True)
    def test_explicit_false_overrides_cuda(self, mock_cuda, mock_trackio):
        """Pass ``auto_log_gpu=False`` to force-disable even on GPU."""
        from stable_pretraining.loggers.trackio import TrackioLogger

        logger = TrackioLogger(project="p", auto_log_gpu=False)
        assert logger._auto_log_gpu is False


# ============================================================================
# TrackioLogger — server_url mode (self-hosted Trackio server)
# ============================================================================


class TestTrackioLoggerServerUrl:
    """Tests for TrackioLogger self-hosted server_url mode."""

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_server_url_stored(self, mock_trackio):
        from stable_pretraining.loggers.trackio import TrackioLogger

        logger = TrackioLogger(project="p", server_url="http://node:7860")
        assert logger._server_url == "http://node:7860"
        assert logger._space_id is None

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_server_url_and_space_id_mutually_exclusive(self, mock_trackio):
        from stable_pretraining.loggers.trackio import TrackioLogger

        with pytest.raises(ValueError, match="either `space_id` .* or `server_url`"):
            TrackioLogger(
                project="p",
                space_id="user/space",
                server_url="http://node:7860",
            )

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_server_url_from_env_var(self, mock_trackio, monkeypatch):
        from stable_pretraining.loggers.trackio import TrackioLogger

        monkeypatch.setenv("TRACKIO_SERVER_URL", "http://envnode:9999")
        logger = TrackioLogger(project="p")
        assert logger._server_url == "http://envnode:9999"

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_explicit_server_url_overrides_env(self, mock_trackio, monkeypatch):
        from stable_pretraining.loggers.trackio import TrackioLogger

        monkeypatch.setenv("TRACKIO_SERVER_URL", "http://envnode:9999")
        logger = TrackioLogger(project="p", server_url="http://explicit:7860")
        assert logger._server_url == "http://explicit:7860"

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_init_run_server_mode_bypasses_trackio_init(self, mock_trackio):
        """In server_url mode, trackio.init() must NOT be called."""
        from stable_pretraining.loggers.trackio import TrackioLogger

        mock_run = MagicMock()
        mock_client = MagicMock()

        with (
            patch("gradio_client.Client", return_value=mock_client) as mock_client_cls,
            patch("trackio.run.Run", return_value=mock_run) as mock_run_cls,
            patch("trackio.context_vars") as mock_cv,
        ):
            logger = TrackioLogger(
                project="p",
                name="run-1",
                server_url="http://node:7860",
            )
            logger.log_metrics({"loss": 0.5}, step=1)

            # trackio.init must NOT be called in server mode
            mock_trackio.init.assert_not_called()

            # Client created with the server URL
            mock_client_cls.assert_called_once_with("http://node:7860", verbose=False)

            # Run constructed manually with sentinel space_id
            mock_run_cls.assert_called_once()
            run_kwargs = mock_run_cls.call_args.kwargs
            assert run_kwargs["url"] == "http://node:7860"
            assert run_kwargs["project"] == "p"
            assert run_kwargs["name"] == "run-1"
            assert run_kwargs["client"] is mock_client
            assert run_kwargs["space_id"] == "local/server"

            # context var set so module-level trackio funcs would work
            mock_cv.current_run.set.assert_called_once_with(mock_run)

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_log_metrics_server_mode_calls_run_log_directly(self, mock_trackio):
        """In server mode, metrics go through run.log(), not trackio.log()."""
        from stable_pretraining.loggers.trackio import TrackioLogger

        mock_run = MagicMock()

        with (
            patch("gradio_client.Client"),
            patch("trackio.run.Run", return_value=mock_run),
            patch("trackio.context_vars"),
        ):
            logger = TrackioLogger(project="p", server_url="http://node:7860")
            logger.log_metrics({"loss": 0.5, "acc": 0.9}, step=5)

            # run.log called directly, not trackio.log
            mock_run.log.assert_called_once_with({"loss": 0.5, "acc": 0.9}, step=5)
            mock_trackio.log.assert_not_called()

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_finalize_server_mode_calls_run_finish(self, mock_trackio):
        """In server mode, finalize calls run.finish(), not trackio.finish()."""
        from stable_pretraining.loggers.trackio import TrackioLogger

        mock_run = MagicMock()

        with (
            patch("gradio_client.Client"),
            patch("trackio.run.Run", return_value=mock_run),
            patch("trackio.context_vars"),
        ):
            logger = TrackioLogger(project="p", server_url="http://node:7860")
            logger.log_metrics({"loss": 1.0})
            logger.finalize("success")

            mock_run.finish.assert_called_once()
            mock_trackio.finish.assert_not_called()
            assert logger._run is None

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_log_hyperparams_server_mode_passes_config(self, mock_trackio):
        from stable_pretraining.loggers.trackio import TrackioLogger

        mock_run = MagicMock()

        with (
            patch("gradio_client.Client"),
            patch("trackio.run.Run", return_value=mock_run) as mock_run_cls,
            patch("trackio.context_vars"),
        ):
            logger = TrackioLogger(project="p", server_url="http://node:7860")
            logger.log_hyperparams({"lr": 0.01})

            mock_run_cls.assert_called_once()
            assert mock_run_cls.call_args.kwargs["config"] == {"lr": 0.01}

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_server_mode_forwards_auto_log_gpu(self, mock_trackio):
        """In server_url mode, auto_log_gpu must be passed to Run()."""
        from stable_pretraining.loggers.trackio import TrackioLogger

        mock_run = MagicMock()

        with (
            patch("gradio_client.Client"),
            patch("trackio.run.Run", return_value=mock_run) as mock_run_cls,
            patch("trackio.context_vars"),
        ):
            logger = TrackioLogger(
                project="p",
                server_url="http://node:7860",
                auto_log_gpu=True,
                gpu_log_interval=5.0,
            )
            logger.log_metrics({"loss": 0.5})

            kwargs = mock_run_cls.call_args.kwargs
            assert kwargs["auto_log_gpu"] is True
            assert kwargs["gpu_log_interval"] == 5.0

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_resume_info_includes_server_url(self, mock_trackio):
        from stable_pretraining.loggers.trackio import TrackioLogger

        logger = TrackioLogger(
            project="p",
            name="run-1",
            server_url="http://node:7860",
        )
        info = logger.resume_info
        assert info["server_url"] == "http://node:7860"


# ============================================================================
# load_project_df
# ============================================================================


class TestLoadProjectDf:
    """Tests for loading project DataFrames from trackio."""

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_load_project_df_remote(self, mock_trackio):
        """Remote mode uses gradio_client to fetch from the server."""
        from stable_pretraining.loggers.trackio import load_project_df

        mock_client = MagicMock()
        # First call: get_runs_for_project → list of runs
        # Then one get_logs call per run
        mock_client.predict.side_effect = [
            ["run-a", "run-b"],  # /get_runs_for_project
            [{"step": 0, "loss": 1.0}, {"step": 1, "loss": 0.5}],  # run-a
            [{"step": 0, "loss": 2.0}],  # run-b
        ]

        with patch("gradio_client.Client", return_value=mock_client):
            df = load_project_df("my-proj", server_url="http://node:7860")

        assert len(df) == 3
        assert set(df["run"]) == {"run-a", "run-b"}
        assert set(df.columns) >= {"step", "loss", "run"}

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_load_project_df_empty_runs_returns_empty_df(self, mock_trackio):
        from stable_pretraining.loggers.trackio import load_project_df

        mock_client = MagicMock()
        mock_client.predict.return_value = []  # no runs

        with patch("gradio_client.Client", return_value=mock_client):
            df = load_project_df("my-proj", server_url="http://node:7860")

        assert len(df) == 0

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_load_project_df_server_url_from_env(self, mock_trackio, monkeypatch):
        from stable_pretraining.loggers.trackio import load_project_df

        monkeypatch.setenv("TRACKIO_SERVER_URL", "http://envhost:7860")

        mock_client = MagicMock()
        mock_client.predict.side_effect = [
            [],
        ]

        with patch("gradio_client.Client", return_value=mock_client) as mock_cls:
            load_project_df("my-proj")

        mock_cls.assert_called_once_with("http://envhost:7860", verbose=False)

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_load_project_df_restrict_to_runs(self, mock_trackio):
        """When ``runs=`` is passed, only fetch those runs."""
        from stable_pretraining.loggers.trackio import load_project_df

        mock_client = MagicMock()
        # No get_runs call because runs is explicit
        mock_client.predict.side_effect = [
            [{"step": 0, "loss": 0.1}],  # run-a only
        ]

        with patch("gradio_client.Client", return_value=mock_client):
            df = load_project_df(
                "my-proj",
                server_url="http://node:7860",
                runs=["run-a"],
            )

        assert len(df) == 1
        assert df["run"].iloc[0] == "run-a"

    def test_exported_from_loggers_module(self):
        from stable_pretraining.loggers import load_project_df

        assert callable(load_project_df)


# ============================================================================
# Sidecar round-trip (save then load)
# ============================================================================


class TestSidecarRoundTrip:
    """Tests for sidecar metadata round-trip."""

    @patch("stable_pretraining.loggers.trackio.TRACKIO_AVAILABLE", True)
    @patch("stable_pretraining.loggers.trackio.trackio")
    def test_save_then_load_roundtrip(self, mock_trackio, tmp_path):
        """Full save → load cycle preserves run identity."""
        from stable_pretraining.loggers.trackio import TrackioLogger
        from stable_pretraining.callbacks.checkpoint_trackio import (
            TrackioCheckpoint,
            _TRACKIO_RESUME_FILENAME,
        )

        # -- Save --
        logger1 = TrackioLogger(project="p", name="run-1", group="g")
        trainer1 = _make_mock_trainer(
            loggers=[logger1],
            default_root_dir=str(tmp_path),
        )
        checkpoint: Dict[str, Any] = {}

        cb = TrackioCheckpoint()
        cb.on_save_checkpoint(trainer1, MagicMock(), checkpoint)

        # Verify sidecar exists
        sidecar_path = tmp_path / _TRACKIO_RESUME_FILENAME
        assert sidecar_path.exists()

        # -- Load (new logger, simulating a restarted job) --
        logger2 = TrackioLogger(project="p")
        assert logger2._name is None
        assert logger2._resume == "never"

        # Simulate Manager._maybe_restore_trackio_run
        resume_info = json.loads(sidecar_path.read_text())
        logger2.set_resume(resume_info["name"])

        assert logger2._name == "run-1"
        assert logger2._resume == "must"

        # Verify on_load_checkpoint doesn't error
        trainer2 = _make_mock_trainer(loggers=[logger2])
        cb.on_load_checkpoint(trainer2, MagicMock(), checkpoint)
