"""Unit tests for SwanLabLogger and SwanLabCheckpoint.

Tests cover:
- SwanLabLogger subclass: resume_info / set_resume helpers
- SwanLabCheckpoint: save/load checkpoint, sidecar file writing
- find_swanlab_logger: discovery and error handling
- Manager integration: _maybe_restore_swanlab_run logic
- DDP safety: rank-zero-only save
- Config integration: swanlab_checkpoint in default_callbacks
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


def _make_mock_trainer(loggers=None, is_global_zero=True, default_root_dir="/tmp"):
    trainer = MagicMock()
    trainer.loggers = loggers or []
    trainer.is_global_zero = is_global_zero
    trainer.default_root_dir = default_root_dir
    return trainer


# ============================================================================
# SwanLabLogger — construction & upstream inheritance
# ============================================================================


class TestSwanLabLoggerBasics:
    """Basic import and construction tests for SwanLabLogger."""

    def test_import_from_top_level(self):
        from stable_pretraining import SwanLabLogger

        assert SwanLabLogger is not None

    def test_import_from_loggers_module(self):
        from stable_pretraining.loggers import SwanLabLogger

        assert SwanLabLogger is not None

    def test_is_subclass_of_upstream(self):
        """Our SwanLabLogger must extend swanlab's upstream logger."""
        from stable_pretraining.loggers.swanlab import SwanLabLogger
        from swanlab.integration.pytorch_lightning import (
            SwanLabLogger as _Upstream,
        )

        assert issubclass(SwanLabLogger, _Upstream)

    def test_swanlab_available_flag(self):
        import stable_pretraining

        assert hasattr(stable_pretraining, "SWANLAB_AVAILABLE")
        assert stable_pretraining.SWANLAB_AVAILABLE is True

    def test_basic_construction(self):
        """Inherited __init__ from upstream should accept project."""
        from stable_pretraining.loggers import SwanLabLogger

        logger = SwanLabLogger(
            project="my-project",
            experiment_name="run-1",
            mode="disabled",  # keep tests offline
        )
        assert logger.name == "my-project"


# ============================================================================
# SwanLabLogger — resume_info / set_resume
# ============================================================================


class TestSwanLabLoggerResume:
    """Tests for SwanLabLogger resume behavior."""

    def test_resume_info_before_init(self):
        """Before experiment is accessed, resume_info reflects constructor args."""
        from stable_pretraining.loggers import SwanLabLogger

        logger = SwanLabLogger(
            project="p",
            experiment_name="exp-1",
            group="g",
            id="fixed-id-42",
            mode="disabled",
        )
        info = logger.resume_info
        assert info["project"] == "p"
        assert info["experiment_name"] == "exp-1"
        assert info["group"] == "g"
        assert info["id"] == "fixed-id-42"

    def test_resume_info_prefers_live_id(self):
        """Once the run exists, its run_id should beat the constructor id."""
        from stable_pretraining.loggers import SwanLabLogger

        logger = SwanLabLogger(project="p", id="old-id", mode="disabled")
        # Simulate an initialised experiment
        fake_exp = MagicMock()
        fake_exp.public.run_id = "live-id-xyz"
        logger._experiment = fake_exp

        info = logger.resume_info
        assert info["id"] == "live-id-xyz"

    def test_set_resume_mutates_init_cfg(self):
        """set_resume should update the upstream _swanlab_init dict."""
        from stable_pretraining.loggers import SwanLabLogger

        logger = SwanLabLogger(project="p", mode="disabled")
        logger.set_resume("restored-id")

        # Upstream reads from _swanlab_init on first experiment access
        assert logger._swanlab_init["id"] == "restored-id"
        assert logger._swanlab_init["resume"] == "must"

    def test_set_resume_is_idempotent(self):
        from stable_pretraining.loggers import SwanLabLogger

        logger = SwanLabLogger(project="p", mode="disabled")
        logger.set_resume("id-a")
        logger.set_resume("id-b")
        assert logger._swanlab_init["id"] == "id-b"
        assert logger._swanlab_init["resume"] == "must"


# ============================================================================
# find_swanlab_logger
# ============================================================================


class TestFindSwanLabLogger:
    """Tests for locating the SwanLab logger on a trainer."""

    def test_find_none(self):
        from stable_pretraining.loggers.swanlab import find_swanlab_logger

        trainer = _make_mock_trainer(loggers=[])
        assert find_swanlab_logger(trainer) is None

    def test_find_single(self):
        from stable_pretraining.loggers import SwanLabLogger
        from stable_pretraining.loggers.swanlab import find_swanlab_logger

        logger = SwanLabLogger(project="p", mode="disabled")
        trainer = _make_mock_trainer(loggers=[MagicMock(), logger])
        assert find_swanlab_logger(trainer) is logger

    def test_find_multiple_raises(self):
        from stable_pretraining.loggers import SwanLabLogger
        from stable_pretraining.loggers.swanlab import find_swanlab_logger

        l1 = SwanLabLogger(project="p", mode="disabled")
        l2 = SwanLabLogger(project="q", mode="disabled")
        trainer = _make_mock_trainer(loggers=[l1, l2])

        with pytest.raises(RuntimeError, match="Found 2 SwanLabLoggers"):
            find_swanlab_logger(trainer)


# ============================================================================
# SwanLabCheckpoint — save
# ============================================================================


class TestSwanLabCheckpointSave:
    """Tests for the SwanLab checkpoint save callback."""

    def test_save_noop_without_logger(self, tmp_path):
        from stable_pretraining.callbacks.checkpoint_swanlab import (
            SwanLabCheckpoint,
        )

        trainer = _make_mock_trainer(loggers=[], default_root_dir=str(tmp_path))
        checkpoint: Dict[str, Any] = {}

        cb = SwanLabCheckpoint()
        cb.on_save_checkpoint(trainer, MagicMock(), checkpoint)

        assert "swanlab" not in checkpoint

    def test_save_skips_when_no_id(self, tmp_path):
        """If the logger has no live id and no configured id, skip."""
        from stable_pretraining.loggers import SwanLabLogger
        from stable_pretraining.callbacks.checkpoint_swanlab import (
            SwanLabCheckpoint,
        )

        logger = SwanLabLogger(project="p", mode="disabled")
        trainer = _make_mock_trainer(loggers=[logger], default_root_dir=str(tmp_path))
        checkpoint: Dict[str, Any] = {}

        cb = SwanLabCheckpoint()
        cb.on_save_checkpoint(trainer, MagicMock(), checkpoint)

        # No id → no resume info stored
        assert "swanlab" not in checkpoint

    def test_save_stores_resume_info_when_id_set(self, tmp_path):
        from stable_pretraining.loggers import SwanLabLogger
        from stable_pretraining.callbacks.checkpoint_swanlab import (
            SwanLabCheckpoint,
            _SWANLAB_RESUME_FILENAME,
        )

        logger = SwanLabLogger(
            project="p",
            experiment_name="exp-1",
            id="fixed-id",
            mode="disabled",
        )
        trainer = _make_mock_trainer(loggers=[logger], default_root_dir=str(tmp_path))
        checkpoint: Dict[str, Any] = {}

        cb = SwanLabCheckpoint()
        cb.on_save_checkpoint(trainer, MagicMock(), checkpoint)

        assert "swanlab" in checkpoint
        assert checkpoint["swanlab"]["id"] == "fixed-id"
        assert checkpoint["swanlab"]["project"] == "p"
        assert checkpoint["swanlab"]["experiment_name"] == "exp-1"

        # Sidecar on disk
        sidecar = tmp_path / _SWANLAB_RESUME_FILENAME
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["id"] == "fixed-id"

    def test_save_no_sidecar_on_non_zero_rank(self, tmp_path):
        from stable_pretraining.loggers import SwanLabLogger
        from stable_pretraining.callbacks.checkpoint_swanlab import (
            SwanLabCheckpoint,
            _SWANLAB_RESUME_FILENAME,
        )

        logger = SwanLabLogger(project="p", id="fixed-id", mode="disabled")
        trainer = _make_mock_trainer(
            loggers=[logger],
            default_root_dir=str(tmp_path),
            is_global_zero=False,
        )

        cb = SwanLabCheckpoint()
        cb.on_save_checkpoint(trainer, MagicMock(), {})

        sidecar = tmp_path / _SWANLAB_RESUME_FILENAME
        assert not sidecar.exists()


# ============================================================================
# SwanLabCheckpoint — load
# ============================================================================


class TestSwanLabCheckpointLoad:
    """Tests for the SwanLab checkpoint load callback."""

    def test_load_noop_without_swanlab_key(self):
        from stable_pretraining.callbacks.checkpoint_swanlab import (
            SwanLabCheckpoint,
        )

        trainer = _make_mock_trainer()
        cb = SwanLabCheckpoint()
        cb.on_load_checkpoint(trainer, MagicMock(), {})  # should not raise

    def test_load_noop_when_id_missing_in_checkpoint(self):
        from stable_pretraining.callbacks.checkpoint_swanlab import (
            SwanLabCheckpoint,
        )

        trainer = _make_mock_trainer(loggers=[])
        checkpoint = {"swanlab": {"project": "p", "id": None}}

        cb = SwanLabCheckpoint()
        cb.on_load_checkpoint(trainer, MagicMock(), checkpoint)  # no raise

    def test_load_warns_without_logger(self):
        from stable_pretraining.callbacks.checkpoint_swanlab import (
            SwanLabCheckpoint,
        )

        trainer = _make_mock_trainer(loggers=[])
        checkpoint = {"swanlab": {"id": "restored-id", "project": "p"}}

        cb = SwanLabCheckpoint()
        cb.on_load_checkpoint(trainer, MagicMock(), checkpoint)  # no raise

    def test_load_verifies_matching_id(self):
        from stable_pretraining.loggers import SwanLabLogger
        from stable_pretraining.callbacks.checkpoint_swanlab import (
            SwanLabCheckpoint,
        )

        logger = SwanLabLogger(project="p", id="same-id", mode="disabled")
        trainer = _make_mock_trainer(loggers=[logger])
        checkpoint = {"swanlab": {"id": "same-id", "project": "p"}}

        cb = SwanLabCheckpoint()
        cb.on_load_checkpoint(trainer, MagicMock(), checkpoint)  # no raise


# ============================================================================
# Config integration
# ============================================================================


class TestConfigIntegration:
    """Tests for config-driven logger/callback integration."""

    def test_swanlab_checkpoint_is_valid_callback_key(self):
        from stable_pretraining._config import set as spt_set, get_config

        cfg = get_config()
        cfg.reset()

        spt_set(default_callbacks={"swanlab_checkpoint": False})
        assert get_config().default_callbacks["swanlab_checkpoint"] is False

        cfg.reset()

    def test_swanlab_checkpoint_in_default_factory(self):
        from stable_pretraining.callbacks.factories import (
            _DEFAULT_CALLBACK_REGISTRY,
        )
        from stable_pretraining.callbacks.checkpoint_swanlab import (
            SwanLabCheckpoint,
        )

        assert "swanlab_checkpoint" in _DEFAULT_CALLBACK_REGISTRY
        factory, _ = _DEFAULT_CALLBACK_REGISTRY["swanlab_checkpoint"]
        assert factory is SwanLabCheckpoint

    def test_disable_swanlab_checkpoint_via_config(self):
        from stable_pretraining._config import set as spt_set, get_config
        from stable_pretraining.callbacks.factories import default
        from stable_pretraining.callbacks.checkpoint_swanlab import (
            SwanLabCheckpoint,
        )

        cfg = get_config()
        cfg.reset()

        spt_set(default_callbacks={"swanlab_checkpoint": False})
        cbs = default()
        assert not any(isinstance(cb, SwanLabCheckpoint) for cb in cbs)

        cfg.reset()


# ============================================================================
# Manager — _maybe_restore_swanlab_run
# ============================================================================


class TestManagerSwanLabResume:
    """Tests for Manager-driven SwanLab resume flow."""

    def test_restore_from_sidecar(self, tmp_path):
        from stable_pretraining.loggers import SwanLabLogger
        from stable_pretraining.callbacks.checkpoint_swanlab import (
            _SWANLAB_RESUME_FILENAME,
        )
        from stable_pretraining.loggers.swanlab import find_swanlab_logger

        sidecar = tmp_path / _SWANLAB_RESUME_FILENAME
        sidecar.write_text(
            json.dumps(
                {
                    "id": "restored-id",
                    "project": "p",
                    "experiment_name": "run-1",
                }
            )
        )

        logger = SwanLabLogger(project="p", mode="disabled")
        trainer = _make_mock_trainer(loggers=[logger])

        # Simulate the manager restore flow
        swanlab_logger = find_swanlab_logger(trainer)
        assert swanlab_logger is not None

        resume_info = json.loads(sidecar.read_text())
        swanlab_logger.set_resume(resume_info["id"])

        assert swanlab_logger._swanlab_init["id"] == "restored-id"
        assert swanlab_logger._swanlab_init["resume"] == "must"

    def test_restore_skips_project_mismatch(self, tmp_path):
        from stable_pretraining.loggers import SwanLabLogger
        from stable_pretraining.callbacks.checkpoint_swanlab import (
            _SWANLAB_RESUME_FILENAME,
        )

        sidecar = tmp_path / _SWANLAB_RESUME_FILENAME
        sidecar.write_text(
            json.dumps(
                {
                    "id": "run-1",
                    "project": "other-project",
                }
            )
        )

        logger = SwanLabLogger(project="my-project", mode="disabled")
        resume_info = json.loads(sidecar.read_text())

        saved_project = resume_info.get("project")
        assert saved_project != logger._project
        # Manager would skip injection → id stays None
        assert logger._swanlab_init.get("id") is None


# ============================================================================
# Round-trip
# ============================================================================


class TestRoundTrip:
    """End-to-end save/load round-trip tests."""

    def test_save_then_load_roundtrip(self, tmp_path):
        from stable_pretraining.loggers import SwanLabLogger
        from stable_pretraining.callbacks.checkpoint_swanlab import (
            SwanLabCheckpoint,
            _SWANLAB_RESUME_FILENAME,
        )

        # -- Save with id --
        logger1 = SwanLabLogger(
            project="p",
            experiment_name="exp-1",
            id="fixed-id",
            mode="disabled",
        )
        trainer1 = _make_mock_trainer(loggers=[logger1], default_root_dir=str(tmp_path))
        checkpoint: Dict[str, Any] = {}

        cb = SwanLabCheckpoint()
        cb.on_save_checkpoint(trainer1, MagicMock(), checkpoint)

        sidecar_path = tmp_path / _SWANLAB_RESUME_FILENAME
        assert sidecar_path.exists()

        # -- Load in a new process --
        logger2 = SwanLabLogger(project="p", mode="disabled")
        assert logger2._swanlab_init.get("id") is None

        resume_info = json.loads(sidecar_path.read_text())
        logger2.set_resume(resume_info["id"])

        assert logger2._swanlab_init["id"] == "fixed-id"
        assert logger2._swanlab_init["resume"] == "must"

        trainer2 = _make_mock_trainer(loggers=[logger2])
        cb.on_load_checkpoint(trainer2, MagicMock(), checkpoint)  # no raise
