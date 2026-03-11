"""Tests for WandbCallback and WandbCheckpoint.

Tests verify: config flattening, offline file I/O,
run directory discovery, and logger state management.

Run with:
    pytest stable_pretraining/tests/unit/test_wandb_lifecycle.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

from omegaconf import OmegaConf

from stable_pretraining.callbacks.wandb_lifecycle import WandbCallback, WandbCheckpoint


def _make_trainer(logger=None):
    """Build a minimal Trainer-like object with the given logger."""
    trainer = MagicMock()
    trainer.logger = logger
    trainer.callbacks = []
    trainer.global_rank = 0
    return trainer


def _make_wandb_logger(version="run123", wandb_init=None):
    """Build a minimal WandbLogger-like object."""
    from lightning.pytorch.loggers import WandbLogger

    logger = MagicMock(spec=WandbLogger)
    logger.version = version
    logger._wandb_init = wandb_init or {}
    logger._experiment = MagicMock()
    logger._id = version
    # .experiment returns an object with .id and .offline
    exp = MagicMock()
    exp.id = version
    exp.offline = False
    type(logger).experiment = PropertyMock(return_value=exp)
    return logger


class TestWandbCallbackNoopWithoutWandB:
    """WandbCallback is a no-op when the trainer has no WandbLogger."""

    def test_noop_with_csv_logger(self):
        from lightning.pytorch.loggers import CSVLogger

        callback = WandbCallback()
        trainer = _make_trainer(logger=MagicMock(spec=CSVLogger))
        module = MagicMock()

        # Should not raise
        callback.setup(trainer, module, stage="fit")
        callback.teardown(trainer, module, stage="fit")

        # Internal state should be untouched
        assert callback._run_id is None
        assert not callback._config_synced

    def test_noop_with_none_logger(self):
        callback = WandbCallback()
        trainer = _make_trainer(logger=None)
        module = MagicMock()

        callback.setup(trainer, module, stage="fit")
        callback.teardown(trainer, module, stage="fit")

        assert callback._run_id is None


class TestConfigFlattening:
    """_sync_config_online flattens nested OmegaConf into dot-separated keys."""

    @patch("stable_pretraining.callbacks.wandb_lifecycle.wandb")
    def test_nested_dicts_flattened(self, mock_wandb):
        mock_wandb.run = MagicMock()
        mock_wandb.config = MagicMock()
        mock_wandb.config.keys.return_value = []  # empty config triggers upload

        captured = {}

        def capture_update(d):
            captured.update(d)

        mock_wandb.config.update = capture_update

        hydra_cfg = {
            "trainer": OmegaConf.create(
                {
                    "max_epochs": 100,
                    "accelerator": "gpu",
                }
            ),
            "module": OmegaConf.create(
                {
                    "backbone": {"type": "resnet", "depth": 18},
                    "loss": {"name": "cross_entropy"},
                }
            ),
        }
        callback = WandbCallback(hydra_config=hydra_cfg)
        callback._sync_config_online()

        # Every value should be a scalar (no nested dicts or lists)
        for k, v in captured.items():
            assert not isinstance(v, (dict, list)), (
                f"Key '{k}' has non-scalar value: {v}"
            )
            assert "." in k or k in ("trainer.max_epochs", "trainer.accelerator"), (
                f"Key '{k}' should be dot-separated"
            )

        assert captured["trainer.max_epochs"] == 100
        assert captured["module.backbone.type"] == "resnet"
        assert captured["module.backbone.depth"] == 18

    @patch("stable_pretraining.callbacks.wandb_lifecycle.wandb")
    def test_lists_expanded_to_indexed_keys(self, mock_wandb):
        mock_wandb.run = MagicMock()
        mock_wandb.config = MagicMock()
        mock_wandb.config.keys.return_value = []

        captured = {}
        mock_wandb.config.update = lambda d: captured.update(d)

        hydra_cfg = {
            "module": OmegaConf.create(
                {
                    "layers": [64, 128, 256],
                    "name": "test",
                }
            ),
        }
        callback = WandbCallback(hydra_config=hydra_cfg)
        callback._sync_config_online()

        assert captured["module.layers.0"] == 64
        assert captured["module.layers.1"] == 128
        assert captured["module.layers.2"] == 256
        assert captured["module.name"] == "test"

    @patch("stable_pretraining.callbacks.wandb_lifecycle.wandb")
    def test_no_config_skips_upload(self, mock_wandb):
        mock_wandb.run = MagicMock()
        mock_wandb.config = MagicMock()
        mock_wandb.config.keys.return_value = []
        mock_wandb.config.update = MagicMock()

        callback = WandbCallback(hydra_config=None)
        callback._sync_config_online()

        mock_wandb.config.update.assert_not_called()


class TestOfflineDataDump:
    """_dump_wandb_data writes real JSON files for offline runs."""

    @patch("stable_pretraining.callbacks.wandb_lifecycle.wandb")
    def test_writes_summary_and_config(self, mock_wandb):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_wandb.run = MagicMock()
            mock_wandb.run.offline = True
            mock_wandb.run.dir = tmpdir
            mock_wandb.run.summary._as_dict.return_value = {
                "train/loss": 0.42,
                "epoch": 10,
            }
            mock_wandb.run.config.as_dict.return_value = {
                "lr": 0.001,
                "model": "resnet18",
            }

            WandbCallback._dump_wandb_data()

            summary_path = Path(tmpdir) / "wandb-summary.json"
            config_path = Path(tmpdir) / "wandb-config.json"

            assert summary_path.is_file()
            assert config_path.is_file()

            with open(summary_path) as f:
                summary = json.load(f)
            assert summary["train/loss"] == 0.42
            assert summary["epoch"] == 10

            with open(config_path) as f:
                config = json.load(f)
            assert config["lr"] == 0.001
            assert config["model"] == "resnet18"

    @patch("stable_pretraining.callbacks.wandb_lifecycle.wandb")
    def test_idempotent_skips_existing(self, mock_wandb):
        """Calling _dump_wandb_data twice doesn't overwrite or raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_wandb.run = MagicMock()
            mock_wandb.run.offline = True
            mock_wandb.run.dir = tmpdir
            mock_wandb.run.summary._as_dict.return_value = {"loss": 1.0}
            mock_wandb.run.config.as_dict.return_value = {"lr": 0.01}

            WandbCallback._dump_wandb_data()

            # Read first write
            with open(Path(tmpdir) / "wandb-summary.json") as f:
                first_summary = json.load(f)

            # Change the mock return — if dump overwrites, content would differ
            mock_wandb.run.summary._as_dict.return_value = {"loss": 999.0}

            WandbCallback._dump_wandb_data()  # should be a no-op

            with open(Path(tmpdir) / "wandb-summary.json") as f:
                second_summary = json.load(f)

            assert first_summary == second_summary

    @patch("stable_pretraining.callbacks.wandb_lifecycle.wandb")
    def test_skips_online_runs(self, mock_wandb):
        """No files written for online runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_wandb.run = MagicMock()
            mock_wandb.run.offline = False
            mock_wandb.run.dir = tmpdir

            WandbCallback._dump_wandb_data()

            assert not (Path(tmpdir) / "wandb-summary.json").exists()
            assert not (Path(tmpdir) / "wandb-config.json").exists()


class TestFindPreviousRunDir:
    """_find_previous_run_dir uses real directory structure."""

    @patch("stable_pretraining.callbacks.wandb_lifecycle.wandb")
    def test_finds_previous_dir(self, mock_wandb):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = "abc123"
            dir1 = Path(tmpdir) / f"offline-run-20250101-{run_id}"
            dir2 = Path(tmpdir) / f"offline-run-20250102-{run_id}"
            dir1.mkdir()
            dir2.mkdir()

            mock_wandb.run = MagicMock()
            mock_wandb.run.id = run_id
            # wandb.run.dir points to <run_dir>/files
            mock_wandb.run.dir = str(dir2 / "files")

            result = WandbCallback._find_previous_run_dir()
            assert result == dir1

    @patch("stable_pretraining.callbacks.wandb_lifecycle.wandb")
    def test_returns_none_for_first_run(self, mock_wandb):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = "abc123"
            dir1 = Path(tmpdir) / f"offline-run-20250101-{run_id}"
            dir1.mkdir()

            mock_wandb.run = MagicMock()
            mock_wandb.run.id = run_id
            mock_wandb.run.dir = str(dir1 / "files")

            result = WandbCallback._find_previous_run_dir()
            assert result is None

    @patch("stable_pretraining.callbacks.wandb_lifecycle.wandb")
    def test_ignores_unrelated_dirs(self, mock_wandb):
        """Only dirs matching the current run ID are considered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = "abc123"
            other_id = "xyz789"
            dir_mine = Path(tmpdir) / f"offline-run-20250101-{run_id}"
            dir_other = Path(tmpdir) / f"offline-run-20250102-{other_id}"
            dir_mine.mkdir()
            dir_other.mkdir()

            mock_wandb.run = MagicMock()
            mock_wandb.run.id = run_id
            mock_wandb.run.dir = str(dir_mine / "files")

            result = WandbCallback._find_previous_run_dir()
            assert result is None  # only one dir for this run ID


class TestSetupCapturesRunId:
    """setup() captures the wandb run ID from the experiment."""

    @patch("stable_pretraining.callbacks.wandb_lifecycle.WANDB_AVAILABLE", True)
    @patch("stable_pretraining.callbacks.wandb_lifecycle.wandb")
    def test_run_id_captured(self, mock_wandb):
        mock_wandb.run = MagicMock()
        mock_wandb.config = MagicMock()
        mock_wandb.config.keys.return_value = ["existing_key"]

        logger = _make_wandb_logger(version="test456")
        trainer = _make_trainer(logger=logger)
        module = MagicMock()

        callback = WandbCallback()
        callback.setup(trainer, module, stage="fit")

        assert callback._run_id == "test456"


class TestTeardownResetsLogger:
    """teardown() calls wandb.finish() and resets the logger."""

    @patch("stable_pretraining.callbacks.wandb_lifecycle.WANDB_AVAILABLE", True)
    @patch("stable_pretraining.callbacks.wandb_lifecycle.wandb")
    def test_logger_experiment_reset(self, mock_wandb):
        mock_wandb.run = MagicMock()
        mock_wandb.run.offline = False

        logger = _make_wandb_logger(version="run789")
        trainer = _make_trainer(logger=logger)
        module = MagicMock()

        callback = WandbCallback()
        callback.teardown(trainer, module, stage="fit")

        mock_wandb.finish.assert_called_once()
        assert trainer.logger._experiment is None

    @patch("stable_pretraining.callbacks.wandb_lifecycle.WANDB_AVAILABLE", True)
    @patch("stable_pretraining.callbacks.wandb_lifecycle.wandb")
    def test_setup_after_teardown_sets_resume(self, mock_wandb):
        """After teardown, next setup() sets resume='allow' with saved run ID."""
        mock_wandb.run = MagicMock()
        mock_wandb.run.offline = False
        mock_wandb.config = MagicMock()
        mock_wandb.config.keys.return_value = ["key"]

        logger = _make_wandb_logger(version="run_abc")
        logger._wandb_init = {"project": "test"}
        trainer = _make_trainer(logger=logger)
        module = MagicMock()

        callback = WandbCallback()

        # First setup captures run ID
        callback.setup(trainer, module, stage="fit")
        assert callback._run_id == "run_abc"

        # Teardown resets logger
        callback.teardown(trainer, module, stage="fit")
        assert trainer.logger._experiment is None

        # Second setup should set resume="allow" and the saved run ID
        callback.setup(trainer, module, stage="validate")
        assert trainer.logger._wandb_init["id"] == "run_abc"
        assert trainer.logger._wandb_init["resume"] == "allow"


class TestWandbCheckpointSave:
    """WandbCheckpoint saves the run ID into the checkpoint dict."""

    def test_saves_run_id(self):
        logger = _make_wandb_logger(version="run_save_test")
        trainer = _make_trainer(logger=logger)
        module = MagicMock()
        checkpoint = {}

        cb = WandbCheckpoint()
        cb.on_save_checkpoint(trainer, module, checkpoint)

        assert "wandb" in checkpoint
        assert checkpoint["wandb"]["id"] == "run_save_test"

    def test_noop_without_wandb_logger(self):
        from lightning.pytorch.loggers import CSVLogger

        trainer = _make_trainer(logger=MagicMock(spec=CSVLogger))
        module = MagicMock()
        checkpoint = {}

        cb = WandbCheckpoint()
        cb.on_save_checkpoint(trainer, module, checkpoint)

        assert "wandb" not in checkpoint

    def test_noop_with_none_logger(self):
        trainer = _make_trainer(logger=None)
        module = MagicMock()
        checkpoint = {}

        cb = WandbCheckpoint()
        cb.on_save_checkpoint(trainer, module, checkpoint)

        assert "wandb" not in checkpoint
