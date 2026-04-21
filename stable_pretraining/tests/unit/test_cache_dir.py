"""Unit tests for the cache_dir / run directory feature."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from stable_pretraining._config import get_config, set as spt_set
from stable_pretraining.manager import (
    Manager,
    _generate_run_id,
    _RunDirCallback,
    _RUN_META_FILENAME,
)
from stable_pretraining.tests.utils import BoringTrainer, BoringModule, BoringDataModule

pytestmark = pytest.mark.unit


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_config():
    """Reset global config before and after every test."""
    cfg = get_config()
    cfg.reset()
    yield
    cfg.reset()


@pytest.fixture()
def cache_dir(tmp_path):
    """Provide a temporary cache_dir and configure it globally."""
    d = tmp_path / "spt_cache"
    spt_set(cache_dir=str(d))
    return d


# ============================================================================
# _config.py — cache_dir property
# ============================================================================


class TestCacheDirConfig:
    """Tests for cache_dir config defaults and env overrides."""

    def test_default_is_set(self, monkeypatch):
        monkeypatch.delenv("SPT_CACHE_DIR", raising=False)
        get_config().reset()
        assert get_config().cache_dir is not None
        assert "stable-pretraining" in get_config().cache_dir

    def test_set_via_spt_set(self, tmp_path):
        spt_set(cache_dir=str(tmp_path))
        assert get_config().cache_dir == str(tmp_path)

    def test_set_via_property(self, tmp_path):
        cfg = get_config()
        cfg.cache_dir = str(tmp_path)
        assert cfg.cache_dir == str(tmp_path)

    def test_set_to_none(self, tmp_path):
        spt_set(cache_dir=str(tmp_path))
        cfg = get_config()
        cfg.cache_dir = None
        assert cfg.cache_dir is None

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="must not be empty"):
            spt_set(cache_dir="")

    def test_rejects_whitespace_only(self):
        with pytest.raises(ValueError, match="must not be empty"):
            spt_set(cache_dir="   ")

    def test_rejects_non_string(self):
        with pytest.raises(TypeError, match="must be a str"):
            get_config().cache_dir = 123

    def test_reset_restores_default_cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SPT_CACHE_DIR", raising=False)
        spt_set(cache_dir=str(tmp_path))
        get_config().reset()
        # reset() restores the default (~/.cache/stable-pretraining), not None
        assert get_config().cache_dir is not None
        assert "stable-pretraining" in get_config().cache_dir

    def test_repr_includes_cache_dir(self, tmp_path):
        spt_set(cache_dir=str(tmp_path))
        assert "cache_dir=" in repr(get_config())

    def test_env_var_sets_default(self, monkeypatch, tmp_path):
        """SPT_CACHE_DIR env var should be picked up on init."""
        monkeypatch.setenv("SPT_CACHE_DIR", str(tmp_path))
        cfg = get_config()
        cfg.reset()  # re-reads env var in _init_defaults
        assert cfg.cache_dir == str(tmp_path)

    def test_tilde_expansion(self):
        """cache_dir with ~ should be stored as-is (expanded in _resolve_run_dir)."""
        spt_set(cache_dir="~/spt_cache")
        assert get_config().cache_dir == "~/spt_cache"

    def test_set_no_args_does_not_affect_cache_dir(self, tmp_path):
        spt_set(cache_dir=str(tmp_path))
        spt_set()  # no-op
        assert get_config().cache_dir == str(tmp_path)


# ============================================================================
# _config.py — requeue_checkpoint property
# ============================================================================


class TestRequeueCheckpointConfig:
    """Tests for requeue checkpoint config resolution."""

    def test_default_is_true(self):
        assert get_config().requeue_checkpoint is True

    def test_set_via_spt_set(self):
        spt_set(requeue_checkpoint=False)
        assert get_config().requeue_checkpoint is False

    def test_set_via_property(self):
        cfg = get_config()
        cfg.requeue_checkpoint = False
        assert cfg.requeue_checkpoint is False

    def test_set_back_to_true(self):
        spt_set(requeue_checkpoint=False)
        spt_set(requeue_checkpoint=True)
        assert get_config().requeue_checkpoint is True

    def test_rejects_non_bool(self):
        with pytest.raises(TypeError, match="must be a bool"):
            get_config().requeue_checkpoint = "yes"

    def test_rejects_int(self):
        with pytest.raises(TypeError, match="must be a bool"):
            spt_set(requeue_checkpoint=1)

    def test_reset_restores_default(self):
        spt_set(requeue_checkpoint=False)
        get_config().reset()
        assert get_config().requeue_checkpoint is True

    def test_repr_includes_requeue_checkpoint(self):
        assert "requeue_checkpoint=" in repr(get_config())

    def test_set_no_args_does_not_affect(self):
        spt_set(requeue_checkpoint=False)
        spt_set()  # no-op
        assert get_config().requeue_checkpoint is False


# ============================================================================
# _generate_run_id
# ============================================================================


class TestGenerateRunId:
    """Tests for run ID generation."""

    def test_uuid_fallback(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
        run_id = _generate_run_id()
        assert len(run_id) == 12
        assert run_id.isalnum()

    def test_two_uuids_are_different(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
        assert _generate_run_id() != _generate_run_id()

    def test_slurm_job_id(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        assert _generate_run_id() == "12345"

    def test_torchelastic_run_id(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", "abc123")
        assert _generate_run_id() == "abc123"

    def test_slurm_takes_priority(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "99999")
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", "abc123")
        assert _generate_run_id() == "99999"

    def test_slurm_deterministic_across_calls(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "77777")
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        assert _generate_run_id() == _generate_run_id() == "77777"

    def test_slurm_array_job(self, monkeypatch):
        """Array tasks within the same job should get different run_ids."""
        monkeypatch.setenv("SLURM_JOB_ID", "100")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
        assert _generate_run_id() == "100_3"

    def test_slurm_array_tasks_differ(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "100")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
        id0 = _generate_run_id()
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
        id1 = _generate_run_id()
        assert id0 != id1


# ============================================================================
# Manager._resolve_run_dir
# ============================================================================


class TestResolveRunDir:
    """Tests for run_dir resolution logic."""

    def _make_manager(self, ckpt_path=None):
        return Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
            ckpt_path=str(ckpt_path) if ckpt_path else None,
        )

    def test_returns_none_when_cache_dir_none(self):
        get_config()._cache_dir = None
        manager = self._make_manager()
        assert manager._resolve_run_dir() is None

    def test_creates_run_dir_under_cache_dir(self, cache_dir):
        manager = self._make_manager()
        run_dir = manager._resolve_run_dir()
        assert run_dir is not None
        assert str(run_dir).startswith(str(cache_dir))
        assert run_dir.is_dir()

    def test_run_dir_has_date_time_id_structure(self, cache_dir, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
        manager = self._make_manager()
        run_dir = manager._resolve_run_dir()
        parts = run_dir.relative_to(cache_dir).parts
        assert parts[0] == "runs"
        assert len(parts[1]) == 8 and parts[1].isdigit()  # YYYYMMDD
        assert len(parts[2]) == 6 and parts[2].isdigit()  # HHMMSS
        assert len(parts[3]) == 12  # uuid hex

    def test_writes_run_meta_json(self, cache_dir):
        manager = self._make_manager()
        run_dir = manager._resolve_run_dir()
        meta_path = run_dir / _RUN_META_FILENAME
        assert meta_path.is_file()
        meta = json.loads(meta_path.read_text())
        assert meta["run_dir"] == str(run_dir)
        assert "run_id" in meta

    def test_restores_from_sidecar(self, cache_dir):
        """If ckpt_path has a run_meta.json sibling, restore that run_dir."""
        prev_run_dir = cache_dir / "runs" / "20260101" / "120000" / "previd123456"
        prev_run_dir.mkdir(parents=True)
        (prev_run_dir / _RUN_META_FILENAME).write_text(
            json.dumps({"run_dir": str(prev_run_dir), "run_id": "previd123456"})
        )
        ckpt_dir = prev_run_dir / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "last.ckpt"
        ckpt_path.touch()
        (ckpt_dir / _RUN_META_FILENAME).write_text(
            json.dumps({"run_dir": str(prev_run_dir), "run_id": "previd123456"})
        )

        manager = self._make_manager(ckpt_path=ckpt_path)
        run_dir = manager._resolve_run_dir()
        assert run_dir == prev_run_dir

    def test_restores_by_slurm_job_id_without_ckpt_path(self, cache_dir, monkeypatch):
        """On SLURM requeue: no ckpt_path but same SLURM_JOB_ID → finds prev run_dir."""
        monkeypatch.setenv("SLURM_JOB_ID", "99999")
        # Simulate a previous run for this SLURM job
        prev_run_dir = cache_dir / "runs" / "20260101" / "120000" / "99999"
        prev_run_dir.mkdir(parents=True)
        (prev_run_dir / _RUN_META_FILENAME).write_text(
            json.dumps({"run_dir": str(prev_run_dir), "run_id": "99999"})
        )
        ckpt_dir = prev_run_dir / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "last.ckpt").touch()

        # Manager with NO ckpt_path — this is the requeue scenario
        manager = self._make_manager(ckpt_path=None)
        run_dir = manager._resolve_run_dir()
        assert run_dir == prev_run_dir

    def test_fresh_run_when_sidecar_missing(self, cache_dir):
        ckpt = cache_dir / "stale.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.touch()
        manager = self._make_manager(ckpt_path=ckpt)
        run_dir = manager._resolve_run_dir()
        assert run_dir is not None
        assert run_dir.is_dir()

    def test_fresh_run_when_sidecar_corrupt(self, cache_dir):
        ckpt = cache_dir / "corrupt.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.touch()
        (cache_dir / _RUN_META_FILENAME).write_text("NOT JSON!!")
        manager = self._make_manager(ckpt_path=ckpt)
        run_dir = manager._resolve_run_dir()
        assert run_dir is not None
        assert run_dir.is_dir()

    def test_uses_slurm_job_id(self, cache_dir, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "42")
        manager = self._make_manager()
        run_dir = manager._resolve_run_dir()
        assert run_dir.name == "42"

    def test_tilde_expanded(self, monkeypatch):
        spt_set(cache_dir="~/spt_test_cache")
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
        manager = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
        )
        run_dir = manager._resolve_run_dir()
        assert "~" not in str(run_dir)
        assert str(run_dir).startswith(os.path.expanduser("~"))
        import shutil

        if run_dir.exists():
            shutil.rmtree(run_dir.parent.parent.parent.parent)

    def test_sets_run_id_attribute(self, cache_dir, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "555")
        manager = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
        )
        manager._resolve_run_dir()
        assert manager._run_id == "555"
        assert manager._run_dir.name == "555"


# ============================================================================
# Manager._inject_run_dir_into_trainer_config
# ============================================================================


class TestInjectRunDir:
    """Tests for injecting run_dir into the config tree."""

    def test_injects_into_dictconfig(self, tmp_path):
        cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.Trainer",
                "max_epochs": 1,
                "enable_checkpointing": False,
                "logger": False,
            }
        )
        manager = Manager(
            trainer=cfg,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        run_dir = tmp_path / "my_run"
        run_dir.mkdir()
        manager._inject_run_dir_into_trainer_config(run_dir)
        assert manager.trainer.default_root_dir == str(run_dir)

    def test_overrides_existing_default_root_dir(self, tmp_path):
        cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.Trainer",
                "default_root_dir": "/old/dir",
                "max_epochs": 1,
                "enable_checkpointing": False,
                "logger": False,
            }
        )
        manager = Manager(
            trainer=cfg,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        run_dir = tmp_path / "my_run"
        run_dir.mkdir()
        manager._inject_run_dir_into_trainer_config(run_dir)
        assert manager.trainer.default_root_dir == str(run_dir)

    def test_warns_for_prebuilt_trainer(self, tmp_path):
        trainer = BoringTrainer(enable_checkpointing=False, logger=False)
        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        # Should not crash — just warn
        manager._inject_run_dir_into_trainer_config(tmp_path / "run")


# ============================================================================
# Manager._resolve_load_path — decoupled load logic
# ============================================================================


class TestResolveLoadPath:
    """Tests for resolving checkpoint load paths."""

    def test_returns_user_ckpt_path_when_exists(self, tmp_path):
        """User's explicit ckpt_path is used for loading."""
        user_ckpt = tmp_path / "pretrained.ckpt"
        user_ckpt.touch()
        manager = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
            ckpt_path=str(user_ckpt),
        )
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        result = manager._resolve_load_path(run_dir)
        assert result == str(user_ckpt.resolve())

    def test_returns_none_when_user_ckpt_path_missing_and_no_last_ckpt(self, tmp_path):
        """User's ckpt_path doesn't exist and no last.ckpt → None."""
        manager = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
            ckpt_path=str(tmp_path / "nonexistent.ckpt"),
        )
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        result = manager._resolve_load_path(run_dir)
        assert result is None

    def test_falls_back_to_last_ckpt_when_user_ckpt_path_missing(self, tmp_path):
        """User's ckpt_path doesn't exist but last.ckpt does → load last.ckpt.

        This is the requeue scenario: a stale ckpt_path from a previous job
        no longer exists, but last.ckpt was saved in the run_dir.
        """
        manager = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
            ckpt_path=str(tmp_path / "nonexistent.ckpt"),
        )
        run_dir = tmp_path / "run"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "last.ckpt").touch()

        result = manager._resolve_load_path(run_dir)
        assert result == str(ckpt_dir / "last.ckpt")

    def test_auto_detects_requeue_checkpoint(self, tmp_path):
        """When no ckpt_path but run_dir/checkpoints/last.ckpt exists, load it."""
        manager = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
        )
        run_dir = tmp_path / "run"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "last.ckpt").touch()

        result = manager._resolve_load_path(run_dir)
        assert result == str(ckpt_dir / "last.ckpt")

    def test_returns_none_for_fresh_run(self, tmp_path):
        """No ckpt_path and no existing checkpoint → None (fresh run)."""
        manager = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
        )
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        result = manager._resolve_load_path(run_dir)
        assert result is None

    def test_user_ckpt_path_takes_priority_over_requeue(self, tmp_path):
        """If user passes ckpt_path AND requeue checkpoint exists, user wins."""
        user_ckpt = tmp_path / "pretrained.ckpt"
        user_ckpt.touch()
        manager = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
            ckpt_path=str(user_ckpt),
        )
        run_dir = tmp_path / "run"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "last.ckpt").touch()

        result = manager._resolve_load_path(run_dir)
        assert result == str(user_ckpt.resolve())


# ============================================================================
# Manager._configure_cache_dir_checkpointing
# ============================================================================


class TestConfigureCacheDirCheckpointing:
    """Tests for configuring cache_dir-based checkpointing."""

    def _make_manager_with_trainer(self, tmp_path):
        """Create a Manager with an instantiated trainer and run_dir."""
        trainer = BoringTrainer(
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            logger=False,
        )
        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        manager._trainer = trainer
        manager._run_dir = tmp_path / "run"
        manager._run_dir.mkdir()
        return manager

    def test_adds_model_checkpoint(self, tmp_path):
        manager = self._make_manager_with_trainer(tmp_path)
        initial_count = len(manager._trainer.callbacks)
        manager._configure_cache_dir_checkpointing()

        mc_callbacks = [
            cb for cb in manager._trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ]
        assert len(mc_callbacks) >= 1
        assert len(manager._trainer.callbacks) == initial_count + 1

    def test_saves_to_run_dir_checkpoints(self, tmp_path):
        manager = self._make_manager_with_trainer(tmp_path)
        manager._configure_cache_dir_checkpointing()

        mc = [
            cb for cb in manager._trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ][-1]
        assert mc.dirpath == str(manager._run_dir / "checkpoints")

    def test_no_requeue_checkpoint_when_disabled(self, tmp_path):
        """spt.set(requeue_checkpoint=False) prevents the requeue checkpoint."""
        spt_set(requeue_checkpoint=False)
        manager = self._make_manager_with_trainer(tmp_path)
        manager._configure_cache_dir_checkpointing()

        mc_callbacks = [
            cb for cb in manager._trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ]
        # No requeue "last" should be added
        assert not any(cb.filename == "last" for cb in mc_callbacks)

    def test_always_adds_requeue_checkpoint(self, tmp_path):
        """Even with user's ModelCheckpoint, a requeue 'last' checkpoint is always added."""
        trainer = BoringTrainer(
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            logger=False,
        )
        run_dir = tmp_path / "run"
        save_dir = run_dir / "checkpoints"
        save_dir.mkdir(parents=True)

        user_mc = ModelCheckpoint(
            dirpath=str(save_dir), filename="best", monitor="val_loss"
        )
        trainer.callbacks.append(user_mc)

        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        manager._trainer = trainer
        manager._run_dir = run_dir

        manager._configure_cache_dir_checkpointing()

        mc_callbacks = [
            cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ]
        # User's "best" + our requeue "last"
        assert len(mc_callbacks) == 2
        filenames = {cb.filename for cb in mc_callbacks}
        assert "best" in filenames
        assert "last" in filenames

    def test_redirects_user_checkpoint_to_run_dir(self, tmp_path):
        """User's ModelCheckpoint with custom dirpath gets redirected."""
        trainer = BoringTrainer(
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            logger=False,
        )
        run_dir = tmp_path / "run"
        (run_dir / "checkpoints").mkdir(parents=True)

        user_mc = ModelCheckpoint(
            dirpath="/some/other/path",
            filename="best-{epoch}",
            monitor="val_loss",
            mode="min",
        )
        trainer.callbacks.append(user_mc)

        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        manager._trainer = trainer
        manager._run_dir = run_dir

        manager._configure_cache_dir_checkpointing()

        # dirpath redirected, but filename/monitor/mode preserved
        assert user_mc.dirpath == str(run_dir / "checkpoints")
        assert user_mc.filename == "best-{epoch}"
        assert user_mc.monitor == "val_loss"

    def test_redirects_multiple_checkpoints(self, tmp_path):
        """All ModelCheckpoint callbacks get redirected, plus requeue is added."""
        trainer = BoringTrainer(
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            logger=False,
        )
        run_dir = tmp_path / "run"
        (run_dir / "checkpoints").mkdir(parents=True)

        mc1 = ModelCheckpoint(dirpath="/path/a", filename="every-epoch")
        mc2 = ModelCheckpoint(dirpath="/path/b", filename="best", monitor="val_loss")
        trainer.callbacks.extend([mc1, mc2])

        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        manager._trainer = trainer
        manager._run_dir = run_dir

        manager._configure_cache_dir_checkpointing()

        expected = str(run_dir / "checkpoints")
        assert mc1.dirpath == expected
        assert mc2.dirpath == expected
        # 2 user + 1 requeue
        mc_count = sum(isinstance(cb, ModelCheckpoint) for cb in trainer.callbacks)
        assert mc_count == 3
        filenames = {
            cb.filename for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)
        }
        assert filenames == {"every-epoch", "best", "last"}


# ============================================================================
# _RunDirCallback
# ============================================================================


class TestRunDirCallback:
    """Tests for the run_dir callback wiring."""

    def test_persists_run_dir_in_checkpoint(self):
        cb = _RunDirCallback("/some/path")
        checkpoint = {}
        cb.on_save_checkpoint(None, None, checkpoint)
        assert checkpoint["spt_run_dir"] == "/some/path"

    def test_stores_as_string(self):
        cb = _RunDirCallback(str(Path("/a/b/c")))
        checkpoint = {}
        cb.on_save_checkpoint(None, None, checkpoint)
        assert isinstance(checkpoint["spt_run_dir"], str)


# ============================================================================
# Manager._warn_hydra_conflicts (static method)
# ============================================================================


class TestHydraConflictWarnings:
    """Tests for Hydra config conflict warnings."""

    def test_no_crash_without_hydra(self):
        Manager._warn_hydra_conflicts()


# ============================================================================
# Manager.save_checkpoint with run_dir
# ============================================================================


class TestSaveCheckpointWithRunDir:
    """Tests for saving checkpoints under run_dir."""

    def test_default_path_uses_run_dir(self, tmp_path):
        trainer = BoringTrainer(
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            logger=False,
        )
        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        manager._trainer = trainer
        manager._run_dir = tmp_path / "my_run"
        (manager._run_dir / "checkpoints").mkdir(parents=True)

        saved_path = None

        def mock_save(path):
            nonlocal saved_path
            saved_path = path

        trainer.save_checkpoint = mock_save
        manager.save_checkpoint(verbose=False)
        assert saved_path is not None
        assert "my_run" in saved_path
        assert "checkpoints" in saved_path

    def test_default_path_without_run_dir(self, tmp_path):
        trainer = BoringTrainer(
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            logger=False,
        )
        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
        )
        manager._trainer = trainer

        saved_path = None

        def mock_save(path):
            nonlocal saved_path
            saved_path = path

        trainer.save_checkpoint = mock_save
        manager.save_checkpoint(verbose=False)
        assert saved_path is not None
        assert "checkpoint.ckpt" in saved_path


# ============================================================================
# Integration: Manager.__call__ with cache_dir
# ============================================================================


class TestManagerCallWithCacheDir:
    """Tests for Manager invocation with cache_dir wiring."""

    def test_full_flow_with_config_trainer(self, cache_dir, monkeypatch):
        """Manager.__call__() creates run_dir, injects into trainer, sets up checkpointing."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)

        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.Trainer",
                "max_epochs": 1,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            }
        )

        manager = Manager(
            trainer=trainer_cfg,
            module=BoringModule(),
            data=BoringDataModule(),
            seed=42,
        )

        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda: None
        )

        def mock_fit(self_trainer, module, **kwargs):
            pass

        monkeypatch.setattr(pl.Trainer, "fit", mock_fit)
        manager()

        # run_dir exists
        assert hasattr(manager, "_run_dir")
        assert manager._run_dir.is_dir()
        assert str(manager._run_dir).startswith(str(cache_dir))

        # ckpt_path is untouched (user didn't pass one)
        assert manager.ckpt_path is None

        # _RunDirCallback was added
        assert any(isinstance(cb, _RunDirCallback) for cb in manager._trainer.callbacks)

        # ModelCheckpoint saving to run_dir/checkpoints/ was added
        mc_callbacks = [
            cb for cb in manager._trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ]
        assert any(
            cb.dirpath == str(manager._run_dir / "checkpoints") for cb in mc_callbacks
        )

        # trainer.default_root_dir points to run_dir
        assert manager._trainer.default_root_dir == str(manager._run_dir)

    def test_cache_dir_none_preserves_old_behavior(self, tmp_path, monkeypatch):
        """When cache_dir is explicitly None, no run_dir is created."""
        monkeypatch.delenv("SPT_CACHE_DIR", raising=False)
        get_config()._cache_dir = None
        assert get_config().cache_dir is None

        trainer = BoringTrainer(
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            logger=False,
        )
        manager = Manager(
            trainer=trainer,
            module=BoringModule(),
            data=BoringDataModule(),
            seed=42,
        )
        monkeypatch.setattr(manager, "init_and_sync_wandb", lambda: None)
        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda: None
        )

        def mock_fit(self_trainer, module, **kwargs):
            pass

        monkeypatch.setattr(pl.Trainer, "fit", mock_fit)
        manager()

        assert not hasattr(manager, "_run_dir")
        assert manager.ckpt_path is None

    def test_user_ckpt_path_not_overridden_by_cache_dir(self, cache_dir, monkeypatch):
        """User's ckpt_path stays untouched — used for load only."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)

        user_ckpt = cache_dir / "custom" / "my_model.ckpt"

        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.Trainer",
                "max_epochs": 1,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            }
        )

        manager = Manager(
            trainer=trainer_cfg,
            module=BoringModule(),
            data=BoringDataModule(),
            seed=42,
            ckpt_path=str(user_ckpt),
        )

        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda: None
        )

        def mock_fit(self_trainer, module, **kwargs):
            pass

        monkeypatch.setattr(pl.Trainer, "fit", mock_fit)
        manager()

        # ckpt_path is exactly what user passed (resolved)
        assert manager.ckpt_path == user_ckpt.resolve()
        # But checkpoints are saved to run_dir, NOT to user_ckpt.parent
        mc_callbacks = [
            cb for cb in manager._trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ]
        assert any(
            cb.dirpath == str(manager._run_dir / "checkpoints") for cb in mc_callbacks
        )

    def test_run_meta_written_for_requeue(self, cache_dir, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)

        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.Trainer",
                "max_epochs": 1,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            }
        )

        manager = Manager(
            trainer=trainer_cfg,
            module=BoringModule(),
            data=BoringDataModule(),
            seed=42,
        )

        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda: None
        )

        def mock_fit(self_trainer, module, **kwargs):
            pass

        monkeypatch.setattr(pl.Trainer, "fit", mock_fit)
        manager()

        meta_path = manager._run_dir / _RUN_META_FILENAME
        assert meta_path.is_file()
        meta = json.loads(meta_path.read_text())
        assert Path(meta["run_dir"]) == manager._run_dir

    def test_requeue_loads_from_run_dir_not_ckpt_path(self, cache_dir, monkeypatch):
        """Simulate requeue: run_dir has last.ckpt, no user ckpt_path.

        The fit call should receive the auto-detected checkpoint.
        """
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)

        # Pre-create a "previous run" directory with a checkpoint
        prev_run_dir = cache_dir / "runs" / "20260101" / "120000" / "prev12345678"
        ckpt_dir = prev_run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "last.ckpt").touch()
        (ckpt_dir / _RUN_META_FILENAME).write_text(
            json.dumps({"run_dir": str(prev_run_dir), "run_id": "prev12345678"})
        )
        # Manager has ckpt_path pointing to the prev checkpoint (for sidecar discovery)
        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.Trainer",
                "max_epochs": 1,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            }
        )

        manager = Manager(
            trainer=trainer_cfg,
            module=BoringModule(),
            data=BoringDataModule(),
            seed=42,
            ckpt_path=str(ckpt_dir / "last.ckpt"),
        )

        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda: None
        )

        captured_kwargs = {}

        def mock_fit(self_trainer, module, **kwargs):
            captured_kwargs.update(kwargs)

        monkeypatch.setattr(pl.Trainer, "fit", mock_fit)
        manager()

        # Should have loaded from the user's ckpt_path
        assert captured_kwargs["ckpt_path"] == str((ckpt_dir / "last.ckpt").resolve())
        # And the run_dir should be the restored prev_run_dir (same directory)
        assert manager._run_dir == prev_run_dir

    def test_slurm_requeue_no_ckpt_path_auto_loads(self, cache_dir, monkeypatch):
        """SLURM requeue: no ckpt_path, same SLURM_JOB_ID.

        Should find prev run_dir by job ID and auto-load last.ckpt.
        """
        monkeypatch.setenv("SLURM_JOB_ID", "88888")
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)

        # Simulate a previous run for this SLURM job
        prev_run_dir = cache_dir / "runs" / "20260101" / "100000" / "88888"
        ckpt_dir = prev_run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "last.ckpt").touch()
        (prev_run_dir / _RUN_META_FILENAME).write_text(
            json.dumps({"run_dir": str(prev_run_dir), "run_id": "88888"})
        )

        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.Trainer",
                "max_epochs": 1,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            }
        )

        # NO ckpt_path — this is the requeue scenario
        manager = Manager(
            trainer=trainer_cfg,
            module=BoringModule(),
            data=BoringDataModule(),
            seed=42,
        )

        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda: None
        )

        captured_kwargs = {}

        def mock_fit(self_trainer, module, **kwargs):
            captured_kwargs.update(kwargs)

        monkeypatch.setattr(pl.Trainer, "fit", mock_fit)
        manager()

        # run_dir should be the restored prev_run_dir
        assert manager._run_dir == prev_run_dir
        # fit should have been called with the auto-detected checkpoint
        assert captured_kwargs["ckpt_path"] == str(ckpt_dir / "last.ckpt")
        # ckpt_path on manager stays None (user didn't set it)
        assert manager.ckpt_path is None

    def test_requeue_checkpoint_disabled_no_last_ckpt(self, cache_dir, monkeypatch):
        """With requeue_checkpoint=False, no 'last' ModelCheckpoint is added."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
        spt_set(requeue_checkpoint=False)

        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.Trainer",
                "max_epochs": 1,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "logger": False,
            }
        )

        manager = Manager(
            trainer=trainer_cfg,
            module=BoringModule(),
            data=BoringDataModule(),
            seed=42,
        )

        monkeypatch.setattr(
            "stable_pretraining.manager.print_logger_info", lambda _: None
        )
        monkeypatch.setattr(
            "stable_pretraining.manager.print_signal_info", lambda: None
        )

        def mock_fit(self_trainer, module, **kwargs):
            pass

        monkeypatch.setattr(pl.Trainer, "fit", mock_fit)
        manager()

        mc_callbacks = [
            cb for cb in manager._trainer.callbacks if isinstance(cb, ModelCheckpoint)
        ]
        assert not any(cb.filename == "last" for cb in mc_callbacks)


# ============================================================================
# Callback path resolution tests
# ============================================================================


class TestCallbackPathResolution:
    """Tests for callback path resolution."""

    def test_hf_checkpoint_resolves_relative_save_dir(self):
        try:
            from stable_pretraining.callbacks.hf_models import (
                HuggingFaceCheckpointCallback,
            )
        except ImportError:
            pytest.skip("transformers not installed")

        cb = HuggingFaceCheckpointCallback(save_dir="hf_exports")
        assert not cb.save_dir.is_absolute()

    def test_online_writer_resolves_relative_path(self, tmp_path):
        from stable_pretraining.callbacks.writer import OnlineWriter

        run_dir = tmp_path / "run_dir"
        run_dir.mkdir()

        cb = OnlineWriter(names="test", path="outputs", during="train")
        assert not cb.path.is_absolute()

        mock_trainer = MagicMock()
        mock_trainer.default_root_dir = str(run_dir)
        cb.setup(mock_trainer, MagicMock(), stage="fit")

        assert cb.path.is_absolute()
        assert str(cb.path).startswith(str(run_dir))
        assert cb.path.is_dir()

    def test_online_writer_absolute_path_unchanged(self, tmp_path):
        from stable_pretraining.callbacks.writer import OnlineWriter

        abs_path = tmp_path / "my_abs_outputs"
        abs_path.mkdir()

        cb = OnlineWriter(names="test", path=str(abs_path), during="train")
        assert cb.path.is_absolute()

        mock_trainer = MagicMock()
        mock_trainer.default_root_dir = str(tmp_path / "different_dir")
        cb.setup(mock_trainer, MagicMock(), stage="fit")

        assert cb.path == abs_path


# ============================================================================
# Collision resistance
# ============================================================================


class TestNoCollisions:
    """Tests that independent runs do not collide on paths."""

    def test_two_managers_get_different_run_dirs(self, cache_dir, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)

        m1 = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
        )
        m2 = Manager(
            trainer=BoringTrainer(enable_checkpointing=False, logger=False),
            module=BoringModule(),
            data=BoringDataModule(),
        )

        d1 = m1._resolve_run_dir()
        d2 = m2._resolve_run_dir()
        assert d1 != d2
        assert d1.is_dir()
        assert d2.is_dir()
