"""Recovery never silently starts a different run when metadata is ambiguous."""

import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

import stable_pretraining.manager as runtime
from stable_pretraining._config import get_config
from stable_pretraining.manager import Manager

pytestmark = pytest.mark.unit


@pytest.fixture
def manager(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith(("SLURM_", "TORCHELASTIC_", "MASTER_")):
            monkeypatch.delenv(key)
    monkeypatch.setattr(get_config(), "_cache_dir", str(tmp_path))
    model = Manager.__new__(Manager)
    model.ckpt_path = None
    model.weights_only = False
    return model


def test_requeue_without_job_id_refuses_fresh_run(manager, monkeypatch):
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    with pytest.raises(RuntimeError, match="SLURM_JOB_ID is unset"):
        manager._resolve_run_dir()


def test_requeue_unreadable_index_is_not_treated_as_missing(
    manager, tmp_path, monkeypatch
):
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    index = tmp_path / ".slurm_index" / "123"
    index.parent.mkdir()
    index.write_text(str(tmp_path / "previous"))
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == index:
            raise PermissionError("denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(RuntimeError, match="Failed to read SLURM index"):
        manager._resolve_run_dir()


def test_fresh_run_collision_refuses_existing_checkpoint(
    manager, tmp_path, monkeypatch
):
    now = datetime(2025, 1, 2, 3, 4, 5)
    monkeypatch.setattr(runtime, "datetime", SimpleNamespace(now=lambda: now))
    monkeypatch.setattr(runtime, "_generate_run_id", lambda: "collision")
    checkpoint = tmp_path / "runs/20250102/030405/collision/checkpoints/last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"previous state")
    with pytest.raises(RuntimeError, match="Sanity check failed"):
        manager._resolve_run_dir()
    assert checkpoint.read_bytes() == b"previous state"


def test_handoff_read_failure_retries_without_changing_identity(
    manager, tmp_path, monkeypatch
):
    previous = tmp_path / "previous"
    previous.mkdir()
    handoff = manager._handoff_path(tmp_path, "key")
    handoff.parent.mkdir()
    handoff.write_text(str(previous))
    read = Mock(side_effect=[OSError("not ready"), str(previous)])
    monkeypatch.setattr(Path, "read_text", read)
    assert manager._wait_for_rank_zero_handoff(tmp_path, "key") == previous
    assert read.call_count == 2


def test_handoff_write_failure_and_index_cleanup_failure_are_reported(
    manager, tmp_path, monkeypatch
):
    warning = Mock()
    monkeypatch.setattr(runtime.logging, "warning", warning)
    monkeypatch.setattr(Path, "write_text", Mock(side_effect=OSError("read only")))
    manager._publish_rank_zero_handoff(tmp_path, "key", tmp_path / "run")
    assert "Could not publish" in warning.call_args.args[0]
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(
        runtime.os, "replace", Mock(side_effect=OSError("replace denied"))
    )
    monkeypatch.setattr(Path, "unlink", Mock(side_effect=OSError("unlink denied")))
    result = manager._write_slurm_index(tmp_path, tmp_path / "run")
    assert "FAILED" in result and "replace denied" in result


def test_launch_key_survives_missing_process_group(manager, monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "12345")
    monkeypatch.setattr(runtime.os, "getpgid", Mock(side_effect=OSError("unavailable")))
    assert runtime._ddp_launch_key() == "local-localhost-12345-nopgid"


def test_missing_trainer_root_and_data_config_are_resolved(manager, tmp_path):
    manager.trainer = OmegaConf.create({"default_root_dir": "???", "devices": 1})
    assert not manager._cfg_has("default_root_dir")
    assert manager._cfg_has("devices")
    manager._inject_run_dir_into_trainer_config(tmp_path)
    assert manager.trainer.default_root_dir == str(tmp_path)
    manager.module = object()
    manager.data = OmegaConf.create({"batch_size": 4})
    assert manager._flatten_hydra_config()["data.batch_size"] == 4
    manager.trainer = object()
    assert not manager._cfg_has("devices")
    manager._apply_fast_trainer_defaults()


def test_disabled_registry_and_uninitialized_wandb_do_not_create_loggers(
    manager, monkeypatch
):
    monkeypatch.setattr(get_config(), "_default_loggers", {"registry": False})
    manager._inject_registry_logger()
    assert not hasattr(manager, "_trainer")
    monkeypatch.setattr(runtime, "WANDB_AVAILABLE", False)
    assert manager._wandb_previous_dir() is None
