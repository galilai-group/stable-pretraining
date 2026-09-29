"""Offline logging and checkpoint upload boundaries without contacting services."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import stable_pretraining.manager as manager_module
from stable_pretraining.manager import Manager

pytestmark = pytest.mark.unit


@pytest.fixture
def offline_run(tmp_path, monkeypatch):
    root = tmp_path / "wandb"
    current = root / "offline-run-20260926_120000-run1"
    files = current / "files"
    files.mkdir(parents=True)
    configuration = {"lr": 0.1}
    run = SimpleNamespace(
        id="run1",
        dir=str(files),
        offline=True,
        config=Mock(),
        summary=SimpleNamespace(_as_dict=lambda: {"loss": 0.25}),
        log_artifact=Mock(),
    )
    run.config.as_dict.return_value = configuration
    sdk = SimpleNamespace(run=run, config={}, Artifact=Mock())
    monkeypatch.setattr(manager_module, "WANDB_AVAILABLE", True)
    monkeypatch.setattr(manager_module, "wandb", sdk)
    manager = Manager.__new__(Manager)
    manager._trainer = object()
    manager._run_dir = tmp_path / "run"
    logger = SimpleNamespace(_wandb_init={}, experiment=run)
    monkeypatch.setattr(manager_module, "find_wandb_logger", lambda _: logger)
    return SimpleNamespace(
        manager=manager, run=run, sdk=sdk, root=root, files=files, logger=logger
    )


def test_first_offline_run_needs_no_previous_config(offline_run):
    case = offline_run
    assert case.manager._wandb_previous_dir() is None
    case.manager.init_and_sync_wandb()
    case.run.config.update.assert_not_called()


def test_offline_resume_uses_latest_previous_run_config(offline_run):
    case = offline_run
    previous = case.root / "offline-run-20260925_120000-run1"
    (previous / "files").mkdir(parents=True)
    (previous / "files" / "wandb-config.json").write_text(
        '{"lr":0.2,"ckpt_path":"saved.ckpt"}'
    )
    assert case.manager._wandb_previous_dir() == previous
    case.manager.init_and_sync_wandb()
    case.run.config.update.assert_called_once_with(
        {"lr": 0.2, "ckpt_path": "saved.ckpt"}
    )


def test_offline_summary_and_config_are_saved_without_overwriting(offline_run):
    case = offline_run
    case.manager._dump_wandb_data()
    assert json.loads((case.files / "wandb-summary.json").read_text()) == {"loss": 0.25}
    assert json.loads((case.files / "wandb-config.json").read_text()) == {"lr": 0.1}
    with pytest.raises(RuntimeError, match="Summary file already exists"):
        case.manager._dump_wandb_data()
    (case.files / "wandb-summary.json").unlink()
    with pytest.raises(RuntimeError, match="Config file already exists"):
        case.manager._dump_wandb_data()
    assert json.loads((case.files / "wandb-config.json").read_text()) == {"lr": 0.1}


def test_offline_requeue_keeps_checkpoint_and_records_its_path(offline_run, tmp_path):
    case = offline_run
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"saved checkpoint")
    case.manager._upload_checkpoint_for_requeue(checkpoint)
    assert checkpoint.read_bytes() == b"saved checkpoint"
    case.run.config.update.assert_called_once_with(
        {"ckpt_path": str(checkpoint.resolve())}
    )
    case.run.log_artifact.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_online_checkpoint_is_retained_until_sdk_accepts_artifact(
    offline_run, tmp_path, fail
):
    case = offline_run
    case.run.offline = False
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"saved checkpoint")

    def log_artifact(artifact):
        assert checkpoint.read_bytes() == b"saved checkpoint"
        if fail:
            raise OSError("upload rejected")

    case.run.log_artifact.side_effect = log_artifact
    if fail:
        with pytest.raises(OSError, match="upload rejected"):
            case.manager._upload_checkpoint_for_requeue(checkpoint)
        assert checkpoint.is_file()
    else:
        case.manager._upload_checkpoint_for_requeue(checkpoint)
        assert not checkpoint.exists()
    artifact = case.sdk.Artifact.return_value
    artifact.add_file.assert_called_once_with(str(checkpoint))
    assert artifact.ttl.days == 30


@pytest.mark.parametrize("config_exists", [False, True])
def test_online_config_preserves_existing_values(offline_run, config_exists):
    case = offline_run
    case.run.offline = False
    case.sdk.config = {"lr": 0.2} if config_exists else {}
    case.manager._flatten_hydra_config = lambda: {"lr": 0.1}
    case.manager.init_and_sync_wandb()
    assert case.sdk.config == {"lr": 0.2 if config_exists else 0.1}


def test_logger_directory_is_set_before_experiment_initialization(offline_run):
    case = offline_run
    case.sdk.run = None
    case.run.offline = False

    class Logger:
        _wandb_init = {}

        @property
        def experiment(self):
            assert self._wandb_init["dir"] == str(case.manager._run_dir)
            assert self._save_dir == str(case.manager._run_dir)
            return case.run

    case.manager._flatten_hydra_config = lambda: {}
    from unittest.mock import patch

    with patch.object(manager_module, "find_wandb_logger", return_value=Logger()):
        case.manager.init_and_sync_wandb()
