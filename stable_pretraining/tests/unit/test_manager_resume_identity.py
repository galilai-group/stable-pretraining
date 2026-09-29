"""A restart must reuse only a compatible logger identity, before SDK init."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import stable_pretraining.manager as manager_module
from stable_pretraining.manager import Manager

pytestmark = pytest.mark.unit


@pytest.fixture(params=["wandb", "trackio", "swanlab"])
def resume_backend(request, tmp_path, monkeypatch):
    backend = request.param
    logger = SimpleNamespace(
        _wandb_init={"project": "project", "entity": "team"},
        _id=None,
        _project="project",
        set_resume=Mock(),
    )
    monkeypatch.setattr(manager_module, f"find_{backend}_logger", lambda _: logger)
    manager = Manager.__new__(Manager)
    manager._trainer = object()
    manager.ckpt_path = None
    manager._run_dir = tmp_path / "run"
    manager._run_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    method = getattr(
        manager,
        "_maybe_restore_wandb_run_id"
        if backend == "wandb"
        else f"_maybe_restore_{backend}_run",
    )
    key = "name" if backend == "trackio" else "id"

    def restored():
        return (
            logger._id
            if backend == "wandb"
            else (
                logger.set_resume.call_args.args[0]
                if logger.set_resume.called
                else None
            )
        )

    return SimpleNamespace(
        manager=manager,
        logger=logger,
        method=method,
        key=key,
        filename=f"{backend}_resume.json",
        restored=restored,
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_uses_saved_identity_before_logger_initialization(
    resume_backend, tmp_path, legacy
):
    case = resume_backend
    root = case.manager._run_dir
    if legacy:
        del case.manager._run_dir
        case.manager.ckpt_path = tmp_path / "checkpoint.ckpt"
        case.manager.ckpt_path.touch()
        root = tmp_path
    (root / case.filename).write_text(
        json.dumps({case.key: "previous-run", "project": "project", "entity": "team"})
    )
    case.method()
    assert case.restored() == "previous-run"
    if case.key == "id" and case.filename.startswith("wandb"):
        assert case.logger._wandb_init["id"] == "previous-run"


@pytest.mark.parametrize(
    "content",
    [
        None,
        "broken json",
        "{}",
        '{"id":"other","name":"other","project":"wrong"}',
        "[]",
        "null",
    ],
)
def test_resume_ignores_missing_corrupt_or_incompatible_sidecar(
    resume_backend, content
):
    case = resume_backend
    if content is not None:
        (case.manager._run_dir / case.filename).write_text(content)
    case.method()
    assert case.restored() is None


def test_resume_prefers_run_directory_over_working_directory(resume_backend, tmp_path):
    case = resume_backend
    (tmp_path / case.filename).write_text(json.dumps({case.key: "wrong-run"}))
    (case.manager._run_dir / case.filename).write_text(
        json.dumps({case.key: "correct-run"})
    )
    case.method()
    assert case.restored() == "correct-run"


def test_legacy_resume_requires_checkpoint_evidence(resume_backend, tmp_path):
    case = resume_backend
    del case.manager._run_dir
    (tmp_path / case.filename).write_text(json.dumps({case.key: "previous-run"}))
    case.method()
    assert case.restored() is None


def test_wandb_does_not_resume_into_another_entity(tmp_path, monkeypatch):
    logger = SimpleNamespace(_wandb_init={"entity": "new-team"}, _id=None)
    monkeypatch.setattr(manager_module, "find_wandb_logger", lambda _: logger)
    manager = Manager.__new__(Manager)
    manager._trainer = object()
    manager.ckpt_path = None
    manager._run_dir = tmp_path
    (tmp_path / "wandb_resume.json").write_text('{"id":"old-run","entity":"old-team"}')
    manager._maybe_restore_wandb_run_id()
    assert logger._id is None


@pytest.mark.parametrize("location", ["run", "cwd", "explicit"])
def test_save_checkpoint_uses_correct_destination(tmp_path, monkeypatch, location):
    monkeypatch.chdir(tmp_path)
    manager = Manager.__new__(Manager)
    manager._trainer = SimpleNamespace(save_checkpoint=Mock())
    manager._upload_checkpoint_for_requeue = Mock()
    if location == "run":
        manager._run_dir = tmp_path / "run"
        expected = manager._run_dir / "checkpoints" / "checkpoint.ckpt"
    elif location == "explicit":
        expected = tmp_path / "custom" / "nested" / "file.ckpt"
    else:
        expected = tmp_path / "checkpoint.ckpt"
    manager.save_checkpoint(
        str(expected) if location == "explicit" else None, upload_wandb=True
    )
    manager._trainer.save_checkpoint.assert_called_once_with(str(expected))
    manager._upload_checkpoint_for_requeue.assert_called_once_with(expected)
    if location == "explicit":
        assert expected.parent.is_dir()


def test_failed_save_never_uploads_checkpoint(tmp_path):
    manager = Manager.__new__(Manager)
    manager._trainer = SimpleNamespace(
        save_checkpoint=Mock(side_effect=OSError("disk full"))
    )
    manager._upload_checkpoint_for_requeue = Mock()
    with pytest.raises(OSError, match="disk full"):
        manager.save_checkpoint(
            str(tmp_path / "last.ckpt"), upload_wandb=True, verbose=False
        )
    manager._upload_checkpoint_for_requeue.assert_not_called()


@pytest.mark.parametrize(
    "component,class_name", [("module", "BoringModule"), ("data", "BoringDataModule")]
)
def test_configured_component_is_instantiated_once_until_replaced(
    component, class_name
):
    manager = Manager.__new__(Manager)
    register = getattr(manager, f"_register_{component}")
    specification = {"_target_": f"stable_pretraining.tests.utils.{class_name}"}
    register(specification)
    first = getattr(manager, f"instantiated_{component}")
    assert getattr(manager, f"instantiated_{component}") is first
    register(specification)
    replacement = getattr(manager, f"instantiated_{component}")
    assert replacement is not first
    assert getattr(manager, f"instantiated_{component}") is replacement


@pytest.mark.parametrize("active", [False, True])
def test_hydra_output_conflicts_are_reported_only_when_active(monkeypatch, active):
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    warn = Mock()
    monkeypatch.setattr(manager_module.logging, "warning", warn)
    monkeypatch.setattr(HydraConfig, "initialized", lambda: active)
    monkeypatch.setattr(
        HydraConfig,
        "get",
        lambda: OmegaConf.create(
            {"job": {"chdir": True}, "run": {"dir": "run"}, "sweep": {"dir": "sweep"}}
        ),
    )
    Manager._warn_hydra_conflicts()
    assert warn.call_count == (3 if active else 0)
    if active:
        messages = " ".join(call.args[0] for call in warn.call_args_list)
        assert (
            "job.chdir=True" in messages
            and "run.dir" in messages
            and "sweep.dir" in messages
        )
