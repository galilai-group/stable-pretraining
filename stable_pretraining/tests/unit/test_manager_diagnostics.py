"""Startup diagnostics stay offline and preemption forwarding tolerates missing metadata."""

import signal
from unittest.mock import Mock

import pytest
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger

import stable_pretraining.manager as manager
from stable_pretraining.loggers.trackio import TrackioLogger
from stable_pretraining.loggers.swanlab import SwanLabLogger

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "kind",
    ["dummy", "csv", "tensorboard", "wandb", "trackio", "swanlab", "none", "unknown"],
)
def test_logger_diagnostics_do_not_initialize_remote_experiment(
    monkeypatch, tmp_path, kind
):
    messages = []
    monkeypatch.setattr(manager.logging, "info", messages.append)
    warning = Mock()
    monkeypatch.setattr(manager.logging, "warning", warning)
    headers = []
    monkeypatch.setattr(manager, "log_header", headers.append)
    if kind == "dummy":
        logger = DummyLogger()
    elif kind == "csv":
        logger = CSVLogger(str(tmp_path), name="project")
    elif kind == "tensorboard":
        pytest.importorskip("tensorboard")
        logger = TensorBoardLogger(str(tmp_path), name="project")
    elif kind == "wandb":
        logger = WandbLogger(project="project", offline=True, save_dir=tmp_path)
    elif kind == "trackio":
        pytest.importorskip("trackio")
        logger = TrackioLogger(
            project="project", name="run", group="group", auto_log_gpu=False
        )
    elif kind == "swanlab":
        pytest.importorskip("swanlab", minversion="0.8")
        logger = SwanLabLogger(
            project="project",
            experiment_name="run",
            group="group",
            id="existing",
            mode="disabled",
        )
    else:
        logger = None if kind == "none" else object()
    manager.print_logger_info(logger)
    if kind in ("none", "unknown"):
        warning.assert_called_once()
    else:
        assert headers == [type(logger).__name__]
        if kind != "dummy":
            assert "project" in " ".join(messages)
        if kind in ("wandb", "trackio", "swanlab"):
            assert getattr(logger, "_experiment", None) is None


@pytest.mark.parametrize(
    "module,tag",
    [
        ("submitit.core", "[submitit]"),
        ("lightning.core", "[lightning]"),
        ("stable_pretraining.manager", "[spt]"),
        ("other", ""),
    ],
)
def test_signal_descriptions_identify_handler_owner(module, tag):
    class Handler:
        def handle(self, *args):
            pass

    Handler.__module__ = module
    description = manager._describe_handler(Handler().handle)
    assert f"{module}.Handler.handle" in description
    assert description.endswith(tag) if tag else "[" not in description


def test_unknown_signal_and_submitit_lookup_failure_still_forward(monkeypatch):
    installed = {}
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(
        manager.submitit.JobEnvironment,
        "_usr_sig",
        Mock(side_effect=RuntimeError("not ready")),
    )
    monkeypatch.setattr(manager.signal, "getsignal", lambda _: signal.SIG_DFL)
    monkeypatch.setattr(
        manager.signal, "signal", lambda sig, handler: installed.update({sig: handler})
    )
    monkeypatch.setattr(manager.os, "uname", Mock(side_effect=OSError("unavailable")))
    kill = Mock()
    monkeypatch.setattr(manager.os, "kill", kill)
    manager._install_sigterm_preempt_handler()
    installed[signal.SIGTERM](99999, None)
    kill.assert_called_once_with(manager.os.getpid(), signal.SIGUSR2)
    assert manager._describe_handler(99999) == "99999"
