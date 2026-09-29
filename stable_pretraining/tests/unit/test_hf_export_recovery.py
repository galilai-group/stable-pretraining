"""A failed export commit must preserve the last loadable model snapshot."""

import errno
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from stable_pretraining.callbacks import hf_models
from stable_pretraining.tests.unit.test_hf_models import SimpleHFConfig, SimpleHFModel

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "problem", ["missing_temp", "disk_full", "concurrent_writer", "rollback_failure"]
)
def test_failed_export_install_preserves_previous_weights(
    tmp_path, monkeypatch, problem
):
    callback = hf_models.HuggingFaceCheckpointCallback(save_dir=str(tmp_path))
    model = nn.Module()
    model.backbone = SimpleHFModel(SimpleHFConfig(dim=4))
    trainer = SimpleNamespace(
        global_rank=0, global_step=1, default_root_dir=str(tmp_path)
    )
    callback._do_export(trainer, model)
    old_weights = (tmp_path / "last/backbone/model.safetensors").read_bytes()
    with torch.no_grad():
        model.backbone.proj.weight.add_(1)
    rename = hf_models.os.rename
    failure = (
        FileNotFoundError("temp gone")
        if problem == "missing_temp"
        else OSError(errno.ENOSPC, "disk full")
    )

    def interrupted(source, destination):
        source, destination = Path(source), Path(destination)
        if ".tmp." in source.name:
            if problem == "concurrent_writer":
                destination.mkdir()
                (destination / "other-writer").write_text("complete")
            raise failure
        if ".old." in source.name and problem == "rollback_failure":
            raise PermissionError("rollback denied")
        return rename(source, destination)

    monkeypatch.setattr(hf_models.os, "rename", interrupted)
    with pytest.raises(type(failure), match="(temp gone|disk full)"):
        callback._do_export(trainer, model)
    assert not list(tmp_path.glob(".last.tmp.*"))
    if problem in ("concurrent_writer", "rollback_failure"):
        saved = list(tmp_path.glob(".last.old.*/backbone/model.safetensors"))
        assert len(saved) == 1 and saved[0].read_bytes() == old_weights
        if problem == "concurrent_writer":
            assert (tmp_path / "last/other-writer").read_text() == "complete"
    else:
        assert (
            tmp_path / "last/backbone/model.safetensors"
        ).read_bytes() == old_weights
        assert not list(tmp_path.glob(".last.old.*"))


def test_nonzero_rank_never_exports(tmp_path, monkeypatch):
    callback = hf_models.HuggingFaceCheckpointCallback(save_dir=str(tmp_path))
    export = Mock()
    monkeypatch.setattr(callback, "_do_export", export)
    callback.on_save_checkpoint(SimpleNamespace(global_rank=1), nn.Module(), {})
    export.assert_not_called()
