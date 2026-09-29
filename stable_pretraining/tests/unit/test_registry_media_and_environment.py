"""Media logs round-trip bytes and environment snapshots tolerate missing tools."""

import json
import pickle
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from stable_pretraining.callbacks import env_info
from stable_pretraining.registry.logger import RegistryLogger

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("kind", ["path", "pil", "tensor", "chw", "gray", "float"])
def test_registry_image_roundtrip_and_default_step(tmp_path, kind):
    root = tmp_path / "run"
    logger = RegistryLogger(root, "run")
    logger.log_metrics({"loss": 1.0}, step=7)
    array = np.full((5, 6, 3), [20, 40, 60], np.uint8)
    if kind == "path":
        source = tmp_path / "source.png"
        Image.fromarray(array).save(source)
    elif kind == "pil":
        source = Image.fromarray(array)
    elif kind in ("tensor", "chw"):
        source = array.transpose(2, 0, 1)
        if kind == "tensor":
            source = torch.from_numpy(source.copy())
    elif kind == "gray":
        source = array[..., :1]
        array = array[..., 0]
    else:
        source = array.astype(float) / 255
    logger.log_image("nested/image", [source], caption=["example"])
    record = json.loads((root / "media.jsonl").read_text())
    assert record["step"] == 7 and record["caption"] == "example"
    np.testing.assert_array_equal(np.asarray(Image.open(root / record["path"])), array)


@pytest.mark.parametrize("kind", ["path", "bytes", "bytearray"])
def test_registry_video_preserves_encoded_content_and_metadata(tmp_path, kind):
    root = tmp_path / "run"
    logger = RegistryLogger(root, "run")
    payload = b"encoded-video-content"
    if kind == "path":
        source = tmp_path / "source.webm"
        source.write_bytes(payload)
    else:
        source = payload if kind == "bytes" else bytearray(payload)
    logger.log_video("rollout", [source], step=4, caption=["sample"], fps=8)
    event = json.loads((root / "media.jsonl").read_text())
    assert (root / event["path"]).read_bytes() == payload
    assert event["fps"] == 8 and event["step"] == 4
    assert event["format"] == ("webm" if kind == "path" else "mp4")


@pytest.mark.parametrize("media", ["image", "video"])
def test_bad_media_does_not_abort_training_or_record_false_success(tmp_path, media):
    logger = RegistryLogger(tmp_path, "run")
    getattr(logger, f"log_{media}")("invalid", [object()])
    assert not (tmp_path / "media.jsonl").exists()
    logger.log_metrics({"loss": 0.5}, step=1)
    logger.finalize("success")
    assert json.loads((tmp_path / "sidecar.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("method", ["_get_cuda_info", "_get_git_info"])
@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError(),
        subprocess.TimeoutExpired("fake", 1),
        subprocess.CalledProcessError(1, "fake"),
    ],
)
def test_environment_external_tool_failures_are_optional(monkeypatch, method, failure):
    monkeypatch.setattr(env_info.subprocess, "check_output", Mock(side_effect=failure))
    assert getattr(env_info.EnvironmentDumpCallback(), method)() is None


def test_environment_reports_driver_and_git_without_remote(monkeypatch):
    callback = env_info.EnvironmentDumpCallback()
    run = Mock(side_effect=["GPU, driver, memory\n", "123.45\n"])
    monkeypatch.setattr(env_info.subprocess, "check_output", run)
    assert callback._get_cuda_info()["driver_version"] == "123.45"
    run.side_effect = [
        ".git",
        "abc123",
        "main",
        " M file.py",
        subprocess.CalledProcessError(1, "fake"),
    ]
    result = callback._get_git_info()
    assert result["has_uncommitted_changes"]
    assert result["commit_hash"] == "abc123" and result["remote_url"] is None


@pytest.mark.parametrize(
    "failure",
    [subprocess.TimeoutExpired("fake", 1), subprocess.CalledProcessError(1, "fake")],
)
def test_package_snapshot_records_failure_without_fabricating_versions(
    monkeypatch, failure
):
    monkeypatch.setattr(env_info.subprocess, "check_output", Mock(side_effect=failure))
    result = env_info.EnvironmentDumpCallback()._get_packages_info()
    assert result["total_packages"] == 0
    assert result["pip_freeze"].startswith("Error")


def test_environment_snapshot_is_serializable_and_does_not_pickle_threads(
    tmp_path, monkeypatch
):
    callback = env_info.EnvironmentDumpCallback(async_dump=False)
    callback._dump_thread = threading.Thread()
    assert pickle.loads(pickle.dumps(callback))._dump_thread is None
    assert callback._make_serializable({"items": (Path("x"), None, 3)}) == {
        "items": ["x", None, 3]
    }
    monkeypatch.setattr(
        callback,
        "_get_python_info",
        lambda: {"version_info": {"major": 3, "minor": 11, "micro": 0}},
    )
    monkeypatch.setattr(
        callback,
        "_get_system_info",
        lambda: {
            "system": "test",
            "release": "test",
            "machine": "cpu",
            "hostname": "local",
        },
    )
    monkeypatch.setattr(
        callback,
        "_get_packages_info",
        lambda: {"total_packages": 1, "pip_freeze": "example==1\n"},
    )
    monkeypatch.setattr(callback, "_get_cuda_info", lambda: {"driver_version": "fake"})
    monkeypatch.setattr(
        callback,
        "_get_git_info",
        lambda: {
            "branch": "main",
            "commit_hash": "abc123",
            "remote_url": None,
            "has_uncommitted_changes": False,
        },
    )
    monkeypatch.setattr(callback, "_get_slurm_info", lambda: {"SLURM_JOB_ID": "123"})
    monkeypatch.setattr(callback, "_get_env_variables", lambda: {})
    callback.setup(
        SimpleNamespace(default_root_dir=str(tmp_path / "snapshot")), None, "fit"
    )
    saved = json.loads((tmp_path / "snapshot/environment.json").read_text())
    assert saved["slurm"] == {"SLURM_JOB_ID": "123"}
    assert (tmp_path / "snapshot/requirements_frozen.txt").read_text() == "example==1\n"
