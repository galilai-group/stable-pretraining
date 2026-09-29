"""Uploader concurrency is real; Google API calls terminate at a fake SDK."""

import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from stable_pretraining.utils import gdrive_utils as drive

pytestmark = pytest.mark.unit


@pytest.fixture
def sdk(monkeypatch):
    modules = {}
    for name in [
        "google",
        "google.oauth2",
        "google.oauth2.service_account",
        "googleapiclient",
        "googleapiclient.discovery",
        "googleapiclient.http",
        "googleapiclient.errors",
    ]:
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)

    class HttpError(Exception):
        resp = SimpleNamespace(status=403, reason="forbidden")

    service = Mock()
    service.files.return_value.list.return_value.execute.return_value = {"files": []}
    service.files.return_value.create.return_value.execute.return_value = {
        "id": "created",
        "name": "file",
        "size": 4,
    }
    service.about.return_value.get.return_value.execute.return_value = {
        "user": {"emailAddress": "fake@example.test"}
    }
    creds = Mock()
    modules["google.oauth2.service_account"].Credentials = creds
    build = Mock(return_value=service)
    modules["googleapiclient.discovery"].build = build
    media = Mock()
    modules["googleapiclient.http"].MediaFileUpload = media
    modules["googleapiclient.errors"].HttpError = HttpError
    monkeypatch.setattr(drive.atexit, "register", Mock())
    return SimpleNamespace(
        service=service, creds=creds, build=build, media=media, HttpError=HttpError
    )


def _uploader(sdk):
    uploader = drive.GDriveUploader.__new__(drive.GDriveUploader)
    uploader.service = sdk.service
    uploader.folder_name = "results"
    uploader.folder_id = "root"
    uploader.parent_folder_id = None
    return uploader


@pytest.mark.parametrize(
    "names,expected",
    [
        ([], "results"),
        (["results"], "results_v2"),
        (["results", "results_v3", "results_vbad"], "results_v4"),
        (["results", "other_results_v99"], "results_v2"),
    ],
)
@pytest.mark.parametrize("parent", [None, "parent"])
def test_folder_versioning_is_scoped_to_exact_base_name(sdk, names, expected, parent):
    uploader = _uploader(sdk)
    uploader.parent_folder_id = parent
    sdk.service.files.return_value.list.return_value.execute.return_value = {
        "files": [{"name": name} for name in names]
    }
    assert uploader._create_or_version_folder() == "created"
    body = sdk.service.files.return_value.create.call_args.kwargs["body"]
    assert body["name"] == expected
    assert body.get("parents") == ([parent] if parent else None)
    assert uploader.get_folder_url().endswith("/root")
    assert uploader.get_folder_id() == "root"


@pytest.mark.parametrize("operation", ["list", "create"])
def test_folder_api_errors_propagate(sdk, operation):
    getattr(
        sdk.service.files.return_value, operation
    ).return_value.execute.side_effect = OSError("service unavailable")
    with pytest.raises(OSError, match="unavailable"):
        _uploader(sdk)._create_or_version_folder()


def test_authentication_validates_file_and_passes_credentials(sdk, tmp_path):
    uploader = _uploader(sdk)
    uploader.credentials_path = str(tmp_path / "credentials.json")
    with pytest.raises(FileNotFoundError):
        uploader._authenticate()
    sdk.build.assert_not_called()
    Path(uploader.credentials_path).write_text("{}")
    assert uploader._authenticate() is sdk.service
    sdk.creds.from_service_account_file.assert_called_once_with(
        uploader.credentials_path, scopes=["https://www.googleapis.com/auth/drive"]
    )
    sdk.build.assert_called_once_with(
        "drive", "v3", credentials=sdk.creds.from_service_account_file.return_value
    )


@pytest.mark.parametrize(
    "suffix,mime",
    [(".txt", "text/plain"), (".unrecognized-spt", "application/octet-stream")],
)
@pytest.mark.parametrize("custom", [False, True])
def test_upload_metadata_and_mime_type(sdk, tmp_path, suffix, mime, custom):
    path = tmp_path / ("content" + suffix)
    path.write_bytes(b"data")
    uploader = _uploader(sdk)
    args = {"custom_name": "renamed", "subfolder_id": "sub"} if custom else {}
    assert uploader._upload_file(str(path), **args) == "created"
    sdk.media.assert_called_once_with(str(path), mimetype=mime, resumable=True)
    body = sdk.service.files.return_value.create.call_args.kwargs["body"]
    assert body == {
        "name": "renamed" if custom else path.name,
        "parents": ["sub" if custom else "root"],
    }


@pytest.mark.parametrize("failure", ["missing", "directory", "http", "other"])
def test_upload_failures_return_none(sdk, tmp_path, failure):
    path = tmp_path / "file.txt"
    if failure == "directory":
        path.mkdir()
    elif failure != "missing":
        path.write_text("data")
        sdk.service.files.return_value.create.return_value.execute.side_effect = (
            sdk.HttpError() if failure == "http" else OSError("failure")
        )
    assert _uploader(sdk)._upload_file(str(path)) is None


@pytest.mark.parametrize("callback_raises", [False, True])
def test_background_worker_finishes_all_tasks_even_if_callback_raises(
    sdk, tmp_path, callback_raises
):
    credentials = tmp_path / "credentials.json"
    credentials.write_text("{}")
    source = tmp_path / "data.txt"
    source.write_text("data")
    completed = threading.Event()
    calls = []

    def callback(path, file_id, success):
        calls.append((path, file_id, success))
        if len(calls) == 12:
            completed.set()
        if callback_raises:
            raise RuntimeError("callback failed")

    uploader = drive.GDriveUploader("results", str(credentials), callback=callback)
    try:
        for _ in range(12):
            uploader.upload_file(str(source))
        assert completed.wait(3), "worker did not process queued uploads"
        assert uploader.wait_for_uploads(timeout=1)
        assert uploader.get_queue_size() == 0
        assert calls == [(str(source), "created", True)] * 12
    finally:
        uploader._cleanup()
    assert not uploader._worker_thread.is_alive()


def test_wait_counts_inflight_task_and_honors_timeout(sdk, tmp_path, monkeypatch):
    credentials = tmp_path / "credentials.json"
    credentials.write_text("{}")
    started, release = threading.Event(), threading.Event()

    def upload(*args):
        started.set()
        assert release.wait(3)
        return "uploaded"

    monkeypatch.setattr(drive.GDriveUploader, "_upload_file", upload)
    uploader = drive.GDriveUploader("results", str(credentials))
    try:
        uploader.upload_file("fake-path")
        assert started.wait(2)
        assert uploader.get_queue_size() == 0
        assert uploader.wait_for_uploads(timeout=0.01) is False
        release.set()
        assert uploader.wait_for_uploads(timeout=2) is True
    finally:
        release.set()
        uploader._cleanup()


def test_constructor_auth_failure_does_not_register_cleanup(sdk, tmp_path):
    with pytest.raises(FileNotFoundError):
        drive.GDriveUploader("results", str(tmp_path / "missing"))
    drive.atexit.register.assert_not_called()


def test_idle_worker_recovers_and_shutdown_is_idempotent(sdk, tmp_path, monkeypatch):
    from queue import Queue, Empty

    idle = threading.Event()

    class IdleOnceQueue(Queue):
        def get(self, *args, **kwargs):
            if not idle.is_set():
                idle.set()
                raise Empty
            return super().get(*args, **kwargs)

    monkeypatch.setattr(drive, "Queue", IdleOnceQueue)
    credentials = tmp_path / "credentials.json"
    credentials.write_text("{}")
    source = tmp_path / "source.txt"
    source.write_text("data")
    uploader = drive.GDriveUploader("results", str(credentials))
    try:
        assert idle.wait(3)
        uploader.upload_file(str(source))
        assert uploader.wait_for_uploads(timeout=3)
    finally:
        uploader._cleanup()
    uploader._cleanup()
    assert not uploader._worker_thread.is_alive()
    with pytest.raises(RuntimeError, match="shut down"):
        uploader.upload_file(str(source))
