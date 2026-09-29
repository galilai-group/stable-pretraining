"""Viewer polling and HTTP responses recover from filesystem and client races."""

import io
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from stable_pretraining.registry._sidecar import write_sidecar
from stable_pretraining.web import scan, server

pytestmark = pytest.mark.unit


@pytest.fixture
def populated(tmp_path):
    run = tmp_path / "run"
    write_sidecar(run, {"run_id": "run", "status": "running"})
    (run / "metrics.csv").write_text("step,loss\n0,2\n")
    scanner = scan.RunScanner(tmp_path, poll_interval=0.001)
    scanner._scan()
    return scanner, run


def test_initial_and_background_scan_failures_are_retried(populated, monkeypatch):
    scanner, _ = populated
    actual = scanner._scan
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) <= 2:
            raise OSError("temporarily offline")
        return actual(*args, **kwargs)

    monkeypatch.setattr(scanner, "_scan", flaky)
    events = scanner.subscribe()
    scanner.start()
    try:
        assert scanner.progress_json()["initial_done"]
        (scanner.root / "second").mkdir()
        write_sidecar(scanner.root / "second", {"run_id": "second"})
        for _ in range(20):
            event = events.get(timeout=5)
            if event["type"] == "update":
                assert "second" in event["data"]["changed"]
                break
        else:
            pytest.fail("scanner never recovered")
        assert len(calls) >= 3
    finally:
        scanner.stop()


def test_discovery_skips_unreadable_entries_and_failed_workers(tmp_path, monkeypatch):
    scanner = scan.RunScanner(tmp_path)
    entry = SimpleNamespace(is_dir=Mock(side_effect=OSError("gone")))
    monkeypatch.setattr(scan.os, "scandir", lambda _: nullcontext(iter([entry])))
    assert scanner._list_dir(tmp_path) == ([], [])
    scanner._walk_cache.clear()
    monkeypatch.setattr(scan.os, "scandir", Mock(side_effect=PermissionError("denied")))
    assert scanner._list_dir(tmp_path) == ([], [])
    monkeypatch.setattr(
        scanner, "_list_dir", Mock(side_effect=OSError("worker failed"))
    )
    assert scanner._parallel_walk(scanner.root, report_progress=True) == []
    assert scanner._run_id_for(tmp_path.parent / "outside") == "outside"


def test_discovery_publishes_progress_for_long_walk(populated, monkeypatch):
    scanner, _ = populated
    clock = iter(range(100))
    monkeypatch.setattr(scan.time, "monotonic", lambda: next(clock))
    events = scanner.subscribe()
    assert scanner._parallel_walk(scanner.root, report_progress=True)
    assert events.get_nowait()["type"] == "progress"


@pytest.mark.parametrize(
    "method", ["metrics_json", "metrics_stream", "metrics_json_bytes"]
)
def test_missing_stat_does_not_prevent_readable_metrics(populated, monkeypatch, method):
    scanner, run = populated
    original = Path.stat
    is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: True if path == run / "metrics.csv" else is_file(path),
    )

    def stat(path, *args, **kwargs):
        if path == run / "metrics.csv":
            raise OSError("stat temporarily unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    result = getattr(scanner, method)("run")
    if method == "metrics_stream":
        result = list(result)[0]
    elif method.endswith("bytes"):
        result = json.loads(result)
    assert result["metrics"]["loss"]["y"] == [2.0]


def test_removed_run_during_serialization_returns_not_found(populated, monkeypatch):
    scanner, _ = populated
    monkeypatch.setattr(scanner, "metrics_json", lambda _: None)
    assert scanner.metrics_json_bytes("run") is None


def test_media_read_and_heartbeat_failures_remain_nonfatal(populated, monkeypatch):
    scanner, run = populated
    (run / "media.jsonl").touch()
    original = Path.open

    def opening(path, *args, **kwargs):
        if path.name == "media.jsonl":
            raise PermissionError("denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opening)
    monkeypatch.setattr(
        scan, "heartbeat_mtime", Mock(side_effect=OSError("unavailable"))
    )
    assert scanner.media_json("run") == {"events": []}
    assert scanner.runs_json()[0]["run_id"] == "run"


def test_log_discovery_handles_invalid_output_directory_and_disappeared_log(
    populated, monkeypatch
):
    scanner, run = populated
    sidecar = scanner._runs["run"].sidecar
    sidecar.update(
        {
            "tags": ["sweep:1"],
            "hparams": {
                "output_dir": "~spt_no_such_user/directory",
                "slurm.task_id": 2,
            },
        }
    )
    assert scanner.logs_index("run") == {"streams": []}
    path = run / "train.out"
    path.write_text("hello")
    monkeypatch.setattr(
        scanner,
        "_rediscover_log_paths",
        lambda *args: {"rank bad .out": path, "gone.out": run / "gone.out"},
    )
    streams = scanner.logs_index("run")["streams"]
    assert len(streams) == 1 and streams[0]["rank"] is None and streams[0]["size"] == 5


def _handler():
    handler = server._Handler.__new__(server._Handler)
    handler.send_response = Mock()
    handler.send_header = Mock()
    handler.end_headers = Mock()
    handler.wfile = io.BytesIO()
    return handler


@pytest.mark.parametrize("error", [BrokenPipeError, ConnectionResetError])
@pytest.mark.parametrize("method", ["GET", "PATCH"])
def test_disconnected_clients_do_not_escape_request_handler(error, method):
    handler = _handler()
    handler.path = "/api/runs" if method == "GET" else "/api/run-meta"
    handler._serve_json = Mock(side_effect=error)
    handler.scanner = SimpleNamespace(
        runs_json=lambda: [], patch_run_meta=lambda *a: None
    )
    handler.headers = {"Content-Length": "2"}
    handler.rfile = io.BytesIO(b"{}")
    getattr(handler, f"do_{method}")()
    handler._serve_json.assert_called_once()


def test_stream_route_dispatches_sse_and_empty_metric_stream_returns_404():
    handler = _handler()
    handler.path = "/api/stream"
    handler._serve_sse = Mock()
    handler.do_GET()
    handler._serve_sse.assert_called_once()
    handler.scanner = SimpleNamespace(metrics_stream=lambda _: iter(()))
    handler._serve_metrics_stream("gone")
    assert handler.send_response.call_args.args == (404,)


def test_metric_stream_tolerates_write_failure():
    handler = _handler()
    handler.scanner = SimpleNamespace(
        metrics_stream=lambda _: iter([{"metrics": {}}, {"done": True}])
    )
    handler.wfile = Mock()
    handler.wfile.write.side_effect = OSError("connection closed")
    handler._serve_metrics_stream("run")
    handler.send_response.assert_called_once_with(200)
