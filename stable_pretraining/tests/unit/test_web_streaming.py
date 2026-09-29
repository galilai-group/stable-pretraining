"""HTTP streaming, media access, and server shutdown contracts."""

import http.client
import io
import json
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from stable_pretraining.web import server
from stable_pretraining.web.scan import RunScanner

pytestmark = pytest.mark.unit


@pytest.fixture
def http_server(tmp_path):
    scanner = RunScanner(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    (run / "sidecar.json").write_text(
        json.dumps({"run_id": "run", "status": "running"})
    )
    (run / "metrics.csv").write_text("step,loss\n0,2\n1,1\n")
    scanner._scan()
    handler = type("Handler", (server._Handler,), {"scanner": scanner})
    srv = server._Server(("127.0.0.1", 0), handler)
    worker = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01})
    worker.start()

    def request(path, method="GET", body=None, headers=None):
        conn = http.client.HTTPConnection(*srv.server_address, timeout=3)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    try:
        yield request, scanner, run
    finally:
        srv.shutdown()
        srv.server_close()
        worker.join(timeout=3)
        assert not worker.is_alive()


def test_metrics_stream_is_decodable_chunked_ndjson(http_server):
    request, _, _ = http_server
    status, headers, body = request("/api/metrics-stream?run_id=run")
    assert status == 200
    assert headers["Transfer-Encoding"] == "chunked"
    chunks = [json.loads(line) for line in body.splitlines()]
    assert chunks[-1]["done"] is True
    assert chunks[0]["metrics"]["loss"]["y"] == [2.0, 1.0]


@pytest.mark.parametrize(
    "path,status",
    [
        ("/api/metrics-stream", 400),
        ("/api/metrics-stream?run_id=absent", 404),
        ("/api/media", 400),
        ("/api/media?run_id=absent", 404),
        ("/api/media?run_id=run", 200),
        ("/api/media-file", 400),
        ("/api/media-file?run_id=run&path=missing.png", 404),
        ("/api/log-content?run_id=run&stream_id=absent", 404),
        ("/favicon.ico", 204),
    ],
)
def test_stream_and_media_routes_handle_missing_inputs(http_server, path, status):
    request, _, _ = http_server
    assert request(path)[0] == status


def test_media_file_streams_all_bytes_with_immutable_cache(http_server):
    request, _, run = http_server
    content = bytes(range(256)) * 600
    (run / "media").mkdir()
    (run / "media/image.png").write_bytes(content)
    status, headers, body = request("/api/media-file?run_id=run&path=media/image.png")
    assert status == 200
    assert body == content
    assert int(headers["Content-Length"]) == len(content)
    assert headers["Content-Type"] == "image/png"
    assert "immutable" in headers["Cache-Control"]


def test_assets_cannot_escape_to_sibling_with_same_prefix(
    http_server, tmp_path, monkeypatch
):
    request, _, _ = http_server
    assets = tmp_path / "assets"
    assets.mkdir()
    sibling = tmp_path / "assets-private"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("private")
    monkeypatch.setattr(server, "ASSETS_DIR", assets)
    status, _, body = request("/assets/../assets-private/secret.txt")
    assert status == 403
    assert b"private" not in body


@pytest.mark.parametrize("length", ["invalid", "-1"])
def test_invalid_content_length_returns_bad_request(http_server, length):
    request, _, _ = http_server
    status, _, body = request(
        "/api/run-meta", "PATCH", headers={"Content-Length": length}
    )
    assert status == 400
    assert "error" in json.loads(body)


def test_sse_sends_ready_keepalive_and_event_then_unsubscribes():
    events = Mock()
    events.get.side_effect = [
        queue.Empty,
        {"type": "update", "data": {"score": float("nan")}},
        BrokenPipeError,
    ]
    handler = server._Handler.__new__(server._Handler)
    handler.scanner = SimpleNamespace(
        subscribe=Mock(return_value=events), unsubscribe=Mock()
    )
    handler.wfile = io.BytesIO()
    handler.send_response = Mock()
    handler.send_header = Mock()
    handler.end_headers = Mock()
    handler._serve_sse()
    assert (
        handler.wfile.getvalue()
        == b'event: ready\ndata: {}\n\n: ping\n\nevent: update\ndata: {"score": null}\n\n'
    )
    handler.scanner.unsubscribe.assert_called_once_with(events)
    handler.send_response.assert_called_once_with(200)


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_serve_always_stops_scanner_and_closes_socket(tmp_path, monkeypatch, failure):
    scanner = Mock()
    srv = Mock()
    srv.serve_forever.side_effect = failure
    monkeypatch.setattr(server, "RunScanner", Mock(return_value=scanner))
    monkeypatch.setattr(server, "_Server", Mock(return_value=srv))
    if failure is RuntimeError:
        with pytest.raises(RuntimeError):
            server.serve(tmp_path, port=0)
    else:
        server.serve(tmp_path, port=0)
    scanner.start.assert_called_once_with()
    scanner.stop.assert_called_once_with()
    srv.server_close.assert_called_once_with()


def test_failed_bind_does_not_start_background_scanner(tmp_path, monkeypatch):
    scanner = Mock()
    monkeypatch.setattr(server, "RunScanner", Mock(return_value=scanner))
    monkeypatch.setattr(server, "_Server", Mock(side_effect=OSError("in use")))
    with pytest.raises(OSError, match="in use"):
        server.serve(tmp_path, port=0)
    scanner.start.assert_not_called()


def test_server_rejects_non_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        server.serve(tmp_path / "missing")
