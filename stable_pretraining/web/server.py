"""Stdlib HTTP server for the spt-web viewer.

Routes:
    GET /                            → assets/index.html
    GET /assets/<file>               → static asset
    GET /api/runs                    → JSON list of runs (sidecar summaries)
    GET /api/scan-status             → JSON {phase, total, done, initial_done}
    GET /api/metrics?run_id=…        → JSON sparse metrics (warm-cache friendly)
    GET /api/metrics-stream?run_id=… → NDJSON chunks streamed during CSV parse
    GET /api/logs?run_id=…           → JSON list of available .out/.err streams
    GET /api/log-content?run_id=…&stream_id=… → text/plain (last ~4 MiB)
    GET /api/stream                  → Server-Sent Events stream of update deltas

ThreadingHTTPServer spawns a thread per request; SSE handlers hold a
thread for the connection lifetime, which is fine for a local viewer.
"""

from __future__ import annotations

import json
import math
import mimetypes
import queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from .scan import RunScanner


def _sanitize_for_json(obj: Any) -> Any:
    """Walk *obj* and replace non-finite floats (``NaN``, ``±Inf``) with ``None``.

    Python's ``json.dumps`` emits ``NaN``/``Infinity`` as bare tokens, which
    :func:`JSON.parse` in browsers rejects with ``Unexpected token 'N'``.
    Training metrics frequently contain non-finite values (early exploding
    losses, intentional ``inf`` upper bounds, etc.) so we need to emit valid
    JSON. ``None`` round-trips to ``null`` in JS.
    """
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _safe_dumps(obj: Any) -> str:
    """:func:`json.dumps` that produces valid JSON for non-finite floats."""
    return json.dumps(_sanitize_for_json(obj))


ASSETS_DIR = (Path(__file__).parent / "assets").resolve()


class _Handler(BaseHTTPRequestHandler):
    server_version = "spt-web/0.1"
    scanner: RunScanner = None  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Default logging is too noisy with SSE pings.
        pass

    def do_GET(self) -> None:  # noqa: N802
        try:
            url = urlparse(self.path)
            path, qs = url.path, parse_qs(url.query)

            if path in ("/", "/index.html"):
                self._serve_asset("index.html", "text/html; charset=utf-8")
            elif path.startswith("/assets/"):
                self._serve_asset(path[len("/assets/") :])
            elif path == "/api/runs":
                self._serve_json(self.scanner.runs_json())
            elif path == "/api/scan-status":
                self._serve_json(self.scanner.progress_json())
            elif path == "/api/metrics":
                run_id = qs.get("run_id", [None])[0]
                if not run_id:
                    self._serve_json({"error": "missing run_id"}, 400)
                    return
                # Pre-serialised + cached path: avoids re-running json.dumps
                # over a multi-MiB metrics dict on every request.
                body = self.scanner.metrics_json_bytes(run_id)
                if body is None:
                    self._serve_json({"error": "run not found"}, 404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/metrics-stream":
                run_id = qs.get("run_id", [None])[0]
                if not run_id:
                    self._serve_json({"error": "missing run_id"}, 400)
                    return
                self._serve_metrics_stream(run_id)
            elif path == "/api/media":
                run_id = qs.get("run_id", [None])[0]
                if not run_id:
                    self._serve_json({"error": "missing run_id"}, 400)
                    return
                data = self.scanner.media_json(run_id)
                if data is None:
                    self._serve_json({"error": "run not found"}, 404)
                    return
                self._serve_json(data)
            elif path == "/api/logs":
                run_id = qs.get("run_id", [None])[0]
                if not run_id:
                    self._serve_json({"error": "missing run_id"}, 400)
                    return
                data = self.scanner.logs_index(run_id)
                if data is None:
                    self._serve_json({"error": "run not found"}, 404)
                    return
                self._serve_json(data)
            elif path == "/api/log-content":
                run_id = qs.get("run_id", [None])[0]
                stream_id = qs.get("stream_id", [None])[0]
                if not run_id or not stream_id:
                    self.send_error(400, "missing run_id or stream_id")
                    return
                body = self.scanner.log_content(run_id, stream_id)
                if body is None:
                    self.send_error(404, "Not Found")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/media-file":
                run_id = qs.get("run_id", [None])[0]
                rel = qs.get("path", [None])[0]
                if not run_id or not rel:
                    self.send_error(400, "missing run_id or path")
                    return
                target = self.scanner.media_file_path(run_id, rel)
                if target is None:
                    self.send_error(404, "Not Found")
                    return
                self._serve_file(target)
            elif path == "/api/stream":
                self._serve_sse()
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self.send_error(404, "Not Found")
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ---- helpers ----

    def _serve_asset(self, name: str, ctype: Optional[str] = None) -> None:
        # Path traversal guard.
        target = (ASSETS_DIR / name).resolve()
        if not str(target).startswith(str(ASSETS_DIR)):
            self.send_error(403, "Forbidden")
            return
        if not target.is_file():
            self.send_error(404, "Not Found")
            return
        if ctype is None:
            guessed, _ = mimetypes.guess_type(name)
            ctype = guessed or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, target: Path) -> None:
        """Send a file's bytes with a guessed Content-Type and a long cache.

        Media files are content-addressed (path includes the step) so they
        don't change once written — safe to cache aggressively.
        """
        ctype, _ = mimetypes.guess_type(target.name)
        ctype = ctype or "application/octet-stream"
        size = target.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "public, max-age=86400, immutable")
        self.end_headers()
        with target.open("rb") as f:
            # Stream in chunks so very large videos don't blow up memory.
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _serve_json(self, obj: Any, status: int = 200) -> None:
        data = _safe_dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_metrics_stream(self, run_id: str) -> None:
        """Stream metrics as NDJSON over HTTP/1.1 chunked transfer-encoding.

        One JSON object per line; each object is either a metrics chunk
        (``{"chunk": N, "metrics": {...}}``) or the terminal ``{"done": true}``.
        Browsers consume this with ``fetch().body.getReader()`` and progressively
        merge chunks into the chart, so the user sees the first points within
        a few hundred ms instead of waiting for the whole CSV to parse.
        """
        # We probe the iterator's first value before sending headers so a 404
        # (run not found) can still be returned cleanly.
        gen = self.scanner.metrics_stream(run_id)
        try:
            first = next(gen)
        except StopIteration:
            first = None
        if first is None:
            self._serve_json({"error": "run not found"}, 404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        # Disable proxy buffering so chunks reach the browser as soon as we
        # flush; otherwise reverse proxies may coalesce them.
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def _write_chunk(line_bytes: bytes) -> None:
            # HTTP/1.1 chunked encoding frame: <hex-size>\r\n<data>\r\n
            self.wfile.write(f"{len(line_bytes):x}\r\n".encode("ascii"))
            self.wfile.write(line_bytes)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        try:
            _write_chunk(_safe_dumps(first).encode("utf-8") + b"\n")
            for item in gen:
                _write_chunk(_safe_dumps(item).encode("utf-8") + b"\n")
            # Terminating zero-length chunk.
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _serve_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        # Disable proxy buffering (nginx, etc.) just in case.
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        q = self.scanner.subscribe()
        try:
            self.wfile.write(b"event: ready\ndata: {}\n\n")
            self.wfile.flush()
            while True:
                try:
                    event = q.get(timeout=15.0)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                payload = (
                    f"event: {event['type']}\ndata: {_safe_dumps(event['data'])}\n\n"
                )
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.scanner.unsubscribe(q)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(
    directory: Path,
    host: str = "127.0.0.1",
    port: int = 4242,
    poll_interval: float = 1.0,
) -> None:
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")

    scanner = RunScanner(directory, poll_interval=poll_interval)
    scanner.start()

    class Handler(_Handler):
        pass

    Handler.scanner = scanner

    srv = _Server((host, port), Handler)
    print(f"[spt web] serving {directory}", flush=True)
    print(f"[spt web] http://{host}:{port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[spt web] shutting down", flush=True)
    finally:
        scanner.stop()
        srv.server_close()
