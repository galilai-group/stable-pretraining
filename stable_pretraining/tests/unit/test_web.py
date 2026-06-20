"""Unit tests for the spt web viewer backend.

  File-deletion safety
· metrics_json  returns {"metrics": {}} when metrics.csv is missing or
raises OSError on open (simulates file deleted mid-request)
· metrics_stream yields a "done" chunk instead of crashing
· log_content   returns None instead of crashing

Rename / edit robustness
· patch_run_meta raises ValueError for unknown keys and wrong value types
· patch_run_meta returns False for an unknown run_id
· patch_run_meta happy path persists changes to the sidecar on disk
· do_PATCH HTTP handler returns 400 for bad fields / types, 404 for
unknown runs, 500 when write_sidecar raises OSError, and 200 on success

Serialization and data shapes
· _serialize display_name resolution order
· runs_json correct list structure and fields
· metrics_json_bytes byte cache and NaN/Inf sanitization
· _safe_dumps NaN/Inf → null
· CSV edge cases (header-only, non-finite values)
· log_content truncation behaviour
· logs_index discovery and stream_id stability

HTTP GET endpoints
· All GET routes (/, /assets, /api/runs, /api/scan-status, /api/metrics,
/api/logs, /api/log-content) return expected status codes
· Path traversal blocked with 403
· Missing parameters return 400
"""

from __future__ import annotations

import http.client
import json
import math
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from stable_pretraining.web.scan import RunScanner, _Run, _safe_log_id
from stable_pretraining.web.server import _Handler, _safe_dumps

pytestmark = pytest.mark.unit


# ── shared helpers ────────────────────────────────────────────────────────────


def _make_scanner(root: Path) -> RunScanner:
    """Return a scanner with no background thread (poll_interval is irrelevant)."""
    return RunScanner(root, poll_interval=999)


def _inject_run(scanner: RunScanner, run_dir: Path, **sidecar_kw) -> str:
    """Write a sidecar and register the run in the scanner's in-memory map.

    Does not start the background thread.
    """
    from stable_pretraining.registry._sidecar import make_sidecar, write_sidecar

    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(run_dir.relative_to(scanner.root))
    sc = make_sidecar(run_id=run_id, run_dir=str(run_dir), **sidecar_kw)
    write_sidecar(run_dir, sc)
    run = _Run(
        run_id=run_id,
        run_dir=run_dir,
        sidecar_mtime=0.0,
        metrics_mtime=0.0,
        metrics_size=0,
        sidecar=sc,
    )
    with scanner._lock:
        scanner._runs[run_id] = run
    return run_id


@pytest.fixture
def web_server(tmp_path):
    """Spin up a real ThreadingHTTPServer on a free port.

    Yields ``(scanner, base_url)`` and tears down cleanly after the test.
    """
    sc = _make_scanner(tmp_path)

    class Handler(_Handler):
        pass

    Handler.scanner = sc

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield sc, f"http://127.0.0.1:{port}"
    srv.shutdown()
    srv.server_close()


def _do_patch(base_url: str, payload: dict) -> tuple[int, dict]:
    """Send a PATCH /api/run-meta and return (status_code, json_body)."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base_url + "/api/run-meta",
        data=body,
        method="PATCH",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ── TOCTOU safety ───────────────────────────────────────────────────────


class TestTOCTOUSafety:
    """Deleting files mid-request must never raise unhandled exceptions."""

    # ---- metrics_json --------------------------------------------------------

    def test_metrics_json_unknown_run_returns_none(self, tmp_path):
        sc = _make_scanner(tmp_path)
        assert sc.metrics_json("nonexistent") is None

    def test_metrics_json_missing_file_returns_empty(self, tmp_path):
        """metrics.csv never written — simulates deletion before request arrives."""
        sc = _make_scanner(tmp_path)
        run_id = _inject_run(sc, tmp_path / "r1")
        result = sc.metrics_json(run_id)
        assert result == {"metrics": {}}

    def test_metrics_json_oserror_on_open_returns_empty(self, tmp_path):
        """TOCTOU: is_file() passes, open() raises OSError.

        Verifies the try/except OSError guard around mpath.open() in
        metrics_json.
        """
        sc = _make_scanner(tmp_path)
        run_id = _inject_run(sc, tmp_path / "r1")
        (tmp_path / "r1" / "metrics.csv").write_text("step,loss\n1,0.5\n")

        original_open = Path.open

        def _raiser(self, *args, **kwargs):
            if self.name == "metrics.csv":
                raise FileNotFoundError("deleted mid-request")
            return original_open(self, *args, **kwargs)

        with patch.object(Path, "open", _raiser):
            result = sc.metrics_json(run_id)

        assert result == {"metrics": {}}

    def test_metrics_json_parses_valid_csv(self, tmp_path):
        """Sanity check: a real CSV is parsed correctly."""
        sc = _make_scanner(tmp_path)
        run_id = _inject_run(sc, tmp_path / "r1")
        (tmp_path / "r1" / "metrics.csv").write_text(
            "step,epoch,train/loss\n0,0,2.3\n10,0,2.1\n20,1,1.9\n"
        )
        result = sc.metrics_json(run_id)
        assert "train/loss" in result["metrics"]
        assert len(result["metrics"]["train/loss"]["y"]) == 3

    # ---- metrics_stream ------------------------------------------------------

    def test_metrics_stream_unknown_run_yields_none(self, tmp_path):
        sc = _make_scanner(tmp_path)
        chunks = list(sc.metrics_stream("nonexistent"))
        assert chunks == [None]

    def test_metrics_stream_missing_file_yields_done(self, tmp_path):
        """metrics.csv absent: stream must end cleanly."""
        sc = _make_scanner(tmp_path)
        run_id = _inject_run(sc, tmp_path / "r1")
        chunks = [c for c in sc.metrics_stream(run_id) if c is not None]
        assert any(c.get("done") for c in chunks)
        metrics_chunks = [c for c in chunks if "metrics" in c]
        assert all(c["metrics"] == {} for c in metrics_chunks)

    def test_metrics_stream_oserror_on_open_yields_done(self, tmp_path):
        """TOCTOU: is_file() passes, open() raises OSError.

        Verifies the try/except OSError guard around mpath.open() in
        metrics_stream cold-path.
        """
        sc = _make_scanner(tmp_path)
        run_id = _inject_run(sc, tmp_path / "r1")
        (tmp_path / "r1" / "metrics.csv").write_text("step,loss\n1,0.5\n")

        original_open = Path.open

        def _raiser(self, *args, **kwargs):
            if self.name == "metrics.csv":
                raise FileNotFoundError("deleted mid-request")
            return original_open(self, *args, **kwargs)

        with patch.object(Path, "open", _raiser):
            chunks = [c for c in sc.metrics_stream(run_id) if c is not None]

        assert any(c.get("done") for c in chunks)
        metrics_chunks = [c for c in chunks if "metrics" in c]
        assert all(c["metrics"] == {} for c in metrics_chunks)

    def test_metrics_stream_yields_data_then_done(self, tmp_path):
        """Sanity check: a real CSV streams data and ends with done."""
        sc = _make_scanner(tmp_path)
        run_id = _inject_run(sc, tmp_path / "r1")
        (tmp_path / "r1" / "metrics.csv").write_text(
            "step,val/acc\n" + "\n".join(f"{i},{i * 0.01:.3f}" for i in range(20))
        )
        chunks = [c for c in sc.metrics_stream(run_id) if c is not None]
        assert any(c.get("done") for c in chunks)
        all_y = []
        for c in chunks:
            if "metrics" in c and "val/acc" in c["metrics"]:
                all_y.extend(c["metrics"]["val/acc"]["y"])
        assert len(all_y) == 20

    # ---- log_content ---------------------------------------------------------

    def test_log_content_unknown_run_returns_none(self, tmp_path):
        sc = _make_scanner(tmp_path)
        assert sc.log_content("nonexistent", "train.out") is None

    def test_log_content_missing_file_returns_none(self, tmp_path):
        """Stream registered in _log_paths but file absent."""
        sc = _make_scanner(tmp_path)
        run_id = _inject_run(sc, tmp_path / "r1")
        missing = tmp_path / "r1" / "train.out"
        sc._log_paths[run_id] = {"train.out": missing}
        stream_id = _safe_log_id("train.out")
        result = sc.log_content(run_id, stream_id)
        assert result is None

    def test_log_content_oserror_on_open_returns_none(self, tmp_path):
        """TOCTOU: stat() passes, open() raises OSError.

        Verifies the try/except OSError guard around path.open("rb") in
        log_content.
        """
        sc = _make_scanner(tmp_path)
        run_id = _inject_run(sc, tmp_path / "r1")
        log_file = tmp_path / "r1" / "train.out"
        log_file.write_text("epoch 1 loss=2.30\n" * 10)
        stream_id = _safe_log_id("train.out")
        sc._log_paths[run_id] = {"train.out": log_file}

        original_open = Path.open

        def _raiser(self, *args, **kwargs):
            if self.name == "train.out":
                raise FileNotFoundError("deleted mid-request")
            return original_open(self, *args, **kwargs)

        with patch.object(Path, "open", _raiser):
            result = sc.log_content(run_id, stream_id)

        assert result is None

    def test_log_content_reads_existing_file(self, tmp_path):
        """Sanity check: log_content returns the file bytes."""
        sc = _make_scanner(tmp_path)
        run_id = _inject_run(sc, tmp_path / "r1")
        log_file = tmp_path / "r1" / "train.out"
        log_file.write_bytes(b"hello log\n")
        stream_id = _safe_log_id("train.out")
        sc._log_paths[run_id] = {"train.out": log_file}
        result = sc.log_content(run_id, stream_id)
        assert result == b"hello log\n"


# ── patch_run_meta validation ──────────────────────────────────────────


class TestPatchRunMetaValidation:
    """patch_run_meta enforces field-name and value-type constraints."""

    def _setup(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_id = _inject_run(sc, tmp_path / "r1")
        return sc, run_id

    # ---- field-name checks ---------------------------------------------------

    def test_unknown_field_raises(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        with pytest.raises(ValueError, match="unknown patch fields"):
            sc.patch_run_meta(run_id, {"bad_key": "x"})

    def test_mix_of_valid_and_unknown_raises(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        with pytest.raises(ValueError, match="unknown patch fields"):
            sc.patch_run_meta(run_id, {"display_name": "ok", "bad_key": "x"})

    # ---- display_name type checks --------------------------------------------

    def test_display_name_int_raises(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        with pytest.raises(ValueError, match="display_name"):
            sc.patch_run_meta(run_id, {"display_name": 123})

    def test_display_name_list_raises(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        with pytest.raises(ValueError, match="display_name"):
            sc.patch_run_meta(run_id, {"display_name": ["name"]})

    def test_display_name_bool_raises(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        with pytest.raises(ValueError, match="display_name"):
            sc.patch_run_meta(run_id, {"display_name": True})

    def test_display_name_string_accepted(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        assert sc.patch_run_meta(run_id, {"display_name": "my-run"}) is True

    def test_display_name_none_accepted(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        assert sc.patch_run_meta(run_id, {"display_name": None}) is True

    # ---- notes type checks ---------------------------------------------------

    def test_notes_int_raises(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        with pytest.raises(ValueError, match="notes"):
            sc.patch_run_meta(run_id, {"notes": 99})

    def test_notes_list_raises(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        with pytest.raises(ValueError, match="notes"):
            sc.patch_run_meta(run_id, {"notes": []})

    def test_notes_string_accepted(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        assert sc.patch_run_meta(run_id, {"notes": "a note"}) is True

    def test_notes_none_accepted(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        assert sc.patch_run_meta(run_id, {"notes": None}) is True

    # ---- tags type checks ----------------------------------------------------

    def test_tags_string_raises(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        with pytest.raises(ValueError, match="tags"):
            sc.patch_run_meta(run_id, {"tags": "not-a-list"})

    def test_tags_none_raises(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        with pytest.raises(ValueError, match="tags"):
            sc.patch_run_meta(run_id, {"tags": None})

    def test_tags_list_accepted(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        assert sc.patch_run_meta(run_id, {"tags": ["sweep", "lr-1e-3"]}) is True

    def test_tags_empty_list_accepted(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        assert sc.patch_run_meta(run_id, {"tags": []}) is True

    # ---- archived type checks ------------------------------------------------

    def test_archived_string_raises(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        with pytest.raises(ValueError, match="archived"):
            sc.patch_run_meta(run_id, {"archived": "yes"})

    def test_archived_int_raises(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        with pytest.raises(ValueError, match="archived"):
            sc.patch_run_meta(run_id, {"archived": 1})

    def test_archived_bool_accepted(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        assert sc.patch_run_meta(run_id, {"archived": True}) is True

    # ---- other edge cases ----------------------------------------------------

    def test_unknown_run_id_returns_false(self, tmp_path):
        sc = _make_scanner(tmp_path)
        assert sc.patch_run_meta("does-not-exist", {"display_name": "x"}) is False

    def test_all_valid_fields_accepted(self, tmp_path):
        sc, run_id = self._setup(tmp_path)
        result = sc.patch_run_meta(
            run_id,
            {
                "display_name": "experiment-v2",
                "notes": "increased weight decay",
                "tags": ["sweep", "wd-1e-2"],
                "archived": False,
            },
        )
        assert result is True

    def test_patch_persists_display_name_to_sidecar(self, tmp_path):
        """The new display_name must survive a sidecar re-read."""
        from stable_pretraining.registry._sidecar import read_sidecar, sidecar_path

        sc, run_id = self._setup(tmp_path)
        sc.patch_run_meta(run_id, {"display_name": "renamed"})
        run_dir = tmp_path / "r1"
        on_disk = read_sidecar(sidecar_path(run_dir))
        assert on_disk["display_name"] == "renamed"

    def test_patch_persists_tags_to_sidecar(self, tmp_path):
        from stable_pretraining.registry._sidecar import read_sidecar, sidecar_path

        sc, run_id = self._setup(tmp_path)
        sc.patch_run_meta(run_id, {"tags": ["a", "b"]})
        on_disk = read_sidecar(sidecar_path(tmp_path / "r1"))
        assert on_disk["tags"] == ["a", "b"]

    def test_patch_persists_notes_to_sidecar(self, tmp_path):
        from stable_pretraining.registry._sidecar import read_sidecar, sidecar_path

        sc, run_id = self._setup(tmp_path)
        sc.patch_run_meta(run_id, {"notes": "my experiment notes"})
        on_disk = read_sidecar(sidecar_path(tmp_path / "r1"))
        assert on_disk["notes"] == "my experiment notes"

    def test_patch_persists_archived_to_sidecar(self, tmp_path):
        from stable_pretraining.registry._sidecar import read_sidecar, sidecar_path

        sc, run_id = self._setup(tmp_path)
        sc.patch_run_meta(run_id, {"archived": True})
        on_disk = read_sidecar(sidecar_path(tmp_path / "r1"))
        assert on_disk["archived"] is True

    def test_patch_updates_in_memory_sidecar(self, tmp_path):
        """In-memory run.sidecar must reflect the patch immediately."""
        sc, run_id = self._setup(tmp_path)
        sc.patch_run_meta(run_id, {"display_name": "live-update"})
        with sc._lock:
            run = sc._runs[run_id]
        assert run.sidecar["display_name"] == "live-update"


# ── HTTP PATCH handler ─────────────────────────────────────────────────


class TestPatchHTTPHandler:
    """do_PATCH must return clean JSON for every outcome."""

    def test_happy_path_returns_200(self, tmp_path, web_server):
        scanner, base = web_server
        run_id = _inject_run(scanner, tmp_path / "r1")
        status, body = _do_patch(base, {"run_id": run_id, "display_name": "new-name"})
        assert status == 200
        assert body == {"ok": True}

    def test_display_name_persisted_after_patch(self, tmp_path, web_server):
        from stable_pretraining.registry._sidecar import read_sidecar, sidecar_path

        scanner, base = web_server
        run_id = _inject_run(scanner, tmp_path / "r2")
        _do_patch(base, {"run_id": run_id, "display_name": "via-http"})
        on_disk = read_sidecar(sidecar_path(tmp_path / "r2"))
        assert on_disk["display_name"] == "via-http"

    def test_unknown_run_returns_404(self, web_server):
        _, base = web_server
        status, body = _do_patch(base, {"run_id": "no-such-run", "display_name": "x"})
        assert status == 404
        assert "error" in body

    def test_missing_run_id_returns_400(self, web_server):
        _, base = web_server
        status, body = _do_patch(base, {"display_name": "x"})
        assert status == 400
        assert "error" in body

    def test_unknown_field_returns_400(self, tmp_path, web_server):
        scanner, base = web_server
        run_id = _inject_run(scanner, tmp_path / "r3")
        status, body = _do_patch(base, {"run_id": run_id, "bad_field": "x"})
        assert status == 400
        assert "error" in body

    def test_wrong_type_display_name_returns_400(self, tmp_path, web_server):
        scanner, base = web_server
        run_id = _inject_run(scanner, tmp_path / "r4")
        status, body = _do_patch(base, {"run_id": run_id, "display_name": 999})
        assert status == 400
        assert "error" in body

    def test_wrong_type_tags_returns_400(self, tmp_path, web_server):
        scanner, base = web_server
        run_id = _inject_run(scanner, tmp_path / "r5")
        status, body = _do_patch(base, {"run_id": run_id, "tags": "not-a-list"})
        assert status == 400
        assert "error" in body

    def test_oserror_from_write_returns_500(self, tmp_path, web_server):
        """write_sidecar raising OSError must not crash the thread — returns 500."""
        scanner, base = web_server
        run_id = _inject_run(scanner, tmp_path / "r6")
        with patch(
            "stable_pretraining.web.scan.write_sidecar",
            side_effect=OSError("simulated disk error"),
        ):
            status, body = _do_patch(base, {"run_id": run_id, "display_name": "x"})
        assert status == 500
        assert "error" in body

    def test_invalid_json_body_returns_400(self, web_server):
        _, base = web_server
        req = urllib.request.Request(
            base + "/api/run-meta",
            data=b"{not valid json",
            method="PATCH",
            headers={"Content-Type": "application/json", "Content-Length": "15"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                status = resp.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert status == 400

    def test_body_too_large_returns_400(self, web_server):
        """Content-Length > 64 KiB must be rejected before reading the body."""
        _, base = web_server
        port = int(base.rsplit(":", 1)[-1])
        big_len = 64 * 1024 + 1
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.putrequest("PATCH", "/api/run-meta")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(big_len))
        conn.endheaders()
        # Don't send the body — the server rejects on the header alone and
        # returns 400 without calling rfile.read(), so no body is needed.
        resp = conn.getresponse()
        status = resp.status
        resp.read()
        conn.close()
        assert status == 400

    def test_non_dict_json_body_returns_400(self, web_server):
        """A valid JSON array or scalar body (not an object) must return 400."""
        _, base = web_server
        body = json.dumps([1, 2, 3]).encode()
        req = urllib.request.Request(
            base + "/api/run-meta",
            data=body,
            method="PATCH",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                status = resp.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert status == 400

    def test_run_id_non_string_returns_400(self, tmp_path, web_server):
        """run_id present but not a string must return 400."""
        scanner, base = web_server
        _inject_run(scanner, tmp_path / "r7")
        status, body = _do_patch(base, {"run_id": 123, "display_name": "x"})
        assert status == 400
        assert "error" in body

    def test_run_id_empty_string_returns_400(self, web_server):
        """run_id present but empty string must return 400."""
        _, base = web_server
        status, body = _do_patch(base, {"run_id": "", "display_name": "x"})
        assert status == 400
        assert "error" in body

    def test_wrong_path_returns_404(self, web_server):
        _, base = web_server
        req = urllib.request.Request(
            base + "/api/no-such-endpoint",
            data=b"{}",
            method="PATCH",
            headers={"Content-Type": "application/json", "Content-Length": "2"},
        )
        try:
            with urllib.request.urlopen(req):
                pass
        except urllib.error.HTTPError as exc:
            assert exc.code == 404


# ── TestPatchSSEPublish ───────────────────────────────────────────────────────


class TestPatchSSEPublish:
    """patch_run_meta must fire an SSE update on success and stay silent on failure."""

    def test_publishes_update_on_success(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_id = _inject_run(sc, tmp_path / "r1")
        q = sc.subscribe()
        try:
            sc.patch_run_meta(run_id, {"display_name": "new-name"})
        finally:
            sc.unsubscribe(q)
        assert not q.empty()
        event = q.get_nowait()
        assert event["type"] == "update"
        assert run_id in event["data"]["changed"]
        assert event["data"]["removed"] == []

    def test_does_not_publish_on_unknown_run(self, tmp_path):
        sc = _make_scanner(tmp_path)
        q = sc.subscribe()
        try:
            result = sc.patch_run_meta("no-such-run", {"display_name": "x"})
        finally:
            sc.unsubscribe(q)
        assert result is False
        assert q.empty()


# ── GET helpers ───────────────────────────────────────────────────────────────


def _do_get(base_url: str, path: str) -> tuple[int, bytes, str]:
    """GET *path* and return (status, body, content-type)."""
    try:
        with urllib.request.urlopen(base_url + path) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "")


def _raw_get(base_url: str, path: str) -> int:
    """GET with no URL normalisation — needed for path-traversal tests."""
    port = int(base_url.rsplit(":", 1)[-1])
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    status = resp.status
    resp.read()
    conn.close()
    return status


# ── TestSafeDumps ─────────────────────────────────────────────────────────────


class TestSafeDumps:
    """Tests for the _safe_dumps NaN/Inf sanitization helper."""

    def test_nan_becomes_null(self):
        assert json.loads(_safe_dumps(float("nan"))) is None

    def test_pos_inf_becomes_null(self):
        assert json.loads(_safe_dumps(float("inf"))) is None

    def test_neg_inf_becomes_null(self):
        assert json.loads(_safe_dumps(float("-inf"))) is None

    def test_nested_nan_in_list(self):
        result = json.loads(_safe_dumps({"y": [1.0, float("nan"), 3.0]}))
        assert result == {"y": [1.0, None, 3.0]}

    def test_nested_inf_in_dict(self):
        result = json.loads(_safe_dumps({"loss": float("inf")}))
        assert result == {"loss": None}

    def test_finite_floats_unchanged(self):
        assert json.loads(_safe_dumps(1.5)) == pytest.approx(1.5)

    def test_integers_unchanged(self):
        assert json.loads(_safe_dumps(42)) == 42

    def test_nested_dict_and_list(self):
        obj = {"a": [float("nan"), {"b": float("inf"), "c": 2.0}]}
        result = json.loads(_safe_dumps(obj))
        assert result == {"a": [None, {"b": None, "c": 2.0}]}


# ── TestSerialize ─────────────────────────────────────────────────────────────


class TestSerialize:
    """Tests for RunScanner._serialize display_name resolution and field shapes."""

    def test_explicit_display_name_wins(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "run1"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = str(run_dir.relative_to(sc.root))
        from stable_pretraining.registry._sidecar import make_sidecar, write_sidecar

        sid = make_sidecar(run_id=run_id, run_dir=str(run_dir))
        sid["display_name"] = "My Run"
        write_sidecar(run_dir, sid)
        run = _Run(
            run_id=run_id,
            run_dir=run_dir,
            sidecar_mtime=0.0,
            metrics_mtime=0.0,
            metrics_size=0,
            sidecar=sid,
        )
        with sc._lock:
            sc._runs[run_id] = run
        runs = sc.runs_json()
        assert len(runs) == 1
        assert runs[0]["display_name"] == "My Run"

    def test_falls_back_to_last_path_component(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "parent" / "child_run"
        _inject_run(sc, run_dir)
        runs = sc.runs_json()
        assert runs[0]["display_name"] == "child_run"

    def test_falls_back_to_full_run_id_when_no_slash(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "flat"
        _inject_run(sc, run_dir)
        runs = sc.runs_json()
        assert runs[0]["display_name"] == "flat"

    def test_has_media_false_when_empty(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "run_nomedia"
        _inject_run(sc, run_dir)
        runs = sc.runs_json()
        assert runs[0]["has_media"] is False

    def test_has_media_true_when_media_size_nonzero(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "run_media"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = str(run_dir.relative_to(sc.root))
        # Construct _Run directly with a non-zero media_size.
        from stable_pretraining.registry._sidecar import make_sidecar, write_sidecar

        sid = make_sidecar(run_id=run_id, run_dir=str(run_dir))
        write_sidecar(run_dir, sid)
        run = _Run(
            run_id=run_id,
            run_dir=run_dir,
            sidecar_mtime=0.0,
            metrics_mtime=0.0,
            metrics_size=0,
            media_mtime=1.0,
            media_size=512,
            sidecar=sid,
        )
        with sc._lock:
            sc._runs[run_id] = run
        runs = sc.runs_json()
        assert runs[0]["has_media"] is True

    def test_heartbeat_at_none_when_no_file(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "run_nohb"
        _inject_run(sc, run_dir)
        runs = sc.runs_json()
        assert runs[0]["heartbeat_at"] is None

    def test_ended_at_roundtrip(self, tmp_path):
        from stable_pretraining.registry._sidecar import make_sidecar, write_sidecar

        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "run_ended"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = str(run_dir.relative_to(sc.root))
        ended_ts = 1_700_000_000.0
        sid = make_sidecar(run_id=run_id, run_dir=str(run_dir), ended_at=ended_ts)
        write_sidecar(run_dir, sid)
        run = _Run(
            run_id=run_id,
            run_dir=run_dir,
            sidecar_mtime=0.0,
            metrics_mtime=0.0,
            metrics_size=0,
            sidecar=sid,
        )
        with sc._lock:
            sc._runs[run_id] = run
        runs = sc.runs_json()
        assert runs[0]["ended_at"] == pytest.approx(ended_ts)

    def test_heartbeat_at_roundtrip(self, tmp_path):
        from stable_pretraining.registry._sidecar import (
            make_sidecar,
            touch_heartbeat,
            write_sidecar,
        )

        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "run_hb"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = str(run_dir.relative_to(sc.root))
        sid = make_sidecar(run_id=run_id, run_dir=str(run_dir))
        write_sidecar(run_dir, sid)
        touch_heartbeat(run_dir)
        hb_mtime = (run_dir / "heartbeat").stat().st_mtime
        run = _Run(
            run_id=run_id,
            run_dir=run_dir,
            sidecar_mtime=0.0,
            metrics_mtime=0.0,
            metrics_size=0,
            sidecar=sid,
        )
        with sc._lock:
            sc._runs[run_id] = run
        runs = sc.runs_json()
        assert runs[0]["heartbeat_at"] == pytest.approx(hb_mtime)


# ── TestRunsJson ──────────────────────────────────────────────────────────────


class TestRunsJson:
    """Tests for RunScanner.runs_json list structure and field correctness."""

    def test_empty_when_no_runs(self, tmp_path):
        sc = _make_scanner(tmp_path)
        assert sc.runs_json() == []

    def test_returns_expected_keys(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "r"
        _inject_run(sc, run_dir, tags=["v1"], notes="hello")
        runs = sc.runs_json()
        assert len(runs) == 1
        r = runs[0]
        for key in (
            "run_id",
            "run_dir",
            "display_name",
            "status",
            "created_at",
            "ended_at",
            "heartbeat_at",
            "tags",
            "notes",
            "hparams",
            "summary",
            "checkpoint_path",
            "metrics_size",
            "has_media",
        ):
            assert key in r, f"missing key: {key}"

    def test_tags_and_notes_roundtrip(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "r2"
        _inject_run(sc, run_dir, tags=["a", "b"], notes="my note")
        r = sc.runs_json()[0]
        assert r["tags"] == ["a", "b"]
        assert r["notes"] == "my note"


# ── TestMetricsCache ──────────────────────────────────────────────────────────


class TestMetricsCache:
    """Tests for metrics_json mtime/size cache hit and invalidation."""

    def _write_csv(self, path: Path, rows: list[list]) -> None:
        import csv

        with path.open("w", newline="") as f:
            csv.writer(f).writerows(rows)

    def test_cache_hit_avoids_reparse(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "r"
        run_id = _inject_run(sc, run_dir)
        csv_path = run_dir / "metrics.csv"
        self._write_csv(csv_path, [["step", "loss"], ["1", "0.5"], ["2", "0.3"]])

        first = sc.metrics_json(run_id)
        # Second call must hit the mtime/size cache without re-opening the file.
        # Patch Path.open (the actual call site) — builtins.open is not used by
        # mpath.open() so patching it would leave the cache test with no teeth.
        with patch.object(Path, "open", side_effect=AssertionError("file re-opened")):
            second = sc.metrics_json(run_id)
        assert first == second

    def test_cache_invalidated_on_file_change(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "r2"
        run_id = _inject_run(sc, run_dir)
        csv_path = run_dir / "metrics.csv"
        self._write_csv(csv_path, [["step", "loss"], ["1", "0.5"]])

        first = sc.metrics_json(run_id)
        assert len(first["metrics"]["loss"]["y"]) == 1

        # Overwrite the file; bump mtime by writing fresh content.
        self._write_csv(
            csv_path,
            [["step", "loss"], ["1", "0.5"], ["2", "0.3"], ["3", "0.1"]],
        )
        # Touch so the mtime definitely changes (some filesystems have 1 s resolution).
        import os
        import time

        os.utime(csv_path, (time.time() + 2, time.time() + 2))

        second = sc.metrics_json(run_id)
        assert len(second["metrics"]["loss"]["y"]) == 3


# ── TestMetricsJsonBytes ──────────────────────────────────────────────────────


class TestMetricsJsonBytes:
    """Tests for metrics_json_bytes byte cache and NaN/Inf sanitization."""

    def test_returns_valid_utf8_json(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "r"
        run_id = _inject_run(sc, run_dir)
        (run_dir / "metrics.csv").write_text("step,loss\n1,0.5\n2,0.3\n")
        body = sc.metrics_json_bytes(run_id)
        assert isinstance(body, bytes)
        parsed = json.loads(body.decode("utf-8"))
        assert "metrics" in parsed

    def test_nan_inf_serialised_as_null(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "r2"
        run_id = _inject_run(sc, run_dir)
        (run_dir / "metrics.csv").write_text("step,loss\n1,nan\n2,inf\n3,0.5\n")
        body = sc.metrics_json_bytes(run_id)
        parsed = json.loads(body.decode("utf-8"))
        y_vals = parsed["metrics"]["loss"]["y"]
        assert y_vals == [None, None, pytest.approx(0.5)]

    def test_byte_cache_reused_on_second_call(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "r3"
        run_id = _inject_run(sc, run_dir)
        (run_dir / "metrics.csv").write_text("step,loss\n1,0.5\n")
        first = sc.metrics_json_bytes(run_id)
        second = sc.metrics_json_bytes(run_id)
        assert first is second  # same bytes object from cache

    def test_unknown_run_returns_none(self, tmp_path):
        sc = _make_scanner(tmp_path)
        assert sc.metrics_json_bytes("no-such-run") is None


# ── TestCsvEdgeCases ──────────────────────────────────────────────────────────


class TestCsvEdgeCases:
    """Tests for CSV parsing edge cases: empty files and non-finite float values."""

    def test_header_only_csv_returns_empty_metrics(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "r"
        run_id = _inject_run(sc, run_dir)
        (run_dir / "metrics.csv").write_text("step,loss\n")
        result = sc.metrics_json(run_id)
        assert result == {"metrics": {}}

    def test_nan_float_in_csv_does_not_crash(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "r2"
        run_id = _inject_run(sc, run_dir)
        (run_dir / "metrics.csv").write_text("step,loss\n1,nan\n2,inf\n3,-inf\n4,0.2\n")
        result = sc.metrics_json(run_id)
        assert "loss" in result["metrics"]
        y = result["metrics"]["loss"]["y"]
        assert len(y) == 4
        assert math.isnan(y[0])
        assert math.isinf(y[1]) and y[1] > 0
        assert math.isinf(y[2]) and y[2] < 0
        assert y[3] == pytest.approx(0.2)

    def test_nan_inf_sanitised_in_bytes(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "r3"
        run_id = _inject_run(sc, run_dir)
        (run_dir / "metrics.csv").write_text("step,loss\n1,nan\n2,inf\n")
        body = sc.metrics_json_bytes(run_id)
        parsed = json.loads(body.decode("utf-8"))
        assert all(v is None for v in parsed["metrics"]["loss"]["y"])


# ── TestLogsIndex ─────────────────────────────────────────────────────────────


class TestLogsIndex:
    """Tests for RunScanner.logs_index discovery and stream_id stability."""

    def test_unknown_run_returns_none(self, tmp_path):
        sc = _make_scanner(tmp_path)
        assert sc.logs_index("no-such") is None

    def test_discovers_out_and_err_files(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "r"
        run_id = _inject_run(sc, run_dir)
        (run_dir / "train.out").write_text("stdout")
        (run_dir / "train.err").write_text("stderr")
        result = sc.logs_index(run_id)
        assert result is not None
        names = {s["name"] for s in result["streams"]}
        assert any("train.out" in n or n == "train.out" for n in names)
        assert any("train.err" in n or n == "train.err" for n in names)

    def test_kind_field_correct(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "r2"
        run_id = _inject_run(sc, run_dir)
        (run_dir / "job.out").write_text("o")
        (run_dir / "job.err").write_text("e")
        result = sc.logs_index(run_id)
        kinds = {s["name"].split("/")[-1]: s["kind"] for s in result["streams"]}
        out_kinds = {k: v for k, v in kinds.items() if "out" in k}
        err_kinds = {k: v for k, v in kinds.items() if "err" in k}
        assert all(v == "out" for v in out_kinds.values())
        assert all(v == "err" for v in err_kinds.values())

    def test_stream_id_is_stable_slug(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "r3"
        run_id = _inject_run(sc, run_dir)
        (run_dir / "run.out").write_text("x")
        r1 = sc.logs_index(run_id)
        r2 = sc.logs_index(run_id)
        ids1 = {s["stream_id"] for s in r1["streams"]}
        ids2 = {s["stream_id"] for s in r2["streams"]}
        assert ids1 == ids2  # stable across calls


# ── TestLogContentTruncation ──────────────────────────────────────────────────


class TestLogContentTruncation:
    """Tests for RunScanner.log_content tail-truncation behaviour."""

    def test_small_file_returned_in_full(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "r"
        run_id = _inject_run(sc, run_dir)
        content = b"hello\nworld\n"
        log_file = run_dir / "job.out"
        log_file.write_bytes(content)
        sc.logs_index(run_id)  # populate path map
        stream_id = _safe_log_id("job.out")
        result = sc.log_content(run_id, stream_id)
        assert result == content

    def test_large_file_returns_tail(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "r2"
        run_id = _inject_run(sc, run_dir)
        # Write 3 × max_bytes worth of lines so truncation is guaranteed.
        max_bytes = 1024
        line = b"x" * 63 + b"\n"  # 64 bytes per line
        lines = line * (max_bytes * 3 // len(line) + 1)
        log_file = run_dir / "job.out"
        log_file.write_bytes(lines)
        sc.logs_index(run_id)
        stream_id = _safe_log_id("job.out")
        result = sc.log_content(run_id, stream_id, max_bytes=max_bytes)
        assert result is not None
        assert len(result) <= max_bytes
        # The result should end with the last lines of the file.
        assert lines.endswith(result)

    def test_unknown_run_returns_none(self, tmp_path):
        sc = _make_scanner(tmp_path)
        assert sc.log_content("no-run", "no-stream") is None

    def test_unknown_stream_returns_none(self, tmp_path):
        sc = _make_scanner(tmp_path)
        run_dir = tmp_path / "r3"
        run_id = _inject_run(sc, run_dir)
        sc.logs_index(run_id)
        assert sc.log_content(run_id, "nonexistent-stream-id") is None


# ── TestGetEndpoints ──────────────────────────────────────────────────────────


class TestGetEndpoints:
    """Integration tests for all HTTP GET routes served by the web viewer."""

    def test_root_returns_200_html(self, web_server):
        _, base = web_server
        status, body, ctype = _do_get(base, "/")
        assert status == 200
        assert "text/html" in ctype
        assert b"<html" in body.lower() or b"<!doctype" in body.lower()

    def test_assets_app_js_returns_200(self, web_server):
        _, base = web_server
        status, body, ctype = _do_get(base, "/assets/app.js")
        assert status == 200
        assert len(body) > 0

    def test_unknown_asset_returns_404(self, web_server):
        _, base = web_server
        status, _, _ = _do_get(base, "/assets/does-not-exist.xyz")
        assert status == 404

    def test_path_traversal_returns_403(self, web_server):
        _, base = web_server
        status = _raw_get(base, "/assets/../../etc/passwd")
        assert status == 403

    def test_api_runs_returns_200_json_list(self, web_server):
        sc, base = web_server
        run_dir = sc.root / "r"
        _inject_run(sc, run_dir)
        status, body, ctype = _do_get(base, "/api/runs")
        assert status == 200
        assert "application/json" in ctype
        parsed = json.loads(body)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["run_id"] == "r"

    def test_api_scan_status_returns_phase(self, web_server):
        _, base = web_server
        status, body, ctype = _do_get(base, "/api/scan-status")
        assert status == 200
        parsed = json.loads(body)
        assert "phase" in parsed

    def test_api_metrics_missing_run_id_returns_400(self, web_server):
        _, base = web_server
        status, _, _ = _do_get(base, "/api/metrics")
        assert status == 400

    def test_api_metrics_unknown_run_returns_404(self, web_server):
        _, base = web_server
        status, _, _ = _do_get(base, "/api/metrics?run_id=ghost")
        assert status == 404

    def test_api_metrics_valid_run_returns_200(self, web_server):
        sc, base = web_server
        run_dir = sc.root / "rm"
        run_id = _inject_run(sc, run_dir)
        (run_dir / "metrics.csv").write_text("step,loss\n1,0.5\n")
        status, body, ctype = _do_get(base, f"/api/metrics?run_id={run_id}")
        assert status == 200
        parsed = json.loads(body)
        assert "metrics" in parsed

    def test_api_logs_missing_run_id_returns_400(self, web_server):
        _, base = web_server
        status, _, _ = _do_get(base, "/api/logs")
        assert status == 400

    def test_api_logs_unknown_run_returns_404(self, web_server):
        _, base = web_server
        status, _, _ = _do_get(base, "/api/logs?run_id=ghost")
        assert status == 404

    def test_api_logs_valid_run_returns_streams(self, web_server):
        sc, base = web_server
        run_dir = sc.root / "rl"
        run_id = _inject_run(sc, run_dir)
        status, body, _ = _do_get(base, f"/api/logs?run_id={run_id}")
        assert status == 200
        parsed = json.loads(body)
        assert "streams" in parsed

    def test_api_log_content_missing_params_returns_400(self, web_server):
        _, base = web_server
        status, _, _ = _do_get(base, "/api/log-content")
        assert status == 400

    def test_api_log_content_valid_returns_200_text(self, web_server):
        sc, base = web_server
        run_dir = sc.root / "rc"
        run_id = _inject_run(sc, run_dir)
        log_file = run_dir / "job.out"
        log_file.write_text("hello log")
        # Populate the path map via logs_index first.
        sc.logs_index(run_id)
        stream_id = _safe_log_id("job.out")
        status, body, ctype = _do_get(
            base, f"/api/log-content?run_id={run_id}&stream_id={stream_id}"
        )
        assert status == 200
        assert "text/plain" in ctype
        assert b"hello log" in body

    def test_unknown_path_returns_404(self, web_server):
        _, base = web_server
        status, _, _ = _do_get(base, "/no-such-route")
        assert status == 404
