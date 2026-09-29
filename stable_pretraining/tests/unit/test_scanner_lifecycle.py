"""Exercise real run discovery, changing files, and streamed viewer data."""

import json
import os
import queue
import shutil

import pytest

from stable_pretraining.registry._sidecar import write_sidecar
from stable_pretraining.web.scan import RunScanner

pytestmark = pytest.mark.unit


def _run(root, name="run", **fields):
    path = root / name
    write_sidecar(path, {"status": "running", "summary": {"loss": 1}, **fields})
    return path


def _rewrite(path, content):
    previous = path.stat().st_mtime
    path.write_text(content)
    os.utime(path, (previous + 2, previous + 2))


def test_scan_discovers_nested_runs_updates_and_removes_them(tmp_path):
    run = _run(tmp_path, "sweep/deep/run")
    scanner = RunScanner(tmp_path)
    events = scanner.subscribe()
    assert scanner._scan(initial=True) == (["sweep/deep/run"], [])
    assert scanner.progress_json()["done"] == 1
    assert events.get_nowait()["type"] == "progress"
    assert scanner._scan() == ([], [])
    assert scanner.runs_json()[0]["summary"] == {"loss": 1}
    (run / "metrics.csv").write_text("step,loss\n0,2\n")
    (run / "media.jsonl").write_text('{"step":0,"path":"media/x.png"}\n')
    assert scanner._scan() == (["sweep/deep/run"], [])
    assert scanner.runs_json()[0]["has_media"]
    assert scanner.metrics_json("sweep/deep/run")["metrics"]["loss"]["y"] == [2]
    _rewrite(run / "sidecar.json", '{"status":"completed","summary":{"loss":0.5}}')
    assert scanner._scan() == (["sweep/deep/run"], [])
    assert scanner.runs_json()[0]["status"] == "completed"
    shutil.rmtree(run)
    assert scanner._scan() == ([], ["sweep/deep/run"])
    assert scanner.runs_json() == []
    assert scanner.metrics_json("sweep/deep/run") is None
    assert "sweep/deep/run" not in scanner._metrics_cache
    assert run not in scanner._walk_cache
    scanner.unsubscribe(events)


@pytest.mark.parametrize("bad_content", ['{"status":', "[]", "null", '"text"'])
def test_bad_sidecar_preserves_last_valid_run_and_recovers(tmp_path, bad_content):
    run = _run(tmp_path)
    scanner = RunScanner(tmp_path)
    scanner._scan()
    _rewrite(run / "sidecar.json", bad_content)
    assert scanner._scan() == ([], [])
    assert scanner.runs_json()[0]["summary"] == {"loss": 1}
    _rewrite(run / "sidecar.json", '{"status":"completed"}')
    assert scanner._scan() == (["run"], [])
    assert scanner.runs_json()[0]["status"] == "completed"


def test_discovery_ignores_symlink_loops_and_handles_missing_root(tmp_path):
    run = _run(tmp_path)
    (run / "loop").symlink_to(tmp_path, target_is_directory=True)
    scanner = RunScanner(tmp_path)
    assert scanner._scan() == (["run"], [])
    assert RunScanner(tmp_path / "missing")._scan() == ([], [])
    assert RunScanner(run)._scan() == (["run"], [])


def test_disappearing_sidecar_during_scan_does_not_abort_others(tmp_path, monkeypatch):
    good = _run(tmp_path)
    scanner = RunScanner(tmp_path)
    monkeypatch.setattr(
        scanner,
        "_parallel_walk",
        lambda *a, **kw: [tmp_path / "gone" / "sidecar.json", good / "sidecar.json"],
    )
    assert scanner._scan() == (["run"], [])


def test_background_scanner_publishes_initial_results_and_stops(tmp_path):
    _run(tmp_path)
    scanner = RunScanner(tmp_path, poll_interval=0.01)
    events = scanner.subscribe()
    scanner.start()
    try:
        observed = []
        for _ in range(10):
            event = events.get(timeout=5)
            observed.append(event)
            if event["type"] == "progress" and event["data"]["initial_done"]:
                break
        assert any(
            e["type"] == "update" and e["data"]["changed"] == ["run"] for e in observed
        )
        assert scanner.progress_json()["initial_done"]
        _run(tmp_path, "second")
        event = events.get(timeout=5)
        assert event == {
            "type": "update",
            "data": {"changed": ["second"], "removed": []},
        }
    finally:
        scanner.stop()
    assert not scanner._thread.is_alive()


def test_slow_subscriber_does_not_block_others(tmp_path):
    scanner = RunScanner(tmp_path)
    slow = scanner.subscribe()
    for _ in range(slow.maxsize):
        slow.put_nowait({})
    fast = scanner.subscribe()
    scanner._publish("update", {"changed": ["run"]})
    assert fast.get_nowait()["data"] == {"changed": ["run"]}
    scanner.unsubscribe(fast)
    scanner._publish("update", {})
    with pytest.raises(queue.Empty):
        fast.get_nowait()


def _join_chunks(chunks):
    result = {}
    assert chunks[-1] == {"done": True}
    for number, chunk in enumerate(chunks[:-1]):
        assert chunk["chunk"] == number
        for name, values in chunk["metrics"].items():
            metric = result.setdefault(name, {"step": [], "epoch": [], "y": []})
            for key, values in values.items():
                metric[key].extend(values)
    return {"metrics": result}


def test_streaming_cold_and_warm_cache_preserve_all_points(tmp_path):
    run = _run(tmp_path)
    count = 6200
    rows = [f"{i},,{i * 2},{i * 3}" for i in range(count)]
    (run / "metrics.csv").write_text("step,epoch,loss,acc\n" + "\n".join(rows))
    scanner = RunScanner(tmp_path)
    scanner._scan()
    cold = list(scanner.metrics_stream("run"))
    warm = list(scanner.metrics_stream("run"))
    assert len(cold) > 2 and len(warm) > 2
    expected = {
        "metrics": {
            name: {
                "step": list(range(count)),
                "epoch": [None] * count,
                "y": [i * scale for i in range(count)],
            }
            for name, scale in [("loss", 2), ("acc", 3)]
        }
    }
    assert _join_chunks(cold) == _join_chunks(warm) == expected
    assert scanner.metrics_json("run") == expected
    assert json.loads(scanner.metrics_json_bytes("run")) == expected
    _rewrite(run / "metrics.csv", "step,loss\n8,0.25\n")
    assert scanner.metrics_json("run")["metrics"] == {
        "loss": {"step": [8.0], "epoch": [None], "y": [0.25]}
    }
    assert json.loads(scanner.metrics_json_bytes("run"))["metrics"]["loss"]["y"] == [
        0.25
    ]


@pytest.mark.parametrize(
    "content", ["", "step,loss\n", "step,epoch,loss,other\nbad,,3,no\n,,4\n2,1,,9\n"]
)
def test_stream_and_bulk_parser_agree_on_empty_and_incomplete_rows(tmp_path, content):
    run = _run(tmp_path)
    (run / "metrics.csv").write_text(content)
    scanner = RunScanner(tmp_path)
    scanner._scan()
    streamed = _join_chunks(list(scanner.metrics_stream("run")))
    scanner._metrics_cache.clear()
    assert scanner.metrics_json("run") == streamed
    if "bad" in content:
        assert streamed["metrics"]["loss"] == {
            "step": [0, 1],
            "epoch": [None, None],
            "y": [3, 4],
        }


def test_media_read_skips_partial_records_and_paths_cannot_escape(tmp_path):
    run = _run(tmp_path)
    media = run / "media"
    media.mkdir()
    image = media / "plot.png"
    image.write_bytes(b"image")
    outside = tmp_path / "private.txt"
    outside.write_text("private")
    (media / "escape.png").symlink_to(outside)
    scanner = RunScanner(tmp_path)
    scanner._scan()
    assert scanner.media_json("unknown") is None
    assert scanner.media_json("run") == {"events": []}
    event = {"step": 1, "path": "media/plot.png", "type": "image"}
    (run / "media.jsonl").write_text("\n" + json.dumps(event) + '\n{"step":')
    assert scanner.media_json("run") == {"events": [event]}
    assert scanner.media_file_path("run", "media/plot.png") == image
    for invalid in [
        "",
        "/etc/passwd",
        "../private.txt",
        "sidecar.json",
        "media",
        "media/missing.png",
        "media/escape.png",
    ]:
        assert scanner.media_file_path("run", invalid) is None
    assert scanner.media_file_path("unknown", "media/plot.png") is None


@pytest.mark.parametrize(
    "task_config", [{"slurm": {"task_id": 2}}, {"slurm.task_id": 2}, {}]
)
def test_submitit_logs_discover_ranks_and_disambiguate_names(tmp_path, task_config):
    output = tmp_path / "outputs"
    output.mkdir()
    run = _run(
        tmp_path,
        "run_2",
        tags=["sweep:100"],
        hparams={"output_dir": str(output), **task_config},
    )
    (run / "train.out").write_text("run\n")
    (output / "train.out").write_text("output\n")
    (output / "same.out").symlink_to(run / "train.out")
    submitit = tmp_path / "100_2" / ".submitit"
    submitit.mkdir(parents=True)
    for name in ["100_2_0_log.out", "100_2_1_log.err", "bad_log.out", "ignore.pkl"]:
        (submitit / name).write_text(name + "\n")
    scanner = RunScanner(tmp_path)
    scanner._scan()
    streams = scanner.logs_index("run_2")["streams"]
    assert len(streams) == 5
    assert [s["rank"] for s in streams if s["rank"] is not None] == [0, 1]
    contents = {scanner.log_content("run_2", s["stream_id"]) for s in streams}
    assert contents == {
        b"run\n",
        b"output\n",
        b"100_2_0_log.out\n",
        b"100_2_1_log.err\n",
        b"bad_log.out\n",
    }
    assert len({s["stream_id"] for s in streams}) == 5
