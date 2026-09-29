"""Registry failures preserve resources, old data, and actionable CLI output."""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from typer.testing import CliRunner

from stable_pretraining import _config, cli
from stable_pretraining.registry import _scanner, _sidecar, _store

pytestmark = pytest.mark.unit
runner = CliRunner()


@pytest.mark.parametrize(
    "command", [["ls"], ["show", "missing"], ["best", "loss"], ["export"]]
)
@pytest.mark.parametrize("failure", [False, True])
def test_cli_closes_registry_for_empty_results_and_query_errors(
    monkeypatch, command, failure
):
    reg = Mock()
    reg.query.return_value = []
    reg.get.return_value = None
    reg.to_dataframe.return_value = pd.DataFrame()
    if failure:
        for method in (reg.query, reg.get, reg.to_dataframe):
            method.side_effect = OSError("database unavailable")
    monkeypatch.setattr(cli, "_open_registry", lambda *a, **kw: reg)
    result = runner.invoke(cli.app, ["registry", *command])
    reg.close.assert_called_once()
    if failure:
        assert isinstance(result.exception, OSError)
    else:
        assert result.exit_code == (1 if command[0] == "show" else 0)


def test_best_with_missing_metric_and_parquet_roundtrip(tmp_path, monkeypatch):
    reg = Mock()
    reg.query.return_value = [SimpleNamespace(summary={})]
    monkeypatch.setattr(cli, "_open_registry", lambda *a, **kw: reg)
    result = runner.invoke(cli.app, ["registry", "best", "loss"])
    assert result.exit_code == 0 and "No runs have metric 'loss'" in result.output
    reg.close.assert_called_once()
    frame = pd.DataFrame({"run_id": ["a", "b"], "summary.loss": [0.1, 0.2]})
    reg.to_dataframe.return_value = frame
    output = tmp_path / "runs.parquet"
    result = runner.invoke(cli.app, ["registry", "export", str(output)])
    assert result.exit_code == 0, result.output
    pd.testing.assert_frame_equal(pd.read_parquet(output), frame)


@pytest.mark.parametrize("config_fails", [False, True])
def test_cache_resolution_reports_unconfigured_state_and_accepts_db_only(
    tmp_path, monkeypatch, config_fails
):
    monkeypatch.delenv("SPT_CACHE_DIR", raising=False)
    monkeypatch.setattr(_config.get_config(), "_cache_dir", None)
    if config_fails:
        monkeypatch.setattr(
            _config, "get_config", Mock(side_effect=RuntimeError("unavailable"))
        )
    assert cli._resolve_cache_dir_only(None) is None
    result = runner.invoke(cli.app, ["registry", "ls"])
    assert result.exit_code == 1 and "No --cache-dir" in result.output
    db = tmp_path / "custom.db"
    assert cli._resolve_cache_and_db(str(db), None) == (tmp_path, db)
    assert cli._resolve_cache_dir_only(str(tmp_path)) == tmp_path


def test_missing_registry_and_empty_scan_have_recovery_instructions(tmp_path):
    with pytest.raises(cli.typer.Exit):
        cli._open_registry(cache=str(tmp_path), scan=False)
    result = runner.invoke(cli.app, ["registry", "scan", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0 and "no sidecars found" in result.output
    result = runner.invoke(
        cli.app,
        [
            "registry",
            "migrate",
            str(tmp_path / "missing.db"),
            "--cache-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1 and "Source DB not found" in result.output


def test_legacy_migration_preserves_existing_runs_and_decodes_bad_optional_fields(
    tmp_path,
):
    db = tmp_path / "legacy.db"
    kept, bad = tmp_path / "kept", tmp_path / "bad"
    kept.mkdir()
    bad.mkdir()
    _sidecar.write_sidecar(kept, {"run_id": "kept", "notes": "newer"})
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE runs (run_id TEXT, run_dir TEXT, status TEXT, created_at REAL, hparams TEXT, summary TEXT, tags TEXT, notes TEXT, checkpoint_path TEXT)"
        )
        conn.executemany(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
            [
                ("kept", str(kept), "completed", 10, "{}", "{}", "[]", "legacy", None),
                ("bad", str(bad), None, 0, "{broken", None, "oops", None, None),
                (
                    "gone",
                    str(tmp_path / "gone"),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                ("no-dir", None, None, None, None, None, None, None, None),
            ],
        )
    args = ["registry", "migrate", str(db), "--cache-dir", str(tmp_path)]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "1 sidecars written, 1 already existed, 2 rows" in result.output
    assert json.loads((kept / "sidecar.json").read_text())["notes"] == "newer"
    migrated = json.loads((bad / "sidecar.json").read_text())
    assert (migrated["hparams"], migrated["summary"], migrated["tags"]) == ({}, {}, [])
    assert migrated["status"] == "unknown"
    result = runner.invoke(cli.app, [*args, "--overwrite"])
    assert result.exit_code == 0
    assert json.loads((kept / "sidecar.json").read_text())["notes"] == "legacy"


def test_scan_rolls_back_partial_writes_and_can_retry(tmp_path, monkeypatch):
    for name in ("a", "b"):
        _sidecar.write_sidecar(
            tmp_path / "runs" / name, {"run_id": name, "status": "running"}
        )
    with _store.Store(tmp_path / "registry.db", readonly=False) as store:
        original = store.upsert
        calls = []

        def fail_second(*args, **kwargs):
            calls.append(args[0])
            if len(calls) == 2:
                raise OSError("disk full")
            return original(*args, **kwargs)

        monkeypatch.setattr(store, "upsert", fail_second)
        with pytest.raises(OSError, match="disk full"):
            _scanner.scan(tmp_path, store)
        assert store.get_run(calls[0]) is None
        monkeypatch.setattr(store, "upsert", original)
        assert _scanner.scan(tmp_path, store).upserted == 2
        assert store.get_meta("last_scan_at") is not None
        assert store.get_meta("missing") is None


def test_scan_handles_disappearing_file_and_authoritative_run_id(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "renamed"
    _sidecar.write_sidecar(run, {"run_id": "original", "status": "completed"})
    monkeypatch.setattr(
        _scanner,
        "_iter_sidecars",
        lambda _: iter([run / "gone.json", run / "sidecar.json"]),
    )
    with _store.Store(tmp_path / "registry.db", readonly=False) as store:
        report = _scanner.scan(tmp_path, store)
        assert report.total_sidecars == 2 and report.upserted == 1
        assert store.get_run("original")["run_dir"] == str(run)
        assert store.get_run("renamed") is None


def test_query_scan_ttl_avoids_repeated_filesystem_walks(tmp_path, monkeypatch):
    _scanner._LAST_SCAN_AT.pop(str((tmp_path / "registry.db").resolve()), None)
    scan = Mock(return_value=_scanner.ScanReport())
    monkeypatch.setattr(_scanner, "scan", scan)
    _scanner.scan_for_query(
        cache_dir=tmp_path, db_path=tmp_path / "registry.db", ttl_s=100
    )
    assert (
        _scanner.scan_for_query(
            cache_dir=tmp_path, db_path=tmp_path / "registry.db", ttl_s=100
        )
        is None
    )
    scan.assert_called_once()


def test_sidecar_cleanup_failure_preserves_original_write_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _sidecar.os, "replace", Mock(side_effect=OSError("replace failed"))
    )
    monkeypatch.setattr(
        _sidecar.os, "unlink", Mock(side_effect=OSError("unlink failed"))
    )
    with pytest.raises(OSError, match="replace failed"):
        _sidecar.write_sidecar(tmp_path, {"run_id": "a"})
    monkeypatch.setattr(Path, "touch", Mock(side_effect=OSError("read only")))
    _sidecar.touch_heartbeat(tmp_path)


def test_corrupt_cached_json_falls_back_without_losing_run_identity():
    row = _store._row_to_dict(
        {
            "run_id": "a",
            "hparams": "bad",
            "summary": "bad",
            "config": "bad",
            "tags": "bad",
            "alive": 0,
        }
    )
    assert row == {
        "run_id": "a",
        "hparams": {},
        "summary": {},
        "config": {},
        "tags": [],
        "alive": False,
    }
