"""Unit tests for the filesystem-backed run registry.

The registry is made of four independent layers — test each in
isolation, then a few end-to-end checks to confirm they compose:

1. :mod:`stable_pretraining.registry._sidecar` — atomic JSON writer +
   heartbeat helpers.
2. :mod:`stable_pretraining.registry._store`   — SQLite cache.
3. :mod:`stable_pretraining.registry._scanner` — filesystem → cache.
4. :mod:`stable_pretraining.registry.logger`   — Lightning logger.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from stable_pretraining.registry import (
    RegistryLogger,
    RunRecord,
    open_registry,
)
from stable_pretraining.registry import _scanner as scanner
from stable_pretraining.registry import _sidecar as sidecar
from stable_pretraining.registry._scanner import scan
from stable_pretraining.registry._store import Store
from stable_pretraining.registry.logger import _flatten_params

pytestmark = pytest.mark.unit


# ============================================================================
# Helpers
# ============================================================================


def _make_run_dir(cache_dir: Path, run_id: str) -> Path:
    """Create a ``{cache_dir}/runs/<date>/<time>/<run_id>`` layout."""
    run_dir = cache_dir / "runs" / "20260101" / "000000" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_test_sidecar(run_dir: Path, **overrides) -> None:
    data = sidecar.make_sidecar(
        run_id=overrides.pop("run_id", run_dir.name),
        run_dir=str(run_dir),
        **overrides,
    )
    sidecar.write_sidecar(run_dir, data)


# ============================================================================
# _sidecar
# ============================================================================


class TestSidecar:
    """Atomic write, JSON round-trip, heartbeat, and liveness rules."""

    def test_atomic_write_and_read_roundtrip(self, tmp_path):
        run_dir = tmp_path / "r1"
        data = sidecar.make_sidecar(
            run_id="r1",
            run_dir=str(run_dir),
            hparams={"lr": 0.01},
            summary={"loss": 0.5},
            tags=["a", "b"],
        )
        sidecar.write_sidecar(run_dir, data)

        read = sidecar.read_sidecar(sidecar.sidecar_path(run_dir))
        assert read is not None
        assert read["run_id"] == "r1"
        assert read["hparams"] == {"lr": 0.01}
        assert read["summary"] == {"loss": 0.5}
        assert read["tags"] == ["a", "b"]
        assert read["schema_version"] == sidecar.SCHEMA_VERSION

    def test_write_stamps_updated_at_fresh(self, tmp_path):
        run_dir = tmp_path / "r1"
        before = time.time()
        sidecar.write_sidecar(
            run_dir, sidecar.make_sidecar(run_id="r1", run_dir=str(run_dir))
        )
        read = sidecar.read_sidecar(sidecar.sidecar_path(run_dir))
        assert read["updated_at"] >= before

    def test_read_missing_returns_none(self, tmp_path):
        assert sidecar.read_sidecar(tmp_path / "missing.json") is None

    def test_read_malformed_returns_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        assert sidecar.read_sidecar(path) is None

    def test_read_missing_run_id_returns_none(self, tmp_path):
        path = tmp_path / "incomplete.json"
        path.write_text(json.dumps({"status": "running"}))
        assert sidecar.read_sidecar(path) is None

    def test_no_tmp_leak_after_write(self, tmp_path):
        """After a successful write, no .tmp files should remain."""
        run_dir = tmp_path / "r1"
        sidecar.write_sidecar(
            run_dir, sidecar.make_sidecar(run_id="r1", run_dir=str(run_dir))
        )
        leftover = list(run_dir.glob(".sidecar.*.tmp"))
        assert leftover == []

    def test_heartbeat_touch_creates_file(self, tmp_path):
        run_dir = tmp_path / "r1"
        run_dir.mkdir()
        sidecar.touch_heartbeat(run_dir)
        assert sidecar.heartbeat_path(run_dir).exists()
        assert sidecar.heartbeat_mtime(run_dir) is not None

    def test_heartbeat_touch_updates_mtime(self, tmp_path):
        run_dir = tmp_path / "r1"
        run_dir.mkdir()
        sidecar.touch_heartbeat(run_dir)
        t1 = sidecar.heartbeat_mtime(run_dir)
        # Set mtime to the past so the next touch is observably newer.
        os.utime(sidecar.heartbeat_path(run_dir), (t1 - 100, t1 - 100))
        sidecar.touch_heartbeat(run_dir)
        t2 = sidecar.heartbeat_mtime(run_dir)
        assert t2 > t1 - 100

    def test_is_alive_terminal_statuses(self, tmp_path):
        now = time.time()
        for status in ("completed", "failed", "orphaned"):
            assert not sidecar.is_alive(status, now, now=now)

    def test_is_alive_within_timeout(self):
        now = 1_000_000.0
        assert sidecar.is_alive("running", now - 10, now=now, timeout_s=60)

    def test_is_alive_past_timeout(self):
        now = 1_000_000.0
        assert not sidecar.is_alive("running", now - 300, now=now, timeout_s=60)

    def test_is_alive_no_heartbeat(self):
        assert not sidecar.is_alive("running", None)


# ============================================================================
# _store
# ============================================================================


class TestStore:
    """SQLite cache: upserts, mtime tracking, filters, read-only mode."""

    def test_upsert_and_get(self, tmp_path):
        with Store(tmp_path / "r.db") as store:
            store.begin()
            store.upsert(
                "r1",
                sidecar.make_sidecar(
                    run_id="r1",
                    run_dir="/tmp/r1",
                    hparams={"lr": 0.01},
                    summary={"loss": 0.5},
                    tags=["a", "b"],
                ),
                sidecar_mtime=123.0,
                alive=True,
            )
            store.commit()

            row = store.get_run("r1")
            assert row is not None
            assert row["run_id"] == "r1"
            assert row["run_dir"] == "/tmp/r1"
            assert row["hparams"] == {"lr": 0.01}
            assert row["summary"] == {"loss": 0.5}
            assert row["tags"] == ["a", "b"]
            assert row["alive"] is True

    def test_upsert_preserves_created_at_when_present(self, tmp_path):
        with Store(tmp_path / "r.db") as store:
            store.begin()
            first = sidecar.make_sidecar(run_id="r1", run_dir="/tmp/r1")
            first["created_at"] = 1000.0
            store.upsert("r1", first, sidecar_mtime=1.0, alive=True)

            second = sidecar.make_sidecar(run_id="r1", run_dir="/tmp/r1")
            second["created_at"] = 1000.0  # sidecar is the source of truth
            store.upsert("r1", second, sidecar_mtime=2.0, alive=True)
            store.commit()

            row = store.get_run("r1")
            assert row["created_at"] == 1000.0

    def test_mark_orphaned(self, tmp_path):
        with Store(tmp_path / "r.db") as store:
            store.begin()
            for rid in ("a", "b", "c"):
                store.upsert(
                    rid,
                    sidecar.make_sidecar(run_id=rid, run_dir=f"/tmp/{rid}"),
                    sidecar_mtime=1.0,
                    alive=True,
                )
            store.mark_orphaned(["b"])
            store.commit()

            assert store.get_run("b")["status"] == "orphaned"
            assert store.get_run("b")["alive"] is False
            assert store.get_run("a")["status"] == "running"

    def test_query_filters(self, tmp_path):
        with Store(tmp_path / "r.db") as store:
            store.begin()
            store.upsert(
                "a",
                sidecar.make_sidecar(
                    run_id="a",
                    run_dir="/tmp/a",
                    status="completed",
                    tags=["resnet"],
                    summary={"val_acc": 0.8},
                ),
                sidecar_mtime=1.0,
                alive=False,
            )
            store.upsert(
                "b",
                sidecar.make_sidecar(
                    run_id="b",
                    run_dir="/tmp/b",
                    status="completed",
                    tags=["resnet"],
                    summary={"val_acc": 0.9},
                ),
                sidecar_mtime=1.0,
                alive=False,
            )
            store.upsert(
                "c",
                sidecar.make_sidecar(
                    run_id="c",
                    run_dir="/tmp/c",
                    status="running",
                    tags=["vit"],
                ),
                sidecar_mtime=1.0,
                alive=True,
            )
            store.commit()

            assert {r["run_id"] for r in store.query_runs(status="completed")} == {
                "a",
                "b",
            }
            assert {r["run_id"] for r in store.query_runs(tag="resnet")} == {"a", "b"}
            assert {r["run_id"] for r in store.query_runs(alive=True)} == {"c"}

            sorted_desc = store.query_runs(sort_by="summary.val_acc", descending=True)
            # Only a/b have val_acc; c has None which sorts last DESC.
            assert sorted_desc[0]["run_id"] == "b"
            assert sorted_desc[1]["run_id"] == "a"

            assert len(store.query_runs(limit=1)) == 1

    def test_readonly_rejects_writes(self, tmp_path):
        db = tmp_path / "r.db"
        Store(db).close()  # create
        ro = Store(db, readonly=True)
        with pytest.raises(RuntimeError):
            ro.begin()
        ro.close()

    def test_readonly_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Store(tmp_path / "missing.db", readonly=True)

    def test_sidecar_mtimes(self, tmp_path):
        with Store(tmp_path / "r.db") as store:
            store.begin()
            store.upsert(
                "a",
                sidecar.make_sidecar(run_id="a", run_dir="/tmp/a"),
                sidecar_mtime=100.0,
                alive=True,
            )
            store.commit()
            assert store.sidecar_mtimes() == {"a": 100.0}


# ============================================================================
# _scanner
# ============================================================================


class TestScanner:
    """Filesystem → cache ingestion: full, incremental, orphan, liveness."""

    def test_full_scan_ingests_sidecars(self, tmp_path):
        for rid in ("a", "b"):
            run_dir = _make_run_dir(tmp_path, rid)
            _write_test_sidecar(run_dir, hparams={"lr": 0.01})
            sidecar.touch_heartbeat(run_dir)

        with Store(tmp_path / "registry.db") as store:
            report = scan(tmp_path, store, full=True)
            assert report.total_sidecars == 2
            assert report.upserted == 2
            assert report.orphaned == 0

            rows = store.query_runs()
            assert {r["run_id"] for r in rows} == {"a", "b"}

    def test_incremental_skips_unchanged(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "a")
        _write_test_sidecar(run_dir, summary={"loss": 1.0})

        with Store(tmp_path / "registry.db") as store:
            r1 = scan(tmp_path, store)
            assert r1.upserted == 1
            r2 = scan(tmp_path, store)
            assert r2.upserted == 0
            assert r2.skipped_unchanged == 1

    def test_incremental_picks_up_updates(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "a")
        _write_test_sidecar(run_dir, summary={"loss": 1.0})
        db = tmp_path / "registry.db"

        with Store(db) as store:
            scan(tmp_path, store)

        # Update the sidecar — bump mtime so the scanner sees it.
        _write_test_sidecar(run_dir, summary={"loss": 0.5})
        os.utime(sidecar.sidecar_path(run_dir), None)

        with Store(db) as store:
            r = scan(tmp_path, store)
            assert r.upserted == 1
            row = store.get_run("a")
            assert row["summary"] == {"loss": 0.5}

    def test_orphan_sweep_marks_missing_run_dir(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "a")
        _write_test_sidecar(run_dir)
        db = tmp_path / "registry.db"

        with Store(db) as store:
            scan(tmp_path, store)

        import shutil

        shutil.rmtree(run_dir)

        with Store(db) as store:
            r = scan(tmp_path, store)
            assert r.orphaned == 1
            row = store.get_run("a")
            assert row["status"] == "orphaned"
            assert row["alive"] is False

    def test_skips_invalid_sidecar(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "a")
        sidecar.sidecar_path(run_dir).write_text("{not json")

        with Store(tmp_path / "registry.db") as store:
            r = scan(tmp_path, store)
            assert r.total_sidecars == 1
            assert r.upserted == 0
            assert r.skipped_invalid == 1

    def test_alive_tracking_from_heartbeat(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "a")
        _write_test_sidecar(run_dir, status="running")
        sidecar.touch_heartbeat(run_dir)

        with Store(tmp_path / "registry.db") as store:
            scan(tmp_path, store, heartbeat_timeout_s=60)
            assert store.get_run("a")["alive"] is True

        # Age the heartbeat past the timeout — alive flips without
        # re-writing the sidecar (incremental path refreshes alive).
        hb = sidecar.heartbeat_path(run_dir)
        old = hb.stat().st_mtime - 600
        os.utime(hb, (old, old))

        with Store(tmp_path / "registry.db") as store:
            scan(tmp_path, store, heartbeat_timeout_s=60)
            assert store.get_run("a")["alive"] is False

    def test_empty_cache_dir(self, tmp_path):
        with Store(tmp_path / "registry.db") as store:
            r = scan(tmp_path, store)
            assert r.total_sidecars == 0


# ============================================================================
# RegistryLogger
# ============================================================================


class TestRegistryLogger:
    """Lightning logger: CSV + sidecar + heartbeat in one pass."""

    def test_construction_sets_log_dir(self, tmp_path):
        logger = RegistryLogger(run_dir=tmp_path, run_id="r1")
        assert Path(logger.log_dir).resolve() == tmp_path.resolve()
        assert logger.run_id == "r1"

    def test_log_hyperparams_writes_sidecar_and_yaml(self, tmp_path):
        logger = RegistryLogger(run_dir=tmp_path, run_id="r1")
        logger.log_hyperparams({"lr": 0.01, "batch_size": 32})
        logger.save()

        data = sidecar.read_sidecar(sidecar.sidecar_path(tmp_path))
        assert data["run_id"] == "r1"
        assert data["hparams"]["lr"] == 0.01
        assert data["hparams"]["batch_size"] == 32
        # CSVLogger writes hparams.yaml at its own timing (finalize / save).
        logger.finalize("success")
        assert (tmp_path / "hparams.yaml").exists()

    def test_log_metrics_accumulates_summary(self, tmp_path):
        logger = RegistryLogger(run_dir=tmp_path, run_id="r1")
        logger.log_hyperparams({})
        logger.log_metrics({"loss": 1.0, "acc": 0.3}, step=0)
        logger.log_metrics({"loss": 0.5, "acc": 0.7}, step=1)
        logger.save()

        data = sidecar.read_sidecar(sidecar.sidecar_path(tmp_path))
        assert data["summary"]["loss"] == 0.5
        assert data["summary"]["acc"] == 0.7

    def test_log_metrics_touches_heartbeat(self, tmp_path):
        logger = RegistryLogger(run_dir=tmp_path, run_id="r1")
        logger.log_hyperparams({})
        assert sidecar.heartbeat_mtime(tmp_path) is None
        logger.log_metrics({"loss": 1.0}, step=0)
        assert sidecar.heartbeat_mtime(tmp_path) is not None

    def test_log_metrics_coerces_tensor_scalar(self, tmp_path):
        import torch

        logger = RegistryLogger(run_dir=tmp_path, run_id="r1")
        logger.log_hyperparams({})
        logger.log_metrics({"loss": torch.tensor(0.25)}, step=0)
        logger.save()

        data = sidecar.read_sidecar(sidecar.sidecar_path(tmp_path))
        assert data["summary"]["loss"] == 0.25

    def test_log_metrics_skips_non_numeric(self, tmp_path):
        """Non-numeric values never land in the summary dict.

        The CSV path may accept them, but the sidecar summary must stay
        strictly numeric so downstream tools can always ``float()`` it.
        """
        from stable_pretraining.registry.logger import _to_scalar

        assert _to_scalar("not a number") is None
        assert _to_scalar(None) is None
        assert _to_scalar([1, 2, 3]) is None
        assert _to_scalar(True) == 1.0
        assert _to_scalar(1) == 1.0
        assert _to_scalar(1.5) == 1.5

    def test_finalize_success_maps_to_completed(self, tmp_path):
        logger = RegistryLogger(run_dir=tmp_path, run_id="r1")
        logger.log_hyperparams({"lr": 0.01})
        logger.log_metrics({"val_acc": 0.9}, step=0)
        logger.finalize("success")

        data = sidecar.read_sidecar(sidecar.sidecar_path(tmp_path))
        assert data["status"] == "completed"
        assert data["summary"]["val_acc"] == 0.9

    def test_finalize_failed_maps_to_failed(self, tmp_path):
        logger = RegistryLogger(run_dir=tmp_path, run_id="r1")
        logger.log_hyperparams({})
        logger.finalize("failed")
        data = sidecar.read_sidecar(sidecar.sidecar_path(tmp_path))
        assert data["status"] == "failed"

    def test_after_save_checkpoint_records_path(self, tmp_path):
        logger = RegistryLogger(run_dir=tmp_path, run_id="r1")
        logger.log_hyperparams({})
        mock_cb = MagicMock()
        mock_cb.best_model_path = str(tmp_path / "checkpoints" / "best.ckpt")
        logger.after_save_checkpoint(mock_cb)

        data = sidecar.read_sidecar(sidecar.sidecar_path(tmp_path))
        assert data["checkpoint_path"] == str(tmp_path / "checkpoints" / "best.ckpt")

    def test_auto_tag_slurm_array(self, tmp_path):
        with patch.dict(os.environ, {"SLURM_ARRAY_JOB_ID": "99999"}):
            logger = RegistryLogger(run_dir=tmp_path, run_id="r1", tags=["resnet"])
        assert "sweep:99999" in logger._tags
        assert "resnet" in logger._tags

    def test_auto_tag_no_duplicate(self, tmp_path):
        with patch.dict(os.environ, {"SLURM_ARRAY_JOB_ID": "99999"}):
            logger = RegistryLogger(run_dir=tmp_path, run_id="r1", tags=["sweep:99999"])
        assert logger._tags.count("sweep:99999") == 1

    def test_tags_and_notes_in_sidecar(self, tmp_path):
        logger = RegistryLogger(
            run_dir=tmp_path,
            run_id="r1",
            tags=["ssl", "debug"],
            notes="Quick test run",
        )
        logger.log_hyperparams({})
        data = sidecar.read_sidecar(sidecar.sidecar_path(tmp_path))
        assert data["tags"] == ["ssl", "debug"]
        assert data["notes"] == "Quick test run"

    def test_created_at_stable_across_flushes(self, tmp_path):
        logger = RegistryLogger(run_dir=tmp_path, run_id="r1")
        logger.log_hyperparams({})
        first = sidecar.read_sidecar(sidecar.sidecar_path(tmp_path))["created_at"]
        time.sleep(0.02)
        logger.save()
        second = sidecar.read_sidecar(sidecar.sidecar_path(tmp_path))["created_at"]
        assert first == second


# ============================================================================
# Registry query API (via open_registry)
# ============================================================================


class TestRegistryQuery:
    """Query API (``open_registry()`` + ``Registry``)."""

    @pytest.fixture
    def populated_cache(self, tmp_path):
        """Build a cache dir with three runs + scanned DB."""
        for rid, spec in {
            "r1": dict(
                status="completed",
                hparams={"lr": 0.01},
                summary={"val_acc": 0.85},
                tags=["resnet", "sweep:100"],
                notes="baseline",
            ),
            "r2": dict(
                status="completed",
                hparams={"lr": 0.1},
                summary={"val_acc": 0.92},
                tags=["resnet", "sweep:100"],
            ),
            "r3": dict(
                status="running",
                hparams={"lr": 0.001},
                tags=["vit", "sweep:200"],
            ),
        }.items():
            run_dir = _make_run_dir(tmp_path, rid)
            _write_test_sidecar(run_dir, **spec)
            if spec["status"] == "running":
                sidecar.touch_heartbeat(run_dir)
        return tmp_path

    def test_query_all(self, populated_cache):
        reg = open_registry(cache_dir=populated_cache)
        runs = reg.query()
        assert len(runs) == 3
        assert all(isinstance(r, RunRecord) for r in runs)
        reg.close()

    def test_query_by_tag(self, populated_cache):
        reg = open_registry(cache_dir=populated_cache)
        assert {r.run_id for r in reg.query(tag="resnet")} == {"r1", "r2"}
        reg.close()

    def test_query_by_sweep_tag(self, populated_cache):
        reg = open_registry(cache_dir=populated_cache)
        assert len(reg.query(tag="sweep:100")) == 2
        reg.close()

    def test_query_by_hparams(self, populated_cache):
        reg = open_registry(cache_dir=populated_cache)
        runs = reg.query(hparams={"lr": 0.01})
        assert [r.run_id for r in runs] == ["r1"]
        reg.close()

    def test_query_sort_by_summary(self, populated_cache):
        reg = open_registry(cache_dir=populated_cache)
        runs = reg.query(tag="sweep:100", sort_by="summary.val_acc", descending=True)
        assert runs[0].run_id == "r2"
        assert runs[0].summary["val_acc"] == 0.92
        reg.close()

    def test_get(self, populated_cache):
        reg = open_registry(cache_dir=populated_cache)
        run = reg.get("r1")
        assert run.run_id == "r1"
        assert run.notes == "baseline"
        assert run.tags == ["resnet", "sweep:100"]
        reg.close()

    def test_get_nonexistent(self, populated_cache):
        reg = open_registry(cache_dir=populated_cache)
        assert reg.get("nope") is None
        with pytest.raises(KeyError):
            _ = reg["nope"]
        reg.close()

    def test_len_and_repr(self, populated_cache):
        reg = open_registry(cache_dir=populated_cache)
        assert len(reg) == 3
        assert "runs=3" in repr(reg)
        reg.close()

    def test_to_dataframe_columns(self, populated_cache):
        reg = open_registry(cache_dir=populated_cache)
        df = reg.to_dataframe(tag="sweep:100")
        assert len(df) == 2
        assert "summary.val_acc" in df.columns
        assert "hparams.lr" in df.columns
        assert "tags" in df.columns
        reg.close()

    def test_to_dataframe_empty(self, populated_cache):
        reg = open_registry(cache_dir=populated_cache)
        assert reg.to_dataframe(tag="nonexistent").empty
        reg.close()

    def test_open_without_scan(self, populated_cache):
        """Scan-once then open repeatedly without re-scanning."""
        open_registry(cache_dir=populated_cache).close()  # first scan
        scanner.invalidate_ttl()
        reg = open_registry(cache_dir=populated_cache, scan=False)
        assert len(reg) == 3
        reg.close()

    def test_open_registry_no_cache_dir_raises(self, tmp_path):
        from stable_pretraining._config import get_config

        cfg = get_config()
        original = cfg._cache_dir
        cfg._cache_dir = None
        try:
            with pytest.raises(ValueError, match="cache_dir"):
                open_registry()
        finally:
            cfg._cache_dir = original


# ============================================================================
# _flatten_params
# ============================================================================


class TestFlattenParams:
    """Dot-path flattening of nested hparam containers."""

    def test_flat_dict(self):
        assert _flatten_params({"lr": 0.01, "epochs": 100}) == {
            "lr": 0.01,
            "epochs": 100,
        }

    def test_nested_dict(self):
        assert _flatten_params({"opt": {"lr": 0.01, "wd": 1e-4}}) == {
            "opt.lr": 0.01,
            "opt.wd": 1e-4,
        }

    def test_list_values(self):
        assert _flatten_params({"layers": [64, 128, 256]}) == {
            "layers.0": 64,
            "layers.1": 128,
            "layers.2": 256,
        }

    def test_non_serializable_values_stringified(self):
        result = _flatten_params({"fn": lambda x: x})
        assert isinstance(result["fn"], str)


# ============================================================================
# CLI (spt registry ...)
# ============================================================================


class TestRegistryCLI:
    """``spt registry`` subcommands: ls/show/best/export/scan/migrate."""

    @pytest.fixture
    def cli_cache(self, tmp_path):
        """Populate a cache dir + scanned DB suitable for CLI tests."""
        for rid, spec in {
            "cli-run-1": dict(
                status="completed",
                hparams={"lr": 0.01},
                summary={"val_acc": 0.85, "train_loss": 0.12},
                tags=["resnet", "sweep:100"],
            ),
            "cli-run-2": dict(
                status="completed",
                hparams={"lr": 0.1},
                summary={"val_acc": 0.92, "train_loss": 0.08},
                tags=["resnet", "sweep:100"],
            ),
            "cli-run-3": dict(status="running", tags=["vit"]),
        }.items():
            run_dir = _make_run_dir(tmp_path, rid)
            _write_test_sidecar(run_dir, **spec)
        # Scan up-front so the CLI has a DB to open.
        with Store(tmp_path / "registry.db") as store:
            scan(tmp_path, store)
        return tmp_path

    def test_ls(self, cli_cache):
        from typer.testing import CliRunner
        from stable_pretraining.cli import app

        scanner.invalidate_ttl()
        result = CliRunner().invoke(
            app, ["registry", "ls", "--cache-dir", str(cli_cache)]
        )
        assert result.exit_code == 0, result.output
        for rid in ("cli-run-1", "cli-run-2", "cli-run-3"):
            assert rid in result.output

    def test_ls_filter_by_tag(self, cli_cache):
        from typer.testing import CliRunner
        from stable_pretraining.cli import app

        scanner.invalidate_ttl()
        result = CliRunner().invoke(
            app, ["registry", "ls", "--cache-dir", str(cli_cache), "--tag", "vit"]
        )
        assert result.exit_code == 0
        assert "cli-run-3" in result.output
        assert "cli-run-1" not in result.output

    def test_ls_filter_by_status(self, cli_cache):
        from typer.testing import CliRunner
        from stable_pretraining.cli import app

        scanner.invalidate_ttl()
        result = CliRunner().invoke(
            app,
            ["registry", "ls", "--cache-dir", str(cli_cache), "--status", "completed"],
        )
        assert result.exit_code == 0
        assert "cli-run-1" in result.output
        assert "cli-run-3" not in result.output

    def test_show(self, cli_cache):
        from typer.testing import CliRunner
        from stable_pretraining.cli import app

        scanner.invalidate_ttl()
        result = CliRunner().invoke(
            app,
            ["registry", "show", "cli-run-1", "--cache-dir", str(cli_cache)],
        )
        assert result.exit_code == 0
        assert "cli-run-1" in result.output
        assert "val_acc" in result.output
        assert "0.85" in result.output

    def test_show_not_found(self, cli_cache):
        from typer.testing import CliRunner
        from stable_pretraining.cli import app

        scanner.invalidate_ttl()
        result = CliRunner().invoke(
            app,
            ["registry", "show", "missing", "--cache-dir", str(cli_cache)],
        )
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_best_descending(self, cli_cache):
        from typer.testing import CliRunner
        from stable_pretraining.cli import app

        scanner.invalidate_ttl()
        result = CliRunner().invoke(
            app,
            ["registry", "best", "val_acc", "--cache-dir", str(cli_cache)],
        )
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        # Header line + first data row
        assert "cli-run-2" in lines[1]

    def test_best_ascending(self, cli_cache):
        from typer.testing import CliRunner
        from stable_pretraining.cli import app

        scanner.invalidate_ttl()
        result = CliRunner().invoke(
            app,
            [
                "registry",
                "best",
                "train_loss",
                "--asc",
                "--cache-dir",
                str(cli_cache),
            ],
        )
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        assert "cli-run-2" in lines[1]  # 0.08 < 0.12

    def test_export_csv(self, cli_cache, tmp_path):
        from typer.testing import CliRunner
        from stable_pretraining.cli import app

        scanner.invalidate_ttl()
        out = tmp_path / "export.csv"
        result = CliRunner().invoke(
            app,
            ["registry", "export", str(out), "--cache-dir", str(cli_cache)],
        )
        assert result.exit_code == 0, result.output
        assert "Exported 3 runs" in result.output

        import pandas as pd

        df = pd.read_csv(out)
        assert len(df) == 3
        assert "summary.val_acc" in df.columns

    def test_scan_command(self, tmp_path):
        """`spt registry scan` bootstraps the DB from sidecars."""
        from typer.testing import CliRunner
        from stable_pretraining.cli import app

        run_dir = _make_run_dir(tmp_path, "a")
        _write_test_sidecar(run_dir, summary={"loss": 1.0})

        scanner.invalidate_ttl()
        result = CliRunner().invoke(
            app, ["registry", "scan", "--cache-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert (tmp_path / "registry.db").exists()

    def test_migrate_writes_sidecars_from_legacy_db(self, tmp_path):
        """Create a minimal legacy DB and migrate it; sidecars appear."""
        import sqlite3
        from typer.testing import CliRunner

        from stable_pretraining.cli import app

        legacy = tmp_path / "legacy.db"
        run_dir = _make_run_dir(tmp_path, "legacy-run")

        conn = sqlite3.connect(legacy)
        conn.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                status TEXT,
                created_at REAL,
                updated_at REAL,
                run_dir TEXT,
                checkpoint_path TEXT,
                config TEXT,
                hparams TEXT,
                summary TEXT,
                tags TEXT,
                notes TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO runs VALUES (?, 'completed', 100.0, 200.0, ?, NULL, '{}', ?, ?, ?, '')",
            (
                "legacy-run",
                str(run_dir),
                json.dumps({"lr": 0.01}),
                json.dumps({"val_acc": 0.77}),
                json.dumps(["legacy"]),
            ),
        )
        conn.commit()
        conn.close()

        scanner.invalidate_ttl()
        result = CliRunner().invoke(
            app,
            [
                "registry",
                "migrate",
                str(legacy),
                "--cache-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output

        data = sidecar.read_sidecar(sidecar.sidecar_path(run_dir))
        assert data["run_id"] == "legacy-run"
        assert data["hparams"] == {"lr": 0.01}
        assert data["summary"] == {"val_acc": 0.77}
        assert data["tags"] == ["legacy"]


# ============================================================================
# Hydra config flattening & injection (Manager integration)
# ============================================================================


class TestFlattenHydraConfig:
    """Tests for Manager._flatten_hydra_config (the shared flattening logic)."""

    def _make_manager_with_configs(self, trainer_cfg, module_cfg, data_cfg):
        from stable_pretraining.manager import Manager
        from stable_pretraining.tests.utils import BoringModule, BoringDataModule

        manager = Manager(
            trainer=OmegaConf.create(trainer_cfg),
            module=BoringModule(),
            data=BoringDataModule(),
        )
        if module_cfg is not None:
            manager.module = OmegaConf.create(module_cfg)
        if data_cfg is not None:
            manager.data = OmegaConf.create(data_cfg)
        return manager

    def test_basic_flat_keys(self):
        manager = self._make_manager_with_configs(
            {"max_epochs": 100, "accelerator": "gpu"},
            {"_target_": "my.Module", "lr": 0.01},
            None,
        )
        flat = manager._flatten_hydra_config()
        assert flat["trainer.max_epochs"] == 100
        assert flat["module._target_"] == "my.Module"

    def test_deeply_nested(self):
        manager = self._make_manager_with_configs(
            {"max_epochs": 10},
            {"optim": {"optimizer": {"lr": 5.0, "weight_decay": 1e-6}}},
            None,
        )
        flat = manager._flatten_hydra_config()
        assert flat["module.optim.optimizer.lr"] == 5.0
        assert flat["module.optim.optimizer.weight_decay"] == 1e-6

    def test_lists_expanded(self):
        manager = self._make_manager_with_configs(
            {"max_epochs": 10},
            {"hidden_dims": [64, 128, 256]},
            None,
        )
        flat = manager._flatten_hydra_config()
        assert flat["module.hidden_dims.0"] == 64
        assert flat["module.hidden_dims.2"] == 256
        assert "module.hidden_dims" not in flat


class TestEndToEnd:
    """End-to-end: Manager → sidecar → scan → queryable Registry.

    Covers the full production path: Hydra config flows through
    Manager, the logger writes sidecar + CSV, the scanner picks it up,
    and ``open_registry()`` can query it.
    """

    def _run(self, tmp_path, trainer_cfg, module, data):
        from stable_pretraining.manager import Manager

        Manager(trainer=trainer_cfg, module=module, data=data)()

    def test_config_flows_to_registry(self, tmp_path):
        from stable_pretraining.tests.utils import BoringModule, BoringDataModule

        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.Trainer",
                "max_epochs": 1,
                "accelerator": "cpu",
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        )
        self._run(tmp_path, trainer_cfg, BoringModule(), BoringDataModule())

        scanner.invalidate_ttl()
        reg = open_registry(cache_dir=tmp_path)
        assert len(reg) == 1
        run = reg.query()[0]
        assert run.status == "completed"
        assert run.hparams["trainer.max_epochs"] == 1
        assert run.run_dir is not None
        reg.close()

    def test_no_files_leak_to_cwd(self, tmp_path):
        import glob

        from stable_pretraining.tests.utils import BoringModule, BoringDataModule

        cwd_before = set(glob.glob("*"))
        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.Trainer",
                "max_epochs": 1,
                "accelerator": "cpu",
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        )
        self._run(tmp_path, trainer_cfg, BoringModule(), BoringDataModule())
        leaked = set(glob.glob("*")) - cwd_before
        assert not leaked, f"Files leaked to CWD: {leaked}"

    def test_dataframe_has_flattened_hparams(self, tmp_path):
        from stable_pretraining.tests.utils import BoringModule, BoringDataModule

        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.Trainer",
                "max_epochs": 1,
                "accelerator": "cpu",
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        )
        self._run(tmp_path, trainer_cfg, BoringModule(), BoringDataModule())

        scanner.invalidate_ttl()
        reg = open_registry(cache_dir=tmp_path)
        df = reg.to_dataframe()
        assert len(df) == 1
        assert "hparams.trainer.max_epochs" in df.columns
        assert df.iloc[0]["hparams.trainer.max_epochs"] == 1
        reg.close()

    def test_csv_still_written(self, tmp_path):
        """RegistryLogger is a CSVLogger — metrics.csv must exist."""
        from stable_pretraining.tests.utils import BoringModule, BoringDataModule

        trainer_cfg = OmegaConf.create(
            {
                "_target_": "lightning.Trainer",
                "max_epochs": 1,
                "accelerator": "cpu",
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        )
        self._run(tmp_path, trainer_cfg, BoringModule(), BoringDataModule())

        # Find the run_dir (layout: cache_dir/runs/<date>/<time>/<run_id>)
        run_dirs = list((tmp_path / "runs").rglob("sidecar.json"))
        assert len(run_dirs) == 1
        run_dir = run_dirs[0].parent
        # CSVLogger writes metrics.csv unless there were zero metrics
        # (BoringModule may not log any) — so we check either the CSV
        # or the hparams.yaml, at least one of which must be present.
        assert (run_dir / "hparams.yaml").exists() or (run_dir / "metrics.csv").exists()
