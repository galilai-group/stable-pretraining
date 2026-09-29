"""Failed log persistence remains recoverable and does not fabricate metrics."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from stable_pretraining.loggers import csv_log_reader as csv
from stable_pretraining.loggers import trackio
from stable_pretraining.registry import logger as registry
from stable_pretraining.registry import _sidecar

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("failure", ["rename", "cleanup"])
def test_compression_selection_preserves_loadable_winner_after_filesystem_failure(
    tmp_path, monkeypatch, failure
):
    frame = pd.DataFrame({"step": [0, 1], "loss": [2.0, 1.0]})
    monkeypatch.setattr(csv, "_get_trials", lambda: [("csv", None), ("pickle", None)])
    if failure == "rename":
        monkeypatch.setattr(csv.os, "rename", Mock(side_effect=OSError("denied")))
    else:
        monkeypatch.setattr(csv.os, "remove", Mock(side_effect=OSError("busy")))
    path = csv.save_best_compressed(frame, str(tmp_path / "result"))
    assert Path(path).is_file()
    restored = csv.load_best_compressed(path)
    pd.testing.assert_frame_equal(restored, frame, check_dtype=False)


def test_bad_compression_output_and_corrupt_input_report_failure(tmp_path):
    frame = pd.DataFrame({"value": [1]})
    assert csv._save_variant(frame, str(tmp_path / "missing/out"), "csv", None) == (
        None,
        None,
    )
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"\xff\xfe\x00\x80")
    with pytest.raises(ValueError, match="Failed to load"):
        csv.load_best_compressed(str(bad))
    good = tmp_path / "good.pkl"
    frame.to_pickle(good)
    pd.testing.assert_frame_equal(csv.load_best_compressed(str(good)), frame)


def test_local_trackio_collection_skips_empty_runs(monkeypatch):
    from trackio.sqlite_storage import SQLiteStorage

    monkeypatch.delenv("TRACKIO_SERVER_URL", raising=False)
    monkeypatch.setattr(SQLiteStorage, "get_runs", lambda _: ["empty", "run"])
    monkeypatch.setattr(
        SQLiteStorage,
        "get_logs",
        lambda project, run: [] if run == "empty" else [{"step": 1, "loss": 0.5}],
    )
    frame = trackio.load_project_df("local")
    assert frame.to_dict("records") == [{"step": 1, "loss": 0.5, "run": "run"}]
    assert trackio._params_to_dict(OmegaConf.create({"width": 8})) == {"width": 8}


@pytest.mark.parametrize("helper", [registry._to_scalar, trackio._to_scalar])
def test_invalid_scalar_values_are_ignored(helper):
    assert helper(SimpleNamespace(item=Mock(side_effect=ValueError("invalid")))) is None
    assert helper(torch.ones(2)) is None
    assert helper(torch.tensor(0.5)) == 0.5


def test_registry_auxiliary_write_failures_preserve_cached_metrics(
    tmp_path, monkeypatch, capsys
):
    logger = registry.RegistryLogger(tmp_path, "run")
    logger.log_metrics({"loss": 0.5}, step=3)
    monkeypatch.setattr(logger, "_write_sidecar", Mock(side_effect=OSError("full")))
    monkeypatch.setattr(logger, "_write_summary", Mock(side_effect=OSError("full")))
    logger._write_sidecar_safe()
    logger._write_summary_safe()
    monkeypatch.setattr(Path, "open", Mock(side_effect=OSError("full")))
    logger._append_media_event({"type": "image"})
    assert "media.jsonl write failed" in capsys.readouterr().out
    assert logger._summary["loss"] == 0.5


def test_hparams_and_sidecars_have_safe_scalar_fallbacks():
    assert registry._flatten_params(SimpleNamespace(width=8)) == {"width": 8}
    assert registry._flatten_params(42) == {"params": "42"}
    assert registry._flatten_params(OmegaConf.create({"shape": [2, 3]})) == {
        "shape.0": 2,
        "shape.1": 3,
    }

    class BrokenScalar:
        def item(self):
            raise ValueError("invalid")

        def __str__(self):
            return "unavailable"

    assert json.dumps(BrokenScalar(), default=_sidecar._json_default) == '"unavailable"'
