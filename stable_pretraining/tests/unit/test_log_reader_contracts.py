"""Log analysis preserves run identity, ordering, and caller-owned options."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from stable_pretraining.utils import log_reader as logs

pytestmark = pytest.mark.unit


def test_local_reader_skips_invalid_records_and_retains_rank(tmp_path):
    (tmp_path / "logs_rank_2.jsonl").write_text(
        '{"loss": 1}\ninvalid\n[]\n{"loss": 2}\n'
    )
    (tmp_path / "logs_rank_10.jsonl").write_text('{"loss": 3}\n')
    assert sorted(logs.read_local_logs(tmp_path), key=lambda row: row["loss"]) == [
        {"loss": 1, "rank": 2},
        {"loss": 2, "rank": 2},
        {"loss": 3, "rank": 10},
    ]


@pytest.mark.parametrize("redirect", [None, nullcontext])
def test_local_project_pairs_each_config_with_its_actual_records(
    tmp_path, monkeypatch, redirect
):
    monkeypatch.setattr(logs, "logging_redirect_tqdm", redirect)
    for i in [10, 2, 1]:
        run = tmp_path / f"run{i}"
        run.mkdir()
        (run / "hparams.yaml").write_text(f"seed: {i}\nmodel:\n  width: {i * 2}\n")
        (run / "logs_rank_0.jsonl").write_text(f'{{"seed": {i}}}\n')
    configs, values = logs.read_local_project(tmp_path, num_workers=2)
    assert len(configs) == len(values) == 3
    for config, records in zip(configs.to_dict("records"), values):
        assert records == [{"seed": config["seed"], "rank": 0}]
        assert config["model.width"] == config["seed"] * 2


@pytest.mark.parametrize("method", ["read", "read_config", "read_project"])
def test_local_reader_reports_missing_directory(tmp_path, method):
    with pytest.raises(ValueError, match="not a directory"):
        getattr(logs.LocalLogReader(), method)(tmp_path / "absent")


def test_local_config_prefers_hydra_and_reports_absent_config(tmp_path):
    reader = logs.LocalLogReader()
    with pytest.raises(FileNotFoundError):
        reader.read_config(tmp_path)
    (tmp_path / "hparams.yaml").write_text("seed: 1\n")
    assert reader.read_config(tmp_path)["seed"] == 1
    (tmp_path / ".hydra").mkdir()
    (tmp_path / ".hydra/config.yaml").write_text("seed: 2\n")
    assert reader.read_config(tmp_path)["seed"] == 2


def test_sort_and_flatten_leave_inputs_unchanged():
    names = ["Run10", "run2", "run1"]
    assert logs.natural_sort(names) == ["run1", "run2", "Run10"]
    assert names[0] == "Run10"
    source = {"model": {"width": 4}, "data": {"split": "train"}, "seed": 1}
    assert logs.flatten_config(source) == {
        "model.width": 4,
        "data.split": "train",
        "seed": 1,
    }
    assert source["model"] == {"width": 4}


@pytest.fixture
def remote(monkeypatch):
    run = SimpleNamespace(
        id="run1",
        name="first",
        lastHistoryStep=7,
        config={"width": 8},
        summary=SimpleNamespace(_json_dict={"accuracy": 0.8}),
        tags=["test"],
        created_at="2026-01-01",
        scan_history=Mock(
            return_value=[{"_step": 6, "loss": 2.0}, {"_step": 7, "loss": 1.0}]
        ),
    )
    api = SimpleNamespace(run=Mock(return_value=run), runs=Mock(return_value=[run]))
    monkeypatch.setattr(logs, "wandbapi", SimpleNamespace(Api=lambda: api))
    return api, run


@pytest.mark.parametrize(
    "min_step,max_step,expected", [(0, -1, (0, 8)), (-2, -1, (6, 8)), (1, 4, (1, 4))]
)
def test_wandb_history_bounds_and_keys(remote, min_step, max_step, expected):
    api, run = remote
    keys = ["loss"]
    frame, config = logs.read_wandb_run(
        "team", "project", "run1", min_step=min_step, max_step=max_step, keys=keys
    )
    assert keys == ["loss"]
    api.run.assert_called_once_with("team/project/run1")
    run.scan_history.assert_called_once_with(
        keys=["loss", "_step"], min_step=expected[0], max_step=expected[1]
    )
    assert frame.index.tolist() == [6, 7]
    assert frame.loss.tolist() == [2.0, 1.0]
    assert config == {"width": 8}


def test_wandb_empty_history(remote):
    _, run = remote
    run.scan_history.return_value = []
    frame, _ = logs.WandbLogReader().read("team", "project", "run1")
    assert frame.empty


def test_wandb_project_fetch_works_with_real_worker_pool(remote):
    api, _ = remote
    frames, configs = logs.read_wandb_project("team", "project", num_workers=2)
    assert list(frames) == list(configs) == ["team/project/run1"]
    assert frames["team/project/run1"].loss.tolist() == [2.0, 1.0]
    api.runs.assert_called_once_with(
        "team/project",
        filters=None,
        order="+created_at",
        per_page=50,
        include_sweeps=True,
    )


def test_wandb_summary_does_not_download_history(remote):
    _, run = remote
    frame = logs.read_wandb_project("team", "project", return_summary=True)
    assert frame.to_dict("records") == [
        {
            "accuracy": 0.8,
            "width": 8,
            "tags": ["test"],
            "name": "first",
            "created_at": "2026-01-01",
            "id": "run1",
        }
    ]
    run.scan_history.assert_not_called()


def test_wandb_unavailable_is_actionable(monkeypatch):
    monkeypatch.setattr(logs, "wandbapi", None)
    with pytest.raises(ImportError, match="install"):
        logs.WandbLogReader()


@pytest.mark.parametrize("filters", [None, {"model": "m2"}, {"model": ["m2"]}])
def test_results_table_aggregates_replicates_and_marks_missing_metrics(filters):
    configs = {
        "a": {"model": "m2", "data": "d1"},
        "b": {"model": "m2", "data": "d1"},
        "c": {"model": "m10", "data": "d2"},
    }
    frames = {
        "a": pd.DataFrame({"score": [1.0, 3.0]}),
        "b": pd.DataFrame({"score": [5.0]}),
        "c": pd.DataFrame({"other": [1.0]}),
    }
    out = logs.create_results_table(
        frames, configs, "score", "model", "data", filters=filters
    )
    assert out.loc["m2", "d1"] == 3.0
    if filters is None:
        assert out.index.tolist() == ["m2", "m10"]
        assert np.isnan(out.loc["m10", "d2"])


def test_results_table_rejects_missing_run():
    with pytest.raises(AssertionError, match="not found"):
        logs.create_results_table(
            {}, {"missing": {"model": "m"}}, "score", "model", "data"
        )


def test_interactive_table_uses_last_metric_and_missing_as_nan(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "model")
    cfg = pd.DataFrame(
        {"model": ["a", "a", "b"], "seed": [1, 2, 1], "fixed": [3, 3, 3]}
    )
    out = logs.TableFormatter.tabulate_runs(
        cfg, [{"score": [1, 2]}, {"score": [3, 4]}, {}], "score"
    )
    assert out.loc["a", 1] == 2
    assert out.loc["a", 2] == 4
    assert np.isnan(out.loc["b", 1])
    assert cfg.columns.tolist() == ["model", "seed", "fixed"]
