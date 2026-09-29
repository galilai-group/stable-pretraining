"""Metric collection preserves run grouping, metadata, and sparse metric updates."""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from stable_pretraining.loggers import csv_log_reader as reader

pytestmark = pytest.mark.unit


def write_run(root, name):
    run = root / name
    (run / ".hydra").mkdir(parents=True)
    (run / ".hydra/config.yaml").write_text("seed: 42\n")
    (run / "hparams.yaml").write_text(
        "seed: 42\nlayers: [2, 3]\noptimizer: {lr: 0.1}\n"
    )
    for version in [0, 1]:
        folder = run / f"version_{version}"
        folder.mkdir()
        pd.DataFrame(
            {
                "step": [version * 2, version * 2 + 1],
                " loss ": [2.0, None],
                "Unnamed: 0": [0, 1],
            }
        ).to_csv(folder / "metrics.csv", index=False)
    return run


def test_collection_groups_versions_and_applies_run_aggregation(monkeypatch, tmp_path):
    monkeypatch.setattr(reader, "ProcessPoolExecutor", ThreadPoolExecutor)
    write_run(tmp_path, "keep")
    write_run(tmp_path, "skip")
    collector = reader.CSVLogAutoSummarizer(
        agg=lambda df: df.iloc[-1], exclude_globs=["skip/*/metrics.csv"], max_workers=2
    )
    result = collector.collect(tmp_path)
    assert len(result) == 1
    assert result.iloc[0]["root"] == tmp_path / "keep"
    assert result.iloc[0]["config/seed"] == 42
    assert result.iloc[0]["config/layers"] == "[2, 3]"
    assert result.iloc[0]["loss"] == 2
    assert not any(column.startswith("Unnamed") for column in result)


def test_collection_multiple_roots_and_include_filter(monkeypatch, tmp_path):
    monkeypatch.setattr(reader, "ProcessPoolExecutor", ThreadPoolExecutor)
    roots = [tmp_path / "a", tmp_path / "b"]
    for root in roots:
        write_run(root, "run")
    collector = reader.CSVLogAutoSummarizer(
        include_globs=["*/version_0/metrics.csv"], max_workers=2
    )
    result = collector.collect(roots)
    assert len(result) == 4
    assert set(result["step"]) == {0, 1}
    assert len(set(result["root"])) == 2
    assert collector.collect(tmp_path / "empty").empty


def test_bad_metadata_is_ignored_and_bad_aggregation_is_rejected(tmp_path):
    collector = reader.CSVLogAutoSummarizer(agg=lambda _: 42)
    path = tmp_path / "metrics.csv"
    pd.DataFrame({"step": [0], "metric": [float("nan")]}).to_csv(path, index=False)
    metadata = tmp_path / "hparams.yaml"
    metadata.write_text("x: [unclosed")
    assert collector._load_yaml(metadata) == {}
    assert collector._load_yaml(tmp_path / "missing") == {}
    metadata.write_text("")
    assert collector._load_yaml(metadata) == {}
    assert collector._find_run_root(path) == tmp_path
    with pytest.raises(RuntimeError, match="series or dataframe"):
        collector._merge_metrics_files((tmp_path, [path]))
    assert collector._merge_metrics_files((tmp_path, [])) is None
