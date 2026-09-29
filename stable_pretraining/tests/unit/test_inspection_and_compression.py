"""Inspection and compressed exports preserve models and experiment results."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from stable_pretraining.backbone.video import info
from stable_pretraining.loggers import csv_log_reader as csv

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("structured", [False, True])
def test_video_summary_preserves_model_mode_and_weights(training, structured):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)
            self.linear.bias.requires_grad_(False)

        def forward(self, x):
            out = self.linear(x)
            return (
                SimpleNamespace(feature_map=out, pooled=out.mean(1))
                if structured
                else out
            )

    model = Model().train(training)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    assert info.count_parameters(model) == 8
    result = info.summarize(model, (2, 3, 4))
    assert result["params"] == 10
    assert result["feature_shape"] == (2, 3, 2)
    if structured:
        assert result["pooled_shape"] == (2, 2)
    assert model.training is training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])


def test_summary_restores_mode_after_forward_error():
    model = nn.Linear(4, 2).train()
    with pytest.raises(RuntimeError):
        info.summarize(model, (2, 3))
    assert model.training
    assert info.summarize(model) == {"params": 10}


@pytest.mark.parametrize("threshold", [0, 10])
def test_video_zoo_counts_on_meta_and_runs_only_small_models(
    monkeypatch, capsys, threshold
):
    seen = []

    def tiny(**kwargs):
        model = nn.Conv3d(3, 1, 1)
        seen.append(next(model.parameters()).device.type)
        return model

    for name in list(vars(info)):
        if name.startswith(
            ("magvit2_", "predrnn_v2_", "recurrent_vit_", "cosmos_", "videomamba_")
        ):
            monkeypatch.setattr(info, name, tiny)
    info.print_video_zoo(input_hw=4, input_t=2, skip_forward_above=threshold)
    output = capsys.readouterr().out
    assert seen.count("meta") == 31
    assert seen.count("cpu") == (31 if threshold else 0)
    assert "(1, 1, 2, 4, 4)" in output if threshold else "(skipped)" in output
    assert info._format_params(2_000_000_000) == "2.00B"
    assert info._format_params(3_000_000) == "3.0M"


@pytest.mark.parametrize(
    "fmt,comp",
    [
        ("parquet", "zstd"),
        ("feather", "lz4"),
        ("pickle", "gzip"),
        ("csv", "gzip"),
        ("csv", "bz2"),
        ("csv", "xz"),
        ("csv", "zip"),
    ],
)
def test_compression_roundtrip_preserves_values(tmp_path, fmt, comp):
    original = pd.DataFrame(
        {"step": [1, 2, 3], "loss": [0.5, 0.25, 0.125], "name": ["a", "b", "c"]}
    )
    filename, size = csv._save_variant(original, str(tmp_path / "run"), fmt, comp)
    assert size == Path(filename).stat().st_size > 0
    result = csv.load_best_compressed(filename)
    pd.testing.assert_frame_equal(result, original, check_dtype=False)


def test_best_compression_retains_only_winner_and_input_frame(tmp_path, monkeypatch):
    frame = pd.DataFrame({"value": [1.0] * 100, "label": ["same"] * 100})
    before = frame.copy(deep=True)
    monkeypatch.setattr(
        csv,
        "_get_trials",
        lambda: [("csv", "gzip"), ("parquet", "zstd"), ("feather", "lz4")],
    )
    path = csv.save_best_compressed(frame, str(tmp_path / "output.csv.gz"))
    assert Path(path).exists()
    assert not list(tmp_path.glob("*_trial_*"))
    result = csv.load_best_compressed(path)
    assert result.value.tolist() == [1.0] * 100
    assert result.label.tolist() == ["same"] * 100
    pd.testing.assert_frame_equal(frame, before)


def test_optimization_converts_paths_and_preserves_empty_frames():
    source = pd.DataFrame({"path": [Path("a")] * 4, "number": [1, 2, 3, 4]})
    optimized = csv._optimize_dataframe(source)
    assert optimized.path.tolist() == ["a"] * 4
    assert isinstance(source.path.iloc[0], Path)
    assert str(optimized.path.dtype) == "category"
    pd.testing.assert_frame_equal(
        csv._optimize_dataframe(pd.DataFrame()), pd.DataFrame()
    )


def test_unknown_export_format_and_failed_trials(tmp_path, monkeypatch):
    assert csv._save_variant(
        pd.DataFrame(), str(tmp_path / "file"), "unknown", "none"
    ) == (None, None)
    monkeypatch.setattr(csv, "_get_trials", lambda: [("unknown", "none")])
    with pytest.raises(RuntimeError, match="No files"):
        csv.save_best_compressed(pd.DataFrame(), str(tmp_path / "result"))
    assert csv._parse_filename("/dir/results__csv__gzip.csv.gz") == ("csv", "gzip")
    with pytest.raises(ValueError):
        csv._parse_filename("invalid.csv")
