"""Check development selection and reuse of the native probe lifecycle."""

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.unit
SCRIPT = Path(__file__).resolve().parents[3] / "benchmarks/imagenet100/probe_sweep.py"
spec = importlib.util.spec_from_file_location("directclr_probe_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


@pytest.fixture(autouse=True)
def restore_settings():
    config = sweep.benchmark.spt.get_config()
    previous = copy.deepcopy(vars(config))
    threads = torch.get_num_threads()
    try:
        yield
    finally:
        vars(config).clear()
        vars(config).update(previous)
        sweep.benchmark.spt.set(verbose=config.verbose, log_rank=config.log_rank)
        torch.set_num_threads(threads)


def test_split_is_reproducible_stratified_and_covers_each_row():
    labels = torch.tensor([0] * 43 + [1] * 37 + [2] * 29)
    split = sweep._stratified_split(labels)
    again = sweep._stratified_split(labels)
    sweep._validate_split(split, labels)
    assert torch.equal(split["fit_indices"], again["fit_indices"])
    assert torch.equal(split["development_indices"], again["development_indices"])
    assert torch.bincount(labels[split["development_indices"]]).tolist() == [4, 4, 3]
    other = sweep._stratified_split(labels, seed=7)
    assert not torch.equal(split["development_indices"], other["development_indices"])


def test_rejects_reordered_labels_and_overlapping_split():
    labels = torch.arange(3).repeat_interleave(10)
    split = sweep._stratified_split(labels)
    with pytest.raises(ValueError, match="ordering"):
        sweep._validate_split(split, labels.flip(0))
    split["development_indices"][0] = split["fit_indices"][0]
    with pytest.raises(ValueError, match="without overlap"):
        sweep._validate_split(split, labels)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan")])
def test_rejects_invalid_development_fraction(fraction):
    with pytest.raises(ValueError, match="fraction"):
        sweep._stratified_split(torch.zeros(10, dtype=torch.long), fraction)


def test_rejects_singleton_classes():
    with pytest.raises(ValueError, match="two training samples"):
        sweep._stratified_split(torch.tensor([0, 0, 1]))


@pytest.mark.parametrize("seed,fraction", [(7, 0.1), (2026, 0.2)])
def test_preparation_rejects_a_different_registered_protocol(
    tmp_path, monkeypatch, seed, fraction
):
    labels = torch.arange(3).repeat_interleave(20)
    split = sweep._stratified_split(labels, fraction=fraction, seed=seed)
    source = tmp_path / "source"
    evaluation = source / "imagenet100-r50-100ep-directclr-seed0/evaluation"
    evaluation.mkdir(parents=True)
    torch.save(
        {"x_train": torch.ones(60, 4), "y_train": labels}, evaluation / "features.pt"
    )
    split_path = tmp_path / "split.pt"
    torch.save(split, split_path)
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "prepare-split",
            "--source",
            str(source),
            "--split",
            str(split_path),
            "--output",
            str(output),
            "--accelerator",
            "cpu",
        ],
    )
    with pytest.raises(ValueError, match="preregistered"):
        sweep.main()
    assert not output.exists()


def test_rejects_empty_feature_width():
    with pytest.raises(ValueError, match="nonempty"):
        sweep._training_data(
            {"x_train": torch.ones(6, 0), "y_train": torch.arange(3).repeat(2)}
        )


def test_prepare_reads_the_requested_encoder_schedule(tmp_path, monkeypatch):
    source = tmp_path / "source"
    evaluation = source / "imagenet100-r50-200ep-directclr-seed0/evaluation"
    evaluation.mkdir(parents=True)
    labels = torch.arange(3).repeat_interleave(20)
    torch.save(
        {"x_train": torch.ones(60, 4), "y_train": labels}, evaluation / "features.pt"
    )
    split_path = tmp_path / "split.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "prepare-split",
            "--source",
            str(source),
            "--split",
            str(split_path),
            "--output",
            str(tmp_path / "output"),
            "--encoder-epochs",
            "200",
            "--accelerator",
            "cpu",
        ],
    )
    sweep.main()
    split = torch.load(split_path, weights_only=True)
    sweep._validate_registered_split(split, labels)
    assert len(split["fit_indices"]) + len(split["development_indices"]) == 60


def test_training_selection_does_not_access_held_out_tensors():
    class GuardedCache(dict):
        def __getitem__(self, key):
            assert key not in {"x_val", "y_val"}, "held-out access during selection"
            return super().__getitem__(key)

    cache = GuardedCache(
        x_train=torch.ones(6, 4), y_train=torch.tensor([0, 0, 1, 1, 2, 2])
    )
    x, y = sweep._training_data(cache)
    assert x.shape == (6, 4) and y.shape == (6,)
    cache["x_train"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        sweep._training_data(cache)


def test_select_uses_development_score_and_deterministic_tie_break():
    candidates = [
        {"learning_rate": 10.0, "development_top1": 0.6},
        {"learning_rate": 1.0, "development_top1": 0.5},
        {"learning_rate": 3.0, "development_top1": 0.6},
    ]
    assert sweep._select(candidates)["learning_rate"] == 3.0
    with pytest.raises(ValueError, match="finite"):
        sweep._select([{"development_top1": float("nan"), "learning_rate": 1}])


def test_gpu_worker_rejects_an_unassigned_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="exactly one CUDA device"):
        sweep._gpu_smoke(argparse.Namespace())


def test_gpu_worker_rejects_a_model_other_than_requested(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: argparse.Namespace(name="NVIDIA L40"),
    )
    with pytest.raises(RuntimeError, match="expected NVIDIA L40S, received NVIDIA L40"):
        sweep._gpu_smoke(argparse.Namespace(expected_gpu_name="NVIDIA L40S"))


@pytest.mark.parametrize("expected", [None, "NVIDIA L40S"])
def test_smoke_cli_passes_the_requested_gpu_model(tmp_path, monkeypatch, expected):
    observed = []
    monkeypatch.setattr(
        sweep, "_gpu_smoke", lambda args: observed.append(args.expected_gpu_name)
    )
    monkeypatch.setattr(sweep, "_run_method", lambda *_: None)
    arguments = [
        str(SCRIPT),
        "sweep",
        "--source",
        str(tmp_path),
        "--split",
        str(tmp_path / "split.pt"),
        "--output",
        str(tmp_path / "output"),
        "--methods",
        "directclr",
        "--smoke-first",
    ]
    if expected is not None:
        arguments += ["--expected-gpu-name", expected]
    monkeypatch.setattr(sys, "argv", arguments)
    sweep.main()
    assert observed == [expected]


@pytest.mark.parametrize("include_option", [False, True])
def test_gpu_smoke_accepts_other_models_when_no_model_is_required(
    tmp_path, monkeypatch, include_option
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: argparse.Namespace(name="NVIDIA A100-SXM4-40GB", uuid="test-device"),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def fit(features, labels, classes, manifest, output, lr, args):
        output.mkdir()
        return object()

    monkeypatch.setattr(sweep, "_fit", fit)
    monkeypatch.setattr(sweep, "_accuracy", lambda *_: {"top1": 1.0})
    args = argparse.Namespace(output=tmp_path, methods=["directclr"])
    if include_option:
        args.expected_gpu_name = None
    sweep._gpu_smoke(args)
    result = json.loads((tmp_path / "smoke-directclr/gpu-smoke.json").read_text())
    assert result["device"] == "NVIDIA A100-SXM4-40GB"
    assert result["visible_devices"] == 1


def test_changing_held_out_labels_cannot_change_selected_recipe(tmp_path):
    generator = torch.Generator().manual_seed(12)
    labels = torch.arange(3).repeat_interleave(24)
    x = torch.nn.functional.one_hot(labels, 4).float()
    x += 0.01 * torch.randn(x.shape, generator=generator)
    val_labels = torch.arange(3).repeat_interleave(4)
    val = torch.nn.functional.one_hot(val_labels, 4).float()
    source = tmp_path / "source"
    evaluation = source / "imagenet100-r50-100ep-directclr-seed0/evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "feature_manifest.json").write_text(
        json.dumps({"checkpoint_sha256": "test"})
    )
    split_path = tmp_path / "split.pt"
    torch.save(sweep._stratified_split(labels), split_path)
    selections, scores = [], []
    for index, held_out in enumerate((val_labels, (val_labels + 1) % 3)):
        torch.save(
            {"x_train": x, "y_train": labels, "x_val": val, "y_val": held_out},
            evaluation / "features.pt",
        )
        args = argparse.Namespace(
            source=source,
            split=split_path,
            output=tmp_path / f"run-{index}",
            learning_rates=[0.1, 1.0],
            epochs=20,
            batch_size=12,
            seed=0,
            cpu_threads=1,
            accelerator="cpu",
        )
        sweep._run_method("directclr", args)
        root = args.output / "directclr"
        selections.append(json.loads((root / "selection.json").read_text()))
        scores.append(json.loads((root / "evaluation.json").read_text())["linear_top1"])
        assert (root / "linear_classifier.pt").is_file()
    assert selections[0] == selections[1]
    assert selections[0]["final_validation_used_for_selection"] is False
    assert scores[0] > 0.95 and scores[1] < 0.05
