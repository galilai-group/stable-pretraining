"""Neonatal input construction and matched real-window training controls."""

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("mne")
pytest.importorskip("edfio")
pytest.importorskip("pyedflib")
pytest.importorskip("profiling.edf_record_loading")

from profiling.edf_record_loading import EdfWindowReader
from examples.eeg_jepa.config import Config
from examples.eeg_jepa.data import (
    CANONICAL_CHANNELS,
    CONTEXT_SECONDS,
    channel_indices,
    model_patches,
    window_schedule,
)
from examples.eeg_jepa.pilot import run
from examples.eeg_jepa.reference import load_reference

pytestmark = pytest.mark.integration


@pytest.fixture
def reference_root():
    root = os.environ.get("EEG_JEPA_REFERENCE")
    if not root:
        pytest.skip("Set EEG_JEPA_REFERENCE to the external pinned reference")
    load_reference(Path(root))
    torch.set_num_threads(1)
    return Path(root)


@pytest.fixture
def neonatal_edf(tmp_path):
    from pyedflib import highlevel

    path = tmp_path / "eeg4.edf"
    rate = 256
    t = np.arange(40 * rate) / rate
    labels = [f"EEG {name}-Ref" for name in reversed(CANONICAL_CHANNELS)] + ["ECG EKG"]
    headers = [
        highlevel.make_signal_header(
            name, sample_frequency=rate, physical_min=-100 + i, physical_max=300 + i
        )
        for i, name in enumerate(labels)
    ]
    values = np.stack(
        [
            (
                2000 * np.sin(2 * np.pi * (2 + i / 10) * t)
                + 500 * np.sin(2 * np.pi * 60 * t)
                + 300 * np.sin(2 * np.pi * 90 * t)
            ).astype(np.int32)
            for i in range(len(labels))
        ]
    )
    highlevel.write_edf(str(path), values, headers, digital=True)
    return path


def test_canonical_channels_reorder_aliases_and_reject_ambiguous_montages():
    names = tuple(f"EEG {name}-Ref" for name in reversed(CANONICAL_CHANNELS)) + (
        "ECG EKG",
    )
    assert channel_indices(names) == list(range(18, -1, -1))
    aliased = tuple(
        name.replace("T3-", "T7-")
        .replace("T4-", "T8-")
        .replace("T5-", "P7-")
        .replace("T6-", "P8-")
        for name in names
    )
    assert channel_indices(aliased) == channel_indices(names)
    for invalid, message in (
        (names[1:], "Missing"),
        (names + ("FP1",), "Duplicate"),
        (tuple(name.replace("FP1-Ref", "FP1-F7") for name in names), "referential"),
    ):
        with pytest.raises(ValueError, match=message):
            channel_indices(invalid)


def test_cpu_inputs_match_independent_mne_edf_preprocessing(neonatal_edf):
    import mne

    reader = EdfWindowReader(neonatal_edf)
    starts = np.array([12, 16, 12]) * 256
    before = hashlib.sha256(neonatal_edf.read_bytes()).hexdigest()
    values = reader.read(starts - CONTEXT_SECONDS * 256, 28 * 256)
    actual = model_patches(values, reader.metadata.channel_names, 256, 4)
    expected = []
    with mne.io.read_raw_edf(str(neonatal_edf), preload=False, verbose="ERROR") as raw:
        raw.pick([f"EEG {name}-Ref" for name in CANONICAL_CHANNELS])
        for start in starts // 256:
            with raw.copy().crop(
                tmin=start - 12, tmax=start + 4 + 12 - 1 / 256
            ) as crop:
                crop.load_data(verbose="ERROR")
                crop.resample(200, npad="auto", verbose="ERROR")
                crop.filter(0.3, 75.0, verbose="ERROR")
                crop.notch_filter([60.0], verbose="ERROR")
                expected.append(
                    crop.get_data(units="uV")[:, 2400:3200].reshape(19, 4, 200)
                )
    assert actual.dtype == torch.float32 and actual.shape == (3, 19, 4, 200)
    np.testing.assert_allclose(actual.numpy(), np.stack(expected), rtol=2e-5, atol=1e-4)
    torch.testing.assert_close(actual[0], actual[2], rtol=0, atol=0)
    assert hashlib.sha256(neonatal_edf.read_bytes()).hexdigest() == before
    print(
        f"Independent MNE normalized-input max_abs={np.max(np.abs(actual.numpy() - expected)) / 100:.9g}"
    )


def test_window_schedule_retains_context_and_rejects_invalid_inputs(neonatal_edf):
    metadata = EdfWindowReader(neonatal_edf).metadata
    config = Config()
    schedule = window_schedule(metadata, config, 7)
    np.testing.assert_array_equal(schedule, window_schedule(metadata, config, 7))
    assert np.all(schedule % 256 == 0)
    assert schedule.min() >= 12 * 256
    assert schedule.max() + (config.patches + 12) * 256 <= metadata.sample_count
    for invalid in (
        replace(config, channels=18),
        replace(config, total_steps=0),
        replace(config, patches=40),
    ):
        with pytest.raises(ValueError):
            window_schedule(metadata, invalid, 7)
    with pytest.raises(ValueError, match="rate"):
        model_patches(torch.zeros(1, 28 * 128, 20), metadata.channel_names, 128, 4)
    with pytest.raises(ValueError, match="shape"):
        model_patches(torch.zeros(1, 28 * 256 - 1, 20), metadata.channel_names, 256, 4)


def test_real_cpu_pilot_replays_weights_windows_masks_and_losses(
    neonatal_edf, reference_root, tmp_path
):
    config = Config(total_steps=2, batch_size=1)
    first = run(neonatal_edf, reference_root, tmp_path / "first", config, seed=9)
    second = run(neonatal_edf, reference_root, tmp_path / "second", config, seed=9)
    for key in ("source_sha256", "core_window_starts", "masks", "checks"):
        assert first[key] == second[key]
    one, two = first["paths"]["edf_memmap"], second["paths"]["edf_memmap"]
    assert one["initial_weights_sha256"] == two["initial_weights_sha256"]
    assert one["teacher_updates"] == two["teacher_updates"] == 2
    assert [s["loss"] for s in one["steps"]] == [s["loss"] for s in two["steps"]]
    assert (tmp_path / "first/report.json").is_file()
    with pytest.raises(FileExistsError):
        run(neonatal_edf, reference_root, tmp_path / "first", config)


@pytest.mark.gpu
def test_cuda_pilot_checks_both_real_loading_paths(
    neonatal_edf, reference_root, tmp_path
):
    report = run(
        neonatal_edf,
        reference_root,
        tmp_path / "cuda",
        Config(total_steps=2, batch_size=1),
        paths=("edf_memmap", "edf_gpu_assembly"),
        accelerator="gpu",
    )
    assert set(report["checks"]["max_abs_model_input_difference"]) == {
        "edf_memmap",
        "edf_gpu_assembly",
    }
    assert (
        report["paths"]["edf_memmap"]["initial_weights_sha256"]
        == (report["paths"]["edf_gpu_assembly"]["initial_weights_sha256"])
    )
    assert all(path["teacher_updates"] == 2 for path in report["paths"].values())


def test_benchmark_case_order_and_startup_budget():
    from examples.eeg_jepa.benchmark import cases, intervals, trial_order
    from examples.eeg_jepa.config import BenchmarkConfig

    matrix = cases(("edf_memmap", "edf_gpu_assembly"), (0, 2))
    assert matrix == [("edf_memmap", 0), ("edf_memmap", 2), ("edf_gpu_assembly", 0)]
    assert trial_order(matrix, 0) == trial_order(matrix, 2) == matrix
    assert trial_order(matrix, 1) == list(reversed(matrix))
    result = intervals([2.0, 3.0, 5.0, 7.0], 2, 4)
    assert result["startup_seconds"] == 3
    assert result["sustained_seconds"] == 4
    assert result["sustained_steps"] == 2
    assert result["sustained_windows_per_second"] == 2
    for invalid in (
        BenchmarkConfig(trials=2),
        BenchmarkConfig(workers=(0,)),
        BenchmarkConfig(startup_steps=4),
    ):
        with pytest.raises(ValueError):
            invalid.validate(4)


def test_main_history_defers_synchronization_and_host_loss_copies(monkeypatch):
    from examples.eeg_jepa import benchmark

    syncs, records = [], []

    class Event:
        def __init__(self, **kwargs):
            self.index = len(records)
            records.append(self)

        def record(self):
            pass

        def elapsed_time(self, end):
            return (end.index - self.index) * 1000.0

    monkeypatch.setattr(benchmark, "_sync", lambda device: syncs.append(device))
    monkeypatch.setattr(torch.cuda, "Event", Event)
    history = benchmark.History(torch.device("cuda"))
    history.begin()
    original_cpu = torch.Tensor.cpu

    def no_cpu_copy(*args, **kwargs):
        raise AssertionError("Main batch hook must not copy losses to the host")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", no_cpu_copy)
        history.on_train_batch_start(None, None, {}, 0)
        history.on_train_batch_end(None, None, {"loss": torch.tensor(1.0)}, {}, 0)
        history.on_train_batch_end(None, None, {"loss": torch.tensor(2.0)}, {}, 1)
    assert torch.Tensor.cpu is original_cpu
    assert len(syncs) == 1
    history.on_train_end(None, None)
    assert len(syncs) == 2
    assert history.values == [1.0, 2.0]
    assert history.ends == [1.0, 2.0]


def test_benchmark_settings_restore_requeue_policy(tmp_path):
    import stable_pretraining as spt
    from examples.eeg_jepa.config import run_settings

    settings = spt.get_config()
    before = (
        settings.requeue_checkpoint,
        settings.default_callbacks,
        settings.default_loggers,
        settings.cache_dir,
    )
    with pytest.raises(RuntimeError, match="injected"):
        with run_settings(tmp_path / "benchmark-settings", benchmark=True):
            assert settings.requeue_checkpoint is False
            raise RuntimeError("injected")
    assert (
        settings.requeue_checkpoint,
        settings.default_callbacks,
        settings.default_loggers,
        settings.cache_dir,
    ) == before


def _assert_benchmark(report, output, expected_cases):
    data = report["benchmark"]
    assert len(data["trials"]) == 3 * expected_cases
    assert len(data["diagnostics"]) == expected_cases
    assert len(data["summary"]) == expected_cases
    assert data["trial_order"][0] == data["trial_order"][2]
    assert data["trial_order"][1] == list(reversed(data["trial_order"][0]))
    assert (
        data["worker_input_sha256"]["2"] == report["checks"]["cpu_model_input_sha256"]
    )
    all_runs = data["trials"] + data["diagnostics"]
    assert len({row["initial_weights_sha256"] for row in all_runs}) == 1
    assert data["parameters"]["total"] > data["parameters"]["trainable"] > 0
    for row in all_runs:
        assert row["teacher_updates"] == 3
        assert row["max_abs_loss_difference"] <= 2e-4
        assert len(row["steps"]) == 3
        assert row["intervals"]["sustained_steps"] == 2
        assert row["intervals"]["startup_steps"] == 1
        assert row["intervals"]["sustained_windows_per_second"] > 0
        assert (
            row["workers"] == 0
            if row["path"] == "edf_gpu_assembly"
            else row["workers"] in (0, 2)
        )
    for row in data["trials"]:
        assert row["diagnostic_phases"] is None
    for row in data["diagnostics"]:
        assert len(row["diagnostic_phases"]) == 3
        assert all(
            all(value >= 0 for value in phase.values())
            for phase in row["diagnostic_phases"]
        )
    assert not list(output.rglob("*.ckpt"))
    from PIL import Image

    with Image.open(output / "learning-progress.png") as image:
        assert image.width > 1000 and image.height > 500
    import json

    records = [
        json.loads(line) for line in (output / "trials.jsonl").read_text().splitlines()
    ]
    assert len(records) == 4 * expected_cases


def test_repeated_benchmark_reopens_cpu_workers_and_preserves_learning(
    neonatal_edf, reference_root, tmp_path
):
    from examples.eeg_jepa.config import BenchmarkConfig

    output = tmp_path / "repeated-cpu"
    report = run(
        neonatal_edf,
        reference_root,
        output,
        Config(total_steps=3, batch_size=1),
        benchmark=BenchmarkConfig(trials=3, startup_steps=1, diagnostic=True),
    )
    _assert_benchmark(report, output, 2)


@pytest.mark.gpu
def test_cuda_benchmark_matches_workers_and_main_process_assembly(
    neonatal_edf, reference_root, tmp_path
):
    from examples.eeg_jepa.config import BenchmarkConfig

    output = tmp_path / "repeated-cuda"
    report = run(
        neonatal_edf,
        reference_root,
        output,
        Config(total_steps=3, batch_size=1),
        paths=("edf_memmap", "edf_gpu_assembly"),
        accelerator="gpu",
        benchmark=BenchmarkConfig(trials=3, startup_steps=1, diagnostic=True),
    )
    _assert_benchmark(report, output, 3)
