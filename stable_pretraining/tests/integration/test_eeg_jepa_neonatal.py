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
