"""Independent MNE checks for the isolated experimental GPU preprocessing route."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.eeg_jepa import gpu_preprocessing as gpu
from examples.eeg_jepa.config import Config, GPU_PREPROCESSING_CASE
from examples.eeg_jepa.data import CANONICAL_CHANNELS

mne = pytest.importorskip("mne")

pytestmark = pytest.mark.integration

NAMES = tuple(f"EEG {name}-Ref" for name in reversed(CANONICAL_CHANNELS)) + ("ECG EKG",)


def _values(batch_size):
    generator = torch.Generator().manual_seed(17)
    values = torch.randn(4, 7168, 20, generator=generator) * 2e-5
    t = torch.arange(7168, dtype=torch.float64) / 256
    values += (1e-5 * torch.sin(2 * torch.pi * 60 * t)).float()[None, :, None]
    values += (1e-5 * torch.sin(2 * torch.pi * 90 * t)).float()[None, :, None]
    values += torch.linspace(-2e-5, 2e-5, 20)
    values[:, [0, 3071, 3072, 4095, 4096, -1], :] += 1e-3
    values[:, :, -1] = 7  # An auxiliary signal must not enter the model montage.
    return values[torch.tensor([2, 0, 3, 2][:batch_size])]


def _mne_reference(values):
    results = []
    for window in values.numpy():
        info = mne.create_info(NAMES, 256, ch_types="eeg")
        with mne.io.RawArray(window.T.astype(np.float64), info, verbose="ERROR") as raw:
            raw.pick([f"EEG {name}-Ref" for name in CANONICAL_CHANNELS])
            raw.resample(200, npad="auto", method="fft", verbose="ERROR")
            raw.filter(0.3, 75, verbose="ERROR")
            raw.notch_filter([60], verbose="ERROR")
            results.append(raw.get_data(units="uV")[:, 2400:3200].reshape(19, 4, 200))
    return torch.from_numpy(np.stack(results).astype(np.float32))


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_tensor_kernels_on_cpu_match_independent_mne(batch_size):
    values = _values(batch_size)
    before = values.clone()
    expected = _mne_reference(values)
    plan = gpu._make_plan(NAMES, torch.device("cpu"))
    actual = gpu._process(values, plan)
    torch.testing.assert_close(actual / 100, expected / 100, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(values, before, rtol=0, atol=0)
    assert actual.shape == (batch_size, 19, 4, 200) and actual.dtype == torch.float32
    if batch_size == 4:
        torch.testing.assert_close(actual[0], actual[3], rtol=0, atol=0)
    print(
        f"CPU tensor kernels vs MNE normalized max_abs={float(((actual - expected) / 100).abs().max()):.9g}; batch={batch_size}"
    )


def test_resampling_matches_mne_including_even_nyquist_and_endpoint_padding():
    t = torch.arange(7168, dtype=torch.float64) / 256
    values = (torch.cos(2 * torch.pi * 100 * t) + torch.cos(2 * torch.pi * 128 * t))[
        None
    ]
    values[:, [0, -1]] += 4
    actual = gpu._resample(values)
    expected = mne.filter.resample(
        values.numpy(), up=200, down=256, npad="auto", verbose="ERROR"
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)
    assert actual.shape[-1] == 5600


@pytest.mark.parametrize("index", [0, 1])
def test_fft_convolution_matches_independent_mne_zero_phase_filter(index):
    generator = torch.Generator().manual_seed(22)
    values = torch.randn(2, 5600, generator=generator, dtype=torch.float64) * 2e-5
    values[:, [0, 2399, 2400, 3199, 3200, -1]] += 1e-3
    plan = gpu._make_plan(NAMES, torch.device("cpu"))
    actual = gpu._apply_fir(values, plan.filters[index])
    if index == 0:
        expected = mne.filter.filter_data(values.numpy(), 200, 0.3, 75, verbose="ERROR")
    else:
        expected = mne.filter.notch_filter(values.numpy(), 200, [60], verbose="ERROR")
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-11, atol=1e-16)


@pytest.mark.parametrize(
    "rate,patches,names",
    [
        (200, 4, NAMES),
        (512, 4, NAMES),
        (256, 1, NAMES),
        (256, 8, NAMES),
        (256, 4, NAMES[1:]),
        (256, 4, NAMES + ("FP1",)),
    ],
)
def test_unsupported_gpu_configurations_fail_before_cuda_setup(rate, patches, names):
    with pytest.raises(ValueError, match="256 Hz|Missing|Duplicate"):
        gpu.GpuPreprocessor(names, rate, patches, torch.device("cuda"))


def test_public_candidate_requires_cuda():
    with pytest.raises(ValueError, match="requires CUDA"):
        gpu.GpuPreprocessor(NAMES, 256, 4, torch.device("cpu"))


def test_gpu_setup_rejects_workers(monkeypatch):
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
    with pytest.raises(ValueError, match="main process"):
        gpu.GpuPreprocessor(NAMES, 256, 4, torch.device("cuda"))


def test_unreviewed_mne_version_is_rejected(monkeypatch):
    monkeypatch.setattr(mne, "__version__", "unreviewed")
    with pytest.raises(ValueError, match="MNE 1.10.2"):
        gpu._make_plan(NAMES, torch.device("cpu"))


def test_candidate_worker_matrix_and_loader_keep_cuda_in_main_process():
    from examples.eeg_jepa.benchmark import History, _loader, cases

    paths = ("current_cache", "edf_memmap", "edf_gpu_assembly", GPU_PREPROCESSING_CASE)
    assert cases(paths, (0, 2)) == [
        ("current_cache", 0),
        ("current_cache", 2),
        ("edf_memmap", 0),
        ("edf_memmap", 2),
        ("edf_gpu_assembly", 0),
        (GPU_PREPROCESSING_CASE, 0),
    ]

    class Candidate(torch.utils.data.Dataset):
        reader = SimpleNamespace(loading_path="edf_gpu_assembly")
        preprocessor = object()

        def __len__(self):
            return 1

    dataset = Candidate()
    loader = _loader(dataset, 0, 0, History(torch.device("cuda")))
    assert loader.pin_memory is False
    with pytest.raises(ValueError, match="cannot run in workers"):
        _loader(dataset, 2, 0)


@pytest.mark.parametrize(
    "paths,accelerator,patches",
    [
        (("edf_memmap",), "cpu", 4),
        (("edf_memmap",), "gpu", 4),
        (("edf_gpu_assembly",), "gpu", 3),
    ],
)
def test_pilot_rejects_invalid_opt_in_before_creating_artifacts(
    tmp_path, paths, accelerator, patches
):
    from examples.eeg_jepa.pilot import run

    output = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="GPU preprocessing requires"):
        run(
            tmp_path / "unused.edf",
            tmp_path / "unused-reference",
            output,
            Config(patches=patches),
            paths=paths,
            accelerator=accelerator,
            gpu_preprocessing=True,
        )
    assert not output.exists()


@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_cuda_preprocessing_matches_mne_without_host_round_trip(
    batch_size, monkeypatch
):
    values = _values(batch_size)
    expected = _mne_reference(values)
    preprocessor = gpu.GpuPreprocessor(NAMES, 256, 4, torch.device("cuda"))
    device_values = values.cuda()
    original_to = torch.Tensor.to

    def no_host_access(*args, **kwargs):
        raise AssertionError(
            "Per-batch preprocessing must not read signals or scalars on host"
        )

    def gpu_only_to(self, *args, **kwargs):
        result = original_to(self, *args, **kwargs)
        if self.is_cuda and not result.is_cuda:
            no_host_access()
        return result

    with monkeypatch.context() as patch:
        for method in ("cpu", "numpy", "item", "tolist"):
            patch.setattr(torch.Tensor, method, no_host_access)
        patch.setattr(torch.Tensor, "to", gpu_only_to)
        actual = preprocessor(device_values)
    assert actual.is_cuda and actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu() / 100, expected / 100, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(device_values.cpu(), values, rtol=0, atol=0)
    print(
        f"CUDA vs independent MNE normalized max_abs={float(((actual.cpu() - expected) / 100).abs().max()):.9g}; batch={batch_size}"
    )
    with pytest.raises(ValueError, match="CUDA device"):
        preprocessor(values)
    with pytest.raises(ValueError, match="contextual windows"):
        preprocessor(device_values[:, :-1])
