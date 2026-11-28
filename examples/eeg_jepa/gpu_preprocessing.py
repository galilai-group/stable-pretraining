"""Experimental float64 FFT preprocessing for fixed neonatal GPU windows."""

import hashlib
from dataclasses import dataclass
from time import perf_counter

import torch

from .config import GPU_PREPROCESSING_ID
from .data import channel_indices


@dataclass(frozen=True)
class _Fir:
    response: torch.Tensor
    edge: int
    n_fft: int


@dataclass(frozen=True)
class _Plan:
    indices: torch.Tensor
    filters: tuple[_Fir, _Fir]
    coefficient_hashes: tuple[str, str]


def _validate_configuration(names, rate_hz, patches):
    if rate_hz != 256 or patches != 4:
        raise ValueError("GPU preprocessing requires 256 Hz and four-second cores")
    return channel_indices(names)


def _reflect(values, edge):
    # Supported lengths never need reflect_limited's additional zero padding.
    return torch.cat(
        (
            2 * values[..., :1] - values[..., 1 : edge + 1].flip(-1),
            values,
            2 * values[..., -1:] - values[..., -edge - 1 : -1].flip(-1),
        ),
        dim=-1,
    )


def _resample(values):
    # MNE npad='auto': 7168 -> 8192, resample to 6400, remove 400 per side.
    spectrum = torch.fft.rfft(_reflect(values, 512), dim=-1)[..., :3201].clone()
    spectrum[..., -1] *= 2  # Downsampling merges the new even-length Nyquist pair.
    return torch.fft.irfft(spectrum * (6400 / 8192), n=6400, dim=-1)[..., 400:6000]


def _apply_fir(values, fir):
    extended = _reflect(values, fir.edge)
    filtered = torch.fft.irfft(
        torch.fft.rfft(extended, n=fir.n_fft, dim=-1) * fir.response,
        n=fir.n_fft,
        dim=-1,
    )
    # Same linear convolution and group-delay compensation as MNE phase='zero'.
    shift = fir.edge + fir.edge // 2
    return filtered[..., shift : shift + values.shape[-1]]


def _make_plan(names, device):
    import mne

    if mne.__version__ != "1.10.2":
        raise ValueError(
            "Experimental GPU preprocessing requires MNE 1.10.2 coefficients"
        )
    bandpass = mne.filter.create_filter(None, 200, 0.3, 75.0, verbose="ERROR")
    # MNE's default notch width is 60/200 Hz, with 0.5 Hz transitions per edge.
    notch = mne.filter.create_filter(
        None,
        200,
        [60.65],
        [59.35],
        l_trans_bandwidth=0.5,
        h_trans_bandwidth=0.5,
        verbose="ERROR",
    )
    filters = []
    for coefficients in (bandpass, notch):
        edge = len(coefficients) - 1
        n_fft = 1 << (5600 + 3 * edge - 1).bit_length()
        response = torch.fft.rfft(torch.from_numpy(coefficients).to(device), n=n_fft)
        filters.append(_Fir(response, edge, n_fft))
    return _Plan(
        torch.tensor(channel_indices(names), dtype=torch.long, device=device),
        tuple(filters),
        tuple(hashlib.sha256(h.tobytes()).hexdigest() for h in (bandpass, notch)),
    )


def _process(values, plan):
    selected = (
        values.detach().index_select(2, plan.indices).transpose(1, 2).to(torch.float64)
    )
    processed = _resample(selected)
    for fir in plan.filters:
        processed = _apply_fir(processed, fir)
    return (processed[..., 2400:3200] * 1e6).to(torch.float32).reshape(-1, 19, 4, 200)


class GpuPreprocessor:
    """Prepare fixed-shape GPU kernels without changing the CPU/MNE baseline."""

    def __init__(
        self, names: tuple[str, ...], rate_hz: float, patches: int, device: torch.device
    ) -> None:
        """Design reference FIRs and prepare device coefficients outside batch timing.

        Args:
            names: Source labels containing the 19 canonical referential electrodes.
            rate_hz: Supported source rate, exactly 256 Hz.
            patches: Supported core duration, exactly four one-second patches.
            device: CUDA device shared by EDF assembly and training.
        """
        _validate_configuration(names, rate_hz, patches)
        if torch.utils.data.get_worker_info() is not None:
            raise ValueError("GPU preprocessing must run in the main process")
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("GPU preprocessing requires CUDA")
        self.device = torch.device(
            "cuda",
            torch.cuda.current_device() if device.index is None else device.index,
        )
        self.source_channels = len(names)
        torch.cuda.synchronize(self.device)
        started = perf_counter()
        self.plan = _make_plan(names, self.device)
        torch.cuda.synchronize(self.device)
        self.provenance = dict(
            identifier=GPU_PREPROCESSING_ID,
            status="experimental; actual CUDA parity and real-recording validation required",
            coefficient_mne_version="1.10.2",
            preparation_seconds=perf_counter() - started,
            excluded_preparation="CPU FIR design and channel mapping; coefficient/index upload and FIR spectra; preflight checks",
            signal_dtype="float64 FFT processing; float32 microvolt patches",
            source_rate_hz=256,
            target_rate_hz=200,
            core_seconds=4,
            context_seconds=12,
            resampling="boxcar FFT; reflect_limited 512 per side; 8192 to 6400; crop 400 per side; even Nyquist correction",
            filtering="MNE firwin/Hamming coefficients; bandpass then notch; reflect_limited; zero-phase linear convolution",
            fir_lengths=[f.edge + 1 for f in self.plan.filters],
            fir_fft_lengths=[f.n_fft for f in self.plan.filters],
            coefficient_sha256=list(self.plan.coefficient_hashes),
            host_signal_round_trip=False,
        )

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        """Construct GPU microvolt patches without host signal copies or scalar reads.

        Args:
            values: CUDA float32 volts shaped batch, 7168, source channels.

        Returns:
            CUDA float32 microvolts shaped batch, 19, 4, 200.
        """
        if torch.utils.data.get_worker_info() is not None:
            raise ValueError("GPU preprocessing must run in the main process")
        if values.device != self.device or values.dtype != torch.float32:
            raise ValueError(
                "GPU preprocessing requires float32 volts on its CUDA device"
            )
        if (
            values.ndim != 3
            or values.shape[0] < 1
            or values.shape[1:] != (7168, self.source_channels)
        ):
            raise ValueError("GPU preprocessing requires 28-second contextual windows")
        return _process(values, self.plan)
