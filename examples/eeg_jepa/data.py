"""Model input construction and the reference sampler bridge."""

from dataclasses import dataclass
from typing import Any
from time import perf_counter

import torch

from .config import Config


@dataclass
class Collate:
    """Preserve N-MET sampling and expose the dictionary expected by Module."""

    sampler: Any

    def __call__(self, samples: list) -> dict[str, torch.Tensor]:
        """Collate reference-format patches with optional valid-token grids."""
        result = self.sampler(samples)
        x, context, target = result[:3]
        valid = (
            result[3] if len(result) == 4 else torch.ones(x.shape[:3], dtype=torch.bool)
        )
        return dict(
            eeg=x,
            context_indices=context[0],
            target_indices=target[0],
            valid_grid=valid,
        )


def synthetic_samples(config: Config, seed: int = 0) -> list:
    """Return seeded microvolt inputs; no dataset reader or preprocessing policy."""
    generator = torch.Generator().manual_seed(seed)
    values = (
        torch.randn(
            config.total_steps * config.batch_size,
            config.channels,
            config.patches,
            200,
            generator=generator,
        )
        * 20
    )
    valid = torch.ones(values.shape[:3], dtype=torch.bool)
    valid[:, -1, -1] = False
    return list(zip(values, valid))


CANONICAL_CHANNELS = (
    "FP1",
    "FP2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "FZ",
    "CZ",
    "PZ",
)
CONTEXT_SECONDS = 12
TARGET_RATE = 200


def channel_indices(names: tuple[str, ...]) -> list[int]:
    """Select all 19 reference electrodes, rejecting ambiguous montages.

    Args:
        names: Source channel labels, including any auxiliary channels.

    Returns:
        Source indices in the reference's canonical order.

    Note:
        Unlike the permissive reference mapper, this pilot rejects bipolar,
        missing and duplicate EEG electrodes rather than silently selecting one.
    """
    aliases = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    found = {}
    for index, name in enumerate(names):
        parts = (
            name.upper()
            .strip()
            .removeprefix("EEG ")
            .replace("_", "-")
            .replace(".", "")
            .split("-")
        )
        electrode = aliases.get(parts[0], parts[0])
        if electrode not in CANONICAL_CHANNELS:
            continue
        if len(parts) > 1 and parts[1:] != ["REF"]:
            raise ValueError(f"Require referential EEG labels, got {name}")
        if electrode in found:
            raise ValueError(f"Duplicate electrode: {electrode}")
        found[electrode] = index
    missing = set(CANONICAL_CHANNELS) - found.keys()
    if missing:
        raise ValueError(f"Missing canonical electrodes: {sorted(missing)}")
    return [found[name] for name in CANONICAL_CHANNELS]


def model_patches(
    values: torch.Tensor, names: tuple[str, ...], rate_hz: float, patches: int
) -> torch.Tensor:
    """Convert contextual voltage windows into reference-format microvolt patches.

    Args:
        values: Float32 volts shaped batch, contextual time, source channels.
        names: Source channel order.
        rate_hz: Integral source rate, at least 200 Hz.
        patches: Number of one-second output patches.

    Returns:
        CPU float32 microvolts shaped batch, 19, patches, 200.

    Note:
        Shared CPU/MNE processing deliberately includes a host copy for CUDA
        input. It matches the reference transform order on contextual windows,
        not the reference's full-record boundary conditions or QC selection.
    """
    import mne
    import numpy as np

    if (
        not np.isfinite(rate_hz)
        or rate_hz < TARGET_RATE
        or int(rate_hz) != rate_hz
        or patches < 1
    ):
        raise ValueError("Require integral source rate >= 200 Hz and positive patches")
    expected = (patches + 2 * CONTEXT_SECONDS) * int(rate_hz)
    if (
        values.dtype != torch.float32
        or values.ndim != 3
        or values.shape[1:] != (expected, len(names))
    ):
        raise ValueError("Unexpected contextual voltage window shape or dtype")
    indices = channel_indices(names)
    selected = values.detach().cpu().numpy()[:, :, indices]
    if not np.isfinite(selected).all():
        raise ValueError("Nonfinite EEG values")
    info = mne.create_info(list(CANONICAL_CHANNELS), rate_hz, ch_types="eeg")
    output = []
    for window in selected:
        with mne.io.RawArray(window.T.astype(np.float64), info, verbose="ERROR") as raw:
            raw.resample(TARGET_RATE, npad="auto", method="fft", verbose="ERROR")
            raw.filter(l_freq=0.3, h_freq=75.0, verbose="ERROR")
            raw.notch_filter(freqs=[60.0], verbose="ERROR")
            start = CONTEXT_SECONDS * TARGET_RATE
            crop = raw.get_data(units="uV")[:, start : start + patches * TARGET_RATE]
        output.append(crop.reshape(19, patches, TARGET_RATE).astype(np.float32))
    return torch.from_numpy(np.stack(output))


def window_schedule(metadata: Any, config: Config, seed: int) -> Any:
    """Choose seeded whole-second core windows with full filtering context.

    Args:
        metadata: Stable-datasets source metadata.
        config: Model dimensions and number of training batches.
        seed: Request-schedule seed.

    Returns:
        Integer sample starts shaped steps, batch, for the core windows.
    """
    import numpy as np

    rate = metadata.sample_rate_hz
    if config.channels != 19 or rate < TARGET_RATE or int(rate) != rate:
        raise ValueError("Require 19 model channels and integral source rate >= 200 Hz")
    if min(config.total_steps, config.batch_size, config.patches) < 1:
        raise ValueError("Require positive steps, batch size and patches")
    channel_indices(metadata.channel_names)
    last = metadata.sample_count // int(rate) - config.patches - CONTEXT_SECONDS
    if last < CONTEXT_SECONDS:
        raise ValueError("Recording is too short for windows plus filtering context")
    seconds = np.random.default_rng(seed).integers(
        CONTEXT_SECONDS, last + 1, size=(config.total_steps, config.batch_size)
    )
    return seconds * int(rate)


def mask_schedule(reference: Any, config: Config, seed: int) -> list[dict]:
    """Create the N-MET masks once, independently of loading and dropout.

    Args:
        reference: Verified reference modules.
        config: Matching model dimensions and batch count.
        seed: CPU mask-generation seed.

    Returns:
        Per-batch dictionaries containing the identical masks for each path.
    """
    collate = Collate(reference.masks.NMETTargetSupportCollator(**config.mask_kwargs()))
    dummy = [
        (
            torch.zeros(19, config.patches, 200),
            torch.ones(19, config.patches, dtype=torch.bool),
        )
    ] * config.batch_size
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return [
            {key: value for key, value in collate(dummy).items() if key != "eeg"}
            for _ in range(config.total_steps)
        ]


class NeonatalBatches(torch.utils.data.Dataset):
    """Read complete batches through stable-datasets and apply model preprocessing."""

    def __init__(
        self,
        reader: Any,
        starts: Any,
        masks: list[dict],
        config: Config,
        diagnostic: bool = False,
    ) -> None:
        self.diagnostic = diagnostic
        self.reader, self.starts, self.masks, self.config = (
            reader,
            starts,
            masks,
            config,
        )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rate = int(self.reader.metadata.sample_rate_hz)
        started = perf_counter() if self.diagnostic else None
        values = self.reader.read(
            self.starts[index] - CONTEXT_SECONDS * rate,
            (self.config.patches + 2 * CONTEXT_SECONDS) * rate,
        )
        if self.diagnostic and values.is_cuda:
            torch.cuda.synchronize(values.device)
        read_done = perf_counter() if self.diagnostic else None
        batch = {key: value.clone() for key, value in self.masks[index].items()}
        batch["eeg"] = model_patches(
            values, self.reader.metadata.channel_names, rate, self.config.patches
        )
        if self.diagnostic:
            batch["_diagnostic"] = dict(
                loading_seconds=read_done - started,
                preprocessing_seconds=perf_counter() - read_done,
            )
        return batch
