"""Matched neonatal EDF training pilot; all source I/O stays in stable-datasets."""

import argparse
import hashlib
import json
from dataclasses import asdict
from importlib.metadata import version
import platform
from pathlib import Path
from time import perf_counter

import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

from profiling import edf_record_loading, edf_source, signal_calibration
from profiling.edf_record_loading import EDF_PATHS, EdfWindowReader

from .config import Config
from .data import (
    CANONICAL_CHANNELS,
    CONTEXT_SECONDS,
    NeonatalBatches,
    channel_indices,
    mask_schedule,
    window_schedule,
)
from .model import build_module
from .reference import REFERENCE_REVISION, load_reference
from .train import fit


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _external(path: Path) -> Path:
    path = path.expanduser().resolve()
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("Reference caches and pilot artifacts must be outside Git")
    return path


def verify_inputs(
    datasets: dict[str, NeonatalBatches], device: torch.device = torch.device("cpu")
) -> dict:
    """Check every requested batch against CPU memmap at the encoder input scale.

    Args:
        datasets: Matched schedules, including the CPU memmap baseline.
        device: Shared model device on which the forward divides inputs by 100.

    Returns:
        Maximum absolute errors after division by 100 and baseline input hashes.
    """
    errors = {name: 0.0 for name in datasets}
    hashes = []
    baseline = datasets["edf_memmap"]
    for index in range(len(baseline)):
        expected = (baseline[index]["eeg"].to(device) / 100.0).cpu()
        hashes.append(hashlib.sha256(expected.numpy().tobytes()).hexdigest())
        for name, dataset in datasets.items():
            if name == "edf_memmap":
                continue
            actual = (dataset[index]["eeg"].to(device) / 100.0).cpu()
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
            errors[name] = max(errors[name], float((actual - expected).abs().max()))
    return dict(
        max_abs_model_input_difference=errors,
        compared_paths=[name for name in datasets if name != "edf_memmap"],
        model_input_device=str(device),
        input_tolerance=dict(rtol=2e-5, atol=1e-6),
        cpu_model_input_sha256=hashes,
    )


class PilotHistory(pl.Callback):
    """Retain per-step losses and elapsed time without external tracking."""

    def __init__(self) -> None:
        self.steps = []

    def on_train_start(self, trainer, pl_module) -> None:
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
        self.started = perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        loss = float(outputs["loss"].detach().cpu())
        if not np.isfinite(loss):
            raise ValueError("Nonfinite pilot loss")
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
        self.steps.append(
            dict(loss=loss, elapsed_seconds=perf_counter() - self.started)
        )


def _train_paths(reference, datasets, paths, config, output, accelerator, seed):
    results = {}
    for name in paths:
        pl.seed_everything(seed, workers=True)
        module = build_module(reference, config)
        initial_hash = hashlib.sha256()
        for value in module.state_dict().values():
            initial_hash.update(value.detach().cpu().numpy().tobytes())
        history = PilotHistory()
        loader = DataLoader(
            datasets[name],
            batch_size=None,
            num_workers=0,
            pin_memory=accelerator == "gpu",
        )
        fit(module, loader, output / name, config, accelerator, [history], seed=seed)
        if int(module.eeg.completed_steps) != config.total_steps:
            raise ValueError("Teacher update count differs from the training budget")
        results[name] = dict(
            initial_weights_sha256=initial_hash.hexdigest(),
            teacher_updates=int(module.eeg.completed_steps),
            steps=history.steps,
        )
    _compare_runs(results, config.total_steps)
    return results


def _compare_runs(results: dict, steps: int) -> None:
    if len({result["initial_weights_sha256"] for result in results.values()}) != 1:
        raise ValueError("Initial weights differ across paths")
    baseline = results.get("edf_memmap", next(iter(results.values())))
    reference_losses = [row["loss"] for row in baseline["steps"]]
    for result in results.values():
        losses = [row["loss"] for row in result["steps"]]
        if len(losses) != steps:
            raise ValueError("Incomplete pilot loss trace")
        np.testing.assert_allclose(losses, reference_losses, rtol=2e-4, atol=2e-5)
        result["max_abs_loss_difference"] = max(
            abs(actual - expected)
            for actual, expected in zip(losses, reference_losses, strict=True)
        )


def run(
    source: Path,
    reference_root: Path,
    output: Path,
    config: Config,
    *,
    paths: tuple[str, ...] = ("edf_memmap",),
    accelerator: str = "cpu",
    seed: int = 0,
) -> dict:
    """Validate matched inputs, then train fresh models on identical EDF requests.

    Args:
        source: Unchanged neonatal EDF recording.
        reference_root: External pinned reference archive.
        output: New external pilot directory.
        config: Small model dimensions, optimizer schedule and training budget.
        paths: Existing stable-datasets paths to train.
        accelerator: Shared training endpoint, cpu or gpu.
        seed: Shared initialization, window and mask seed.

    Returns:
        Local provenance, input checks and matched per-step training results.

    Note:
        CPU/MNE preprocessing includes a CUDA-to-host copy for GPU assembly.
        Timings are a correctness diagnostic, not an end-to-end speedup claim.
    """
    if (
        not paths
        or len(set(paths)) != len(paths)
        or any(p not in EDF_PATHS for p in paths)
    ):
        raise ValueError("Require distinct supported EDF paths")
    if accelerator not in ("cpu", "gpu"):
        raise ValueError("Require cpu or gpu accelerator")
    if "edf_gpu_assembly" in paths and accelerator != "gpu":
        raise ValueError("GPU assembly requires the shared GPU training endpoint")
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    output, reference_root = _external(output), _external(reference_root)
    reference = load_reference(reference_root)
    before = _checksum(source)
    readers = {"edf_memmap": EdfWindowReader(source)}
    if "edf_gpu_assembly" in paths:
        readers["edf_gpu_assembly"] = EdfWindowReader(
            source, "edf_gpu_assembly", torch.device("cuda")
        )
    metadata = readers["edf_memmap"].metadata
    starts = window_schedule(metadata, config, seed)
    masks = mask_schedule(reference, config, seed)
    datasets = {
        name: NeonatalBatches(reader, starts, masks, config)
        for name, reader in readers.items()
    }
    output.mkdir(parents=True, exist_ok=False)
    checks = verify_inputs(
        datasets, torch.device("cuda" if accelerator == "gpu" else "cpu")
    )
    results = _train_paths(
        reference, datasets, paths, config, output, accelerator, seed
    )
    if _checksum(source) != before:
        raise ValueError("Source changed during the pilot")
    report = dict(
        scope="Single-recording, contextual-window correctness pilot; shared CPU/MNE preprocessing; no held-out evaluation or speedup claim",
        reference_revision=REFERENCE_REVISION,
        source_sha256=before,
        source_name=source.name,
        source_rate_hz=metadata.sample_rate_hz,
        source_channels=list(metadata.channel_names),
        selected_source_indices=channel_indices(metadata.channel_names),
        model_channels=list(CANONICAL_CHANNELS),
        context_seconds=CONTEXT_SECONDS,
        preprocessing="MNE FFT resample to 200 Hz; FIR 0.3-75 Hz; FIR 60 Hz notch; crop context; volts to microvolts; 200-sample patches; trainer divides by 100",
        config=asdict(config),
        seed=seed,
        accelerator=accelerator,
        core_window_starts=starts.tolist(),
        masks=[
            {key: value.tolist() for key, value in batch.items()} for batch in masks
        ],
        checks=checks,
        paths=results,
        loss_tolerance=dict(rtol=2e-4, atol=2e-5),
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            lightning=pl.__version__,
            numpy=np.__version__,
            mne=version("mne"),
            scipy=version("scipy"),
        ),
    )
    report["code_sha256"] = {
        str(path): _checksum(path)
        for path in (
            *sorted(Path(__file__).parent.glob("*.py")),
            Path(edf_record_loading.__file__),
            Path(edf_source.__file__),
            Path(signal_calibration.__file__),
        )
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Run a short local pilot; leave cluster job submission to Pascal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths", nargs="+", choices=EDF_PATHS, default=["edf_memmap"])
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seconds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run(
        args.source,
        args.reference_root,
        args.output,
        Config(
            total_steps=args.steps, batch_size=args.batch_size, patches=args.seconds
        ),
        paths=tuple(args.paths),
        accelerator=args.accelerator,
        seed=args.seed,
    )
    print(json.dumps(report["checks"], indent=2))


if __name__ == "__main__":
    main()
