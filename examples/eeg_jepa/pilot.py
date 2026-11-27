"""Matched neonatal EDF training pilot; all source I/O stays in stable-datasets."""

import argparse
import hashlib
import json
from dataclasses import asdict
from importlib.metadata import version
import platform
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from profiling import (
    edf_record_loading,
    edf_source,
    signal_calibration,
    training_windows,
)
from profiling.edf_record_loading import EDF_PATHS, EdfWindowReader

from .config import BenchmarkConfig, Config
from .data import (
    CANONICAL_CHANNELS,
    CONTEXT_SECONDS,
    NeonatalBatches,
    channel_indices,
    mask_schedule,
    window_schedule,
)
from .reference import REFERENCE_REVISION, load_reference


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


def _train_paths(reference, datasets, paths, config, output, accelerator, seed):
    # Spawn re-executes this module; only the training process needs these imports.
    from .benchmark import PilotHistory
    from .train import fit, initialize_module, validate_runs

    results = {}
    for name in paths:
        module, initial_hash = initialize_module(reference, config, seed)
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
            initial_weights_sha256=initial_hash,
            teacher_updates=int(module.eeg.completed_steps),
            steps=history.steps,
        )
    validate_runs(results, config.total_steps)
    return results


def _validate_options(paths, accelerator, benchmark, steps):
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
    if benchmark is not None:
        benchmark.validate(steps)


def _prepare_datasets(source, reference, config, paths, seed):
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
    return metadata, starts, masks, datasets


def _pilot_report(
    source, before, metadata, starts, masks, checks, results, config, seed, accelerator
):
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
            cuda_runtime=torch.version.cuda,
            torch_threads=torch.get_num_threads(),
            lightning=version("lightning"),
            numpy=np.__version__,
            mne=version("mne"),
            scipy=version("scipy"),
            host=platform.node(),
            device=torch.cuda.get_device_name(0) if accelerator == "gpu" else "cpu",
            cpu_affinity=sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            allocation={
                name: os.environ.get(name)
                for name in (
                    "SLURM_JOB_ID",
                    "SLURM_CPUS_PER_TASK",
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
        ),
    )
    report["code_sha256"] = {
        str(path): _checksum(path)
        for path in (
            *sorted(Path(__file__).parent.glob("*.py")),
            Path(edf_record_loading.__file__),
            Path(edf_source.__file__),
            Path(signal_calibration.__file__),
            Path(training_windows.__file__),
            *(
                Path(__file__).resolve().parents[2] / "stable_pretraining" / name
                for name in ("module.py", "manager.py", "callbacks/teacher_student.py")
            ),
        )
    }
    return report


def run(
    source: Path,
    reference_root: Path,
    output: Path,
    config: Config,
    *,
    paths: tuple[str, ...] = ("edf_memmap",),
    accelerator: str = "cpu",
    seed: int = 0,
    benchmark: BenchmarkConfig | None = None,
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
        benchmark: Optional repeated worker comparison and timing diagnostics.

    Returns:
        Local provenance, input checks and matched per-step training results.

    Note:
        CPU/MNE preprocessing includes a CUDA-to-host copy for GPU assembly.
        Timings are a correctness diagnostic, not an end-to-end speedup claim.
    """
    _validate_options(paths, accelerator, benchmark, config.total_steps)
    output, reference_root = _external(output), _external(reference_root)
    reference = load_reference(reference_root)
    before = _checksum(source)
    metadata, starts, masks, datasets = _prepare_datasets(
        source, reference, config, paths, seed
    )
    output.mkdir(parents=True, exist_ok=False)
    checks = verify_inputs(
        datasets, torch.device("cuda" if accelerator == "gpu" else "cpu")
    )
    benchmark_report = None
    if benchmark is None:
        results = _train_paths(
            reference, datasets, paths, config, output, accelerator, seed
        )
    else:
        from .benchmark import measure

        benchmark_report = measure(
            reference,
            datasets,
            paths,
            config,
            benchmark,
            output,
            accelerator,
            seed,
            checks,
        )
        results = {}
    if _checksum(source) != before:
        raise ValueError("Source changed during the pilot")
    report = _pilot_report(
        source,
        before,
        metadata,
        starts,
        masks,
        checks,
        results,
        config,
        seed,
        accelerator,
    )
    if benchmark_report is not None:
        from .benchmark import plot

        report.pop("paths")
        report["benchmark"] = benchmark_report
        report["benchmark_config"] = asdict(benchmark)
        plot(report, output)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Run a short pilot with external inputs and local artifacts."""
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
    parser.add_argument("--trials", type=int)
    parser.add_argument("--workers", nargs="+", type=int)
    parser.add_argument("--startup-steps", type=int, default=2)
    parser.add_argument("--diagnostic", action="store_true")
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
        benchmark=BenchmarkConfig(
            trials=args.trials if args.trials is not None else 3,
            workers=tuple(args.workers if args.workers is not None else (0, 2)),
            startup_steps=args.startup_steps,
            diagnostic=args.diagnostic,
        )
        if args.trials is not None or args.workers is not None or args.diagnostic
        else None,
    )
    print(json.dumps(report["checks"], indent=2))


if __name__ == "__main__":
    main()
