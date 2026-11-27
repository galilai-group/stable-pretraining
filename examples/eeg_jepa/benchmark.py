"""Repeated pilot measurements, separate diagnostics and local learning plots."""

import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

from profiling.training_windows import configure_reader_threads

from .config import BenchmarkConfig, Config
from .train import fit, initialize_module, validate_runs


def cases(paths: tuple[str, ...], workers: tuple[int, ...]) -> list[tuple[str, int]]:
    """Expand CPU worker settings while keeping GPU assembly in the main process.

    Args:
        paths: Requested stable-datasets loading paths.
        workers: CPU worker settings.

    Returns:
        Ordered path/worker pairs for one trial.
    """
    return [
        (path, count)
        for path in paths
        for count in (workers if path == "edf_memmap" else (0,))
    ]


def trial_order(matrix: list[tuple[str, int]], trial: int) -> list[tuple[str, int]]:
    """Alternate forward and reverse case order across paired trials.

    Args:
        matrix: Base case order.
        trial: Zero-based trial number.

    Returns:
        A fresh ordered case list.
    """
    return list(matrix if trial % 2 == 0 else reversed(matrix))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


class History(pl.Callback):
    """Record completed-step times after EMA, deferring scalar copies until fit ends."""

    def __init__(self, device: torch.device, diagnostic: bool = False) -> None:
        self.device, self.diagnostic = device, diagnostic
        self.losses, self.ends, self.deliveries, self.phases = [], [], [], []

    def begin(self) -> None:
        """Start before DataLoader iterator creation and worker startup."""
        _sync(self.device)
        self.started = perf_counter()
        self.origin = None
        if self.device.type == "cuda":
            self.origin = torch.cuda.Event(enable_timing=True)
            self.origin.record()

    def delivery(self, waiting: float) -> None:
        """Retain loader wait only for the separate diagnostic pass."""
        if self.diagnostic:
            self.deliveries.append((waiting, perf_counter()))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if self.diagnostic:
            _sync(self.device)
            self.compute_started = perf_counter()
            waiting, delivered = self.deliveries[batch_idx]
            self.phase = dict(
                batch["_diagnostic"],
                loader_wait_seconds=waiting,
                delivery_to_compute_seconds=self.compute_started - delivered,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.losses.append(outputs["loss"].detach())
        if self.diagnostic:
            _sync(self.device)
            self.phase["model_step_seconds"] = perf_counter() - self.compute_started
            self.phases.append(self.phase)
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.ends.append(event)
        else:
            self.ends.append(perf_counter() - self.started)

    def on_train_end(self, trainer, pl_module) -> None:
        _sync(self.device)
        if self.device.type == "cuda":
            self.ends = [self.origin.elapsed_time(event) / 1000 for event in self.ends]
        self.values = torch.stack(self.losses).cpu().tolist()
        if not np.isfinite(self.values).all():
            raise ValueError("Nonfinite benchmark loss")


class TimedLoader(DataLoader):
    """Start the main clock before worker creation; instrument waits only in diagnostics."""

    def __init__(self, *args, history: History, **kwargs) -> None:
        self.history = history
        super().__init__(*args, **kwargs)

    def __iter__(self):
        self.history.begin()
        iterator = super().__iter__()
        self.history.iterator_seconds = perf_counter() - self.history.started
        try:
            for _ in range(len(self)):
                started = perf_counter() if self.history.diagnostic else None
                batch = next(iterator)
                if self.history.diagnostic:
                    self.history.delivery(perf_counter() - started)
                yield batch
        finally:
            del iterator


def _loader(dataset, workers: int, seed: int, history: History | None = None):
    if dataset.reader.loading_path == "edf_gpu_assembly" and workers:
        raise ValueError("GPU assembly cannot run in workers")
    kwargs = dict(
        dataset=dataset,
        batch_size=None,
        num_workers=workers,
        multiprocessing_context="spawn" if workers else None,
        worker_init_fn=configure_reader_threads,
        prefetch_factor=2 if workers else None,
        timeout=120 if workers else 0,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=history is not None and history.device.type == "cuda",
    )
    return TimedLoader(history=history, **kwargs) if history else DataLoader(**kwargs)


def _verify_workers(dataset, workers, seed, expected_hashes, device):
    actual_hashes = []
    for index, batch in enumerate(_loader(dataset, workers, seed)):
        values = (batch["eeg"].to(device) / 100.0).cpu()
        digest = hashlib.sha256(values.numpy().tobytes()).hexdigest()
        if digest != expected_hashes[index]:
            raise ValueError(
                "Spawned worker model input differs from the validated baseline"
            )
        for key, expected in dataset.masks[index].items():
            torch.testing.assert_close(batch[key], expected, rtol=0, atol=0)
        actual_hashes.append(digest)
    if len(actual_hashes) != len(expected_hashes):
        raise ValueError("Incomplete worker input verification")
    return actual_hashes


def intervals(times: list[float], startup_steps: int, batch_size: int) -> dict:
    """Separate the predeclared first updates from sustained training.

    Args:
        times: Cumulative completed-step wall times, including loader startup.
        startup_steps: Number of updates in the startup interval.
        batch_size: Windows processed by each update.

    Returns:
        First-batch, startup, total and sustained interval measurements.
    """
    if not 1 <= startup_steps < len(times) or not np.isfinite(times).all():
        raise ValueError("Require finite times and a nonempty sustained interval")
    if times[0] <= 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Completed-step times must be positive and increasing")
    startup = times[startup_steps - 1]
    sustained = times[-1] - startup
    return dict(
        first_batch_seconds=times[0],
        startup_steps=startup_steps,
        startup_seconds=startup,
        total_seconds=times[-1],
        sustained_steps=len(times) - startup_steps,
        sustained_seconds=sustained,
        sustained_windows_per_second=(len(times) - startup_steps)
        * batch_size
        / sustained,
    )


def _run_one(
    reference, dataset, case, config, options, output, accelerator, seed, diagnostic
):
    module, initial_hash = initialize_module(reference, config, seed)
    dataset = copy.copy(dataset)
    dataset.diagnostic = diagnostic
    history = History(
        torch.device("cuda" if accelerator == "gpu" else "cpu"), diagnostic
    )
    loader = _loader(dataset, case[1], seed, history)
    fit(
        module,
        loader,
        output,
        config,
        accelerator,
        seed=seed,
        benchmark=True,
        after_callbacks=[history],
    )
    if int(module.eeg.completed_steps) != config.total_steps:
        raise ValueError("Teacher update count differs from the training budget")
    return dict(
        path=case[0],
        workers=case[1],
        case=f"{case[0]}/workers={case[1]}",
        initial_weights_sha256=initial_hash,
        parameters=dict(
            total=sum(p.numel() for p in module.parameters()),
            trainable=sum(p.numel() for p in module.parameters() if p.requires_grad),
        ),
        teacher_updates=int(module.eeg.completed_steps),
        iterator_setup_seconds=history.iterator_seconds,
        steps=[
            dict(loss=loss, elapsed_seconds=elapsed)
            for loss, elapsed in zip(history.values, history.ends, strict=True)
        ],
        intervals=intervals(history.ends, options.startup_steps, config.batch_size),
        diagnostic_phases=history.phases if diagnostic else None,
    )


def _summary(runs):
    summaries = {}
    for case in dict.fromkeys(run["case"] for run in runs):
        selected = [run for run in runs if run["case"] == case]
        summaries[case] = {}
        for metric in (
            "first_batch_seconds",
            "startup_seconds",
            "total_seconds",
            "sustained_seconds",
            "sustained_windows_per_second",
        ):
            values = [run["intervals"][metric] for run in selected]
            summaries[case][metric] = dict(
                median=float(np.median(values)), min=min(values), max=max(values)
            )
    return summaries


def measure(
    reference: Any,
    datasets: dict,
    paths: tuple[str, ...],
    config: Config,
    options: BenchmarkConfig,
    output: Path,
    accelerator: str,
    seed: int,
    checks: dict,
) -> dict:
    """Run paired alternating trials and optional separately instrumented replays.

    Args:
        reference: Verified EEG-JEPA modules.
        datasets: Shared validated inputs and predetermined masks.
        paths: Loading paths to compare.
        config: Matched model and training budget.
        options: Trial count, workers, startup exclusion and diagnostic choice.
        output: External directory for all artifacts.
        accelerator: Shared model endpoint.
        seed: Shared initialization and training seed.
        checks: Parent-process input checks used to validate spawned workers.

    Returns:
        All measurements, observed ranges and separate diagnostics.
    """
    options.validate(config.total_steps)
    matrix = cases(paths, options.workers)
    device = torch.device("cuda" if accelerator == "gpu" else "cpu")
    worker_checks = {}
    if "edf_memmap" in paths:
        for workers in options.workers:
            if workers:
                worker_checks[str(workers)] = _verify_workers(
                    datasets["edf_memmap"],
                    workers,
                    seed,
                    checks["cpu_model_input_sha256"],
                    device,
                )
    runs, diagnostics, validated = [], [], {}
    orders = []
    for trial in range(options.trials):
        order = trial_order(matrix, trial)
        orders.append([f"{path}/workers={workers}" for path, workers in order])
        for case in order:
            run = _run_one(
                reference,
                datasets[case[0]],
                case,
                config,
                options,
                output / f"trial-{trial}" / case[0] / f"workers-{case[1]}",
                accelerator,
                seed,
                False,
            )
            run.update(mode="main", trial=trial)
            validated[f"main/{trial}/{run['case']}"] = run
            validate_runs(validated, config.total_steps)
            runs.append(run)
            with (output / "trials.jsonl").open("a") as stream:
                stream.write(json.dumps(run) + "\n")
    if options.diagnostic:
        for case in matrix:
            run = _run_one(
                reference,
                datasets[case[0]],
                case,
                config,
                options,
                output / "diagnostic" / case[0] / f"workers-{case[1]}",
                accelerator,
                seed,
                True,
            )
            run.update(mode="diagnostic", trial=None)
            validated[f"diagnostic/{run['case']}"] = run
            validate_runs(validated, config.total_steps)
            diagnostics.append(run)
            with (output / "trials.jsonl").open("a") as stream:
                stream.write(json.dumps(run) + "\n")
    return dict(
        trial_order=orders,
        trials=runs,
        diagnostics=diagnostics,
        summary=_summary(runs),
        worker_input_sha256=worker_checks,
        parameters=runs[0]["parameters"],
        timing="Start before DataLoader iterator/worker creation; end after final optimizer and EMA. Main CUDA event timestamps resolved after fit; no added per-step synchronization or host loss copies. Model/Manager setup, preflight, shutdown, checkpoint I/O and plots excluded. Startup is the first configured updates; sustained uses all remaining updates.",
        diagnostic_timing="Separate replays synchronize phase boundaries. Reader/preprocessing times are per-item worker work, which can overlap model compute. Delivery-to-compute includes transfer and framework overhead; model-step includes forward/backward, optimizer and EMA.",
        condition="Warm-source/preflight checked, CPU/MNE preprocessing shared; no cold-cache or general GPU speedup claim",
    )


def plot(report: dict, output: Path) -> None:
    """Write local loss-versus-step and completed-wall-time panels.

    Args:
        report: Completed pilot report containing benchmark trials and provenance.
        output: External artifact directory.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = report["benchmark"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, case in enumerate(data["summary"]):
        color = plt.get_cmap("tab10")(index % 10)
        runs = [run for run in data["trials"] if run["case"] == case]
        for run in runs:
            rows = run["steps"]
            axes[0].plot(
                np.arange(1, len(rows) + 1),
                [r["loss"] for r in rows],
                alpha=0.25,
                color=color,
            )
            axes[1].plot(
                [r["elapsed_seconds"] for r in rows],
                [r["loss"] for r in rows],
                alpha=0.25,
                color=color,
            )
        losses = np.median(
            [[row["loss"] for row in run["steps"]] for run in runs], axis=0
        )
        times = np.median(
            [[row["elapsed_seconds"] for row in run["steps"]] for run in runs], axis=0
        )
        axes[0].plot(np.arange(1, len(losses) + 1), losses, label=case, color=color)
        axes[1].plot(times, losses, label=case, color=color)
    axes[0].set(xlabel="Optimizer step", ylabel="Reference loss")
    axes[1].set(
        xlabel="Completed training wall time (seconds)", ylabel="Reference loss"
    )
    axes[1].legend(fontsize="small")
    figure.suptitle(
        f"EEG-JEPA: {data['parameters']['trainable']:,} trainable / "
        f"{data['parameters']['total']:,} total parameters"
    )
    figure.text(
        0.5,
        0.01,
        f"{len(data['trial_order'])} paired trials | "
        f"EDF {report['source_sha256'][:12]} | reference {report['reference_revision'][:12]} | "
        "shared CPU preprocessing",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(output / "learning-progress.png", dpi=160)
    plt.close(figure)
