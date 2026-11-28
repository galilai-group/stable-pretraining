"""Shared Manager wiring and the synthetic EEG-JEPA entry point."""

import argparse
import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

import stable_pretraining as spt

from .config import Config
from .data import Collate, synthetic_samples
from .model import build_module
from .reference import load_reference


@contextmanager
def run_settings(output: Path, benchmark: bool = False) -> Iterator[None]:
    """Temporarily configure local logging and callbacks for an example run.

    Args:
        output: External directory for Manager artifacts.
        benchmark: Disable requeue checkpoint I/O inside timing intervals.

    Yields:
        Control with the temporary settings active.
    """
    settings = spt.get_config()
    previous_requeue = settings.requeue_checkpoint
    previous_cache = settings.cache_dir
    previous_callbacks = settings.default_callbacks
    previous_loggers = settings.default_loggers
    try:
        if benchmark:
            spt.set(requeue_checkpoint=False)
        spt.set(
            cache_dir=str(output),
            default_loggers={"registry": True},
            default_callbacks={
                key: False
                for key in (
                    "progress_bar",
                    "logging",
                    "env_dump",
                    "trainer_info",
                    "sklearn_checkpoint",
                    "wandb_checkpoint",
                    "trackio_checkpoint",
                    "swanlab_checkpoint",
                    "module_summary",
                    "slurm_info",
                    "unused_params",
                    "hf_checkpoint",
                )
            },
        )
        yield
    finally:
        # Public setters merge dictionaries, so they cannot remove temporary keys.
        settings._default_callbacks = previous_callbacks
        settings._default_loggers = previous_loggers
        settings.cache_dir = previous_cache
        settings.requeue_checkpoint = previous_requeue


def initialize_module(
    reference: Any, config: Config, seed: int
) -> tuple[spt.Module, str]:
    """Initialize a matched model and fingerprint its complete initial state.

    Args:
        reference: Verified reference model and objective modules.
        config: Model dimensions and training schedule.
        seed: Shared model initialization seed.

    Returns:
        Fresh Module and its initial state SHA256.
    """
    pl.seed_everything(seed, workers=True)
    module = build_module(reference, config)
    digest = hashlib.sha256()
    for value in module.state_dict().values():
        digest.update(value.detach().cpu().numpy().tobytes())
    return module, digest.hexdigest()


def validate_runs(results: dict[str, Any], steps: int) -> None:
    """Validate matched initialization and complete, equivalent loss traces.

    Args:
        results: Pilot or benchmark runs, updated with maximum loss differences.
        steps: Required number of optimizer updates in every trace.
    """
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
    reference_root: Path,
    output: Path,
    config: Config,
    accelerator: str = "cpu",
    callbacks: list[Any] | None = None,
) -> spt.Module:
    """Run a small synthetic fit, keeping all generated artifacts outside Git.

    Args:
        reference_root: External extracted archive at the pinned revision.
        output: External local directory for Manager artifacts.
        config: Synthetic model and optimizer schedule.
        accelerator: Lightning accelerator, cpu or gpu.
        callbacks: Optional observers, registered before the EMA callback.

    Returns:
        Trained Module for numerical inspection.
    """
    output = output.expanduser().resolve()
    for candidate in (output, reference_root.expanduser().resolve()):
        if any(
            (parent / ".git").exists() for parent in (candidate, *candidate.parents)
        ):
            raise ValueError(
                "Reference caches and experiment artifacts must be outside Git"
            )
    reference = load_reference(reference_root)
    pl.seed_everything(0, workers=True)
    module = build_module(reference, config)
    loader = DataLoader(
        synthetic_samples(config),
        batch_size=config.batch_size,
        collate_fn=Collate(
            reference.masks.NMETTargetSupportCollator(**config.mask_kwargs())
        ),
        num_workers=0,
    )
    fit(module, loader, output, config, accelerator, callbacks)
    return module


def fit(
    module: spt.Module,
    loader: DataLoader,
    output: Path,
    config: Config,
    accelerator: str = "cpu",
    callbacks: list[Any] | None = None,
    seed: int = 0,
    benchmark: bool = False,
    after_callbacks: list[Any] | None = None,
) -> None:
    """Fit an example Module using the shared Manager and callback wiring.

    Args:
        module: Freshly initialized adapted model.
        loader: Synthetic or real pre-batched input loader.
        output: External directory for local run artifacts.
        config: Optimizer schedule and training budget.
        accelerator: Lightning accelerator, cpu or gpu.
        callbacks: Optional observers registered before the EMA callback.
        seed: Shared training RNG seed.
        benchmark: Disable automatic requeue checkpoints during measurement.
        after_callbacks: Observers that must see completed teacher updates.
    """
    with run_settings(output, benchmark=benchmark):
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            precision="32-true",
            max_epochs=1,
            max_steps=config.total_steps,
            log_every_n_steps=1,
            logger=False,
            enable_checkpointing=False,
            gradient_clip_val=config.clip_value,
            gradient_clip_algorithm="norm",
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            deterministic=True,
            default_root_dir=str(output),
            callbacks=[
                *(callbacks or []),
                spt.TeacherStudentCallback(verbose=False),
                *(after_callbacks or []),
            ],
        )
        spt.Manager(
            trainer=trainer,
            module=module,
            data=spt.data.DataModule(train=loader),
            seed=seed,
        )()


def main() -> None:
    """Run synthetic reference training with local artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.reference_root, args.output, Config(), args.accelerator)


if __name__ == "__main__":
    main()
