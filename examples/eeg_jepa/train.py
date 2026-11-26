"""Shared Manager wiring and the synthetic EEG-JEPA entry point."""

import argparse
from pathlib import Path
from typing import Any

import lightning as pl
import torch
from torch.utils.data import DataLoader

import stable_pretraining as spt

from .config import Config, run_settings
from .data import Collate, synthetic_samples
from .model import build_module
from .reference import load_reference


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
    """
    with run_settings(output):
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
            callbacks=[*(callbacks or []), spt.TeacherStudentCallback(verbose=False)],
        )
        spt.Manager(
            trainer=trainer,
            module=module,
            data=spt.data.DataModule(train=loader),
            seed=seed,
        )()


def main() -> None:
    """Run the milestone locally without downloads, uploads or dataset reads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.reference_root, args.output, Config(), args.accelerator)


if __name__ == "__main__":
    main()
