"""ADNCE on CIFAR
==============

Run ``python examples/adnce_cifar10.py --help`` for options. This development
benchmark holds out part of the *training* set and never reads the official
CIFAR test set. CIFAR-10 is the default; select CIFAR-100 with
``--dataset cifar100``. See ``docs/source/guides/adnce.md`` for the evaluation
protocol and differences from the authors' experiment.
"""  # noqa: D205, D415

import argparse
import json
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Any

import lightning as pl
import numpy as np
import torch
import torchmetrics
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100

import stable_pretraining as spt


REFERENCE_COMMIT = "cdd110f85402641d52762ba63aa2a17ba907a6f9"
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}
# The authors' image/utils.py shares these normalization values across CIFAR datasets.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse training and evaluation settings.

    Args:
        argv: Optional arguments, excluding the executable name.

    Returns:
        Validated command-line settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("adnce", "simclr"), default="adnce")
    parser.add_argument("--dataset", choices=tuple(NUM_CLASSES), default="cifar10")
    parser.add_argument(
        "--backbone", choices=("resnet50", "resnet18"), default="resnet50"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("~/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/adnce"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Images per process."
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--mu", type=float, default=0.7)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--accelerator", choices=("auto", "cpu", "gpu", "mps"), default="auto"
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Optional training cap for debugging.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="Optional step limit for debugging."
    )
    parser.add_argument("--probe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--probe-samples",
        type=int,
        default=None,
        help="Optional cap on frozen-probe training images.",
    )
    parser.add_argument("--probe-alpha", type=float, default=1.0)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Trusted local checkpoint; restores full training state.",
    )
    args = parser.parse_args(argv)
    num_classes = NUM_CLASSES[args.dataset]
    if args.batch_size < 2 or args.epochs < 1 or args.devices < 1:
        parser.error(
            "batch-size must be at least 2; epochs and devices must be positive"
        )
    max_val_size = 50000 - 2 * num_classes
    if not num_classes <= args.val_size <= max_val_size:
        parser.error(f"val-size must be between {num_classes} and {max_val_size}")
    for name in ("train_samples", "probe_samples"):
        value = getattr(args, name)
        if value is not None and value < num_classes:
            parser.error(f"{name.replace('_', '-')} must be at least {num_classes}")
    if args.num_workers < 0 or args.probe_alpha <= 0:
        parser.error("num-workers must be nonnegative and probe-alpha must be positive")
    for name in ("data_dir", "output_dir", "resume"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def build_loaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build disjoint SSL-training, frozen-probe-training, and validation loaders.

    Args:
        args: Settings returned by :func:`parse_args`.

    Returns:
        Two-view SSL loader and two deterministic single-view loaders. Both
        training loaders use the same images; validation images are excluded.
    """
    transforms = spt.data.transforms
    augment = transforms.Compose(
        transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=0.5),
        transforms.ToImage(mean=MEAN, std=STD),
    )
    dataset_type = CIFAR10 if args.dataset == "cifar10" else CIFAR100
    dataset = dataset_type(root=args.data_dir, train=True, download=args.download)
    train_ids, val_ids = train_test_split(
        np.arange(len(dataset)),
        test_size=args.val_size,
        random_state=args.split_seed,
        stratify=dataset.targets,
    )
    if args.train_samples is not None:
        train_ids = train_ids[: args.train_samples]
    if len(train_ids) < args.batch_size * args.devices:
        raise ValueError(
            "Training split must contain at least one full batch per device."
        )
    probe_ids = (
        train_ids if args.probe_samples is None else train_ids[: args.probe_samples]
    )
    settings = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.accelerator != "cpu" and torch.cuda.is_available(),
    }

    def loader(
        indices: np.ndarray, transform: Any, training: bool = False
    ) -> DataLoader:
        wrapped = spt.data.FromTorchDataset(
            Subset(dataset, indices.tolist()),
            names=["image", "label"],
            transform=transform,
        )
        return DataLoader(
            wrapped,
            shuffle=training,
            drop_last=training,
            generator=torch.Generator().manual_seed(args.seed),
            **settings,
        )

    return (
        loader(
            train_ids, transforms.MultiViewTransform([augment, augment]), training=True
        ),
        loader(probe_ids, transforms.ToImage(mean=MEAN, std=STD)),
        loader(val_ids, transforms.ToImage(mean=MEAN, std=STD)),
    )


def build_module(args: argparse.Namespace) -> spt.Module:
    """Build either objective with identical initialization and architecture.

    Args:
        args: Settings returned by :func:`parse_args`.

    Returns:
        A module ready for :class:`spt.Manager` and evaluation callbacks.
    """
    pl.seed_everything(args.seed, workers=True)
    backbone = spt.backbone.from_torchvision(
        args.backbone, low_resolution=True, weights=None
    )
    features = 2048 if args.backbone == "resnet50" else 512
    projector = nn.Sequential(
        nn.Linear(features, 512, bias=False),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 128),
    )
    optim = {
        "optimizer": partial(
            torch.optim.Adam, lr=args.lr, weight_decay=args.weight_decay
        ),
        "scheduler": {"type": "ConstantLR", "factor": 1.0},
    }
    if args.method == "adnce":
        return spt.ADNCE(
            backbone=backbone,
            projector=projector,
            optim=optim,
            temperature=args.temperature,
            mu=args.mu,
            sigma=args.sigma,
        )
    return spt.Module(
        forward=spt.forward.simclr,
        backbone=backbone,
        projector=projector,
        simclr_loss=spt.losses.NTXEntLoss(temperature=args.temperature),
        optim=optim,
    )


@torch.inference_mode()
def extract_features(
    backbone: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Extract normalized frozen features without changing encoder statistics.

    Args:
        backbone: Encoder to evaluate, already on ``device``.
        loader: Deterministic, single-view image and label batches.
        device: Encoder device.

    Returns:
        Feature matrix and labels on CPU, in loader order.
    """
    was_training = backbone.training
    parameter = next(backbone.parameters(), None)
    features, labels = [], []
    backbone.eval()
    try:
        for batch in loader:
            image = batch["image"]
            dtype = parameter.dtype if parameter is not None else image.dtype
            embedding = backbone(image.to(device=device, dtype=dtype))
            features.append(nn.functional.normalize(embedding.float(), dim=-1).cpu())
            labels.append(batch["label"].cpu())
    finally:
        backbone.train(was_training)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


class FrozenProbeReport(pl.Callback):
    """Compare initial and final frozen features with a fixed ridge classifier.

    Args:
        train: Single-view training images, disjoint from validation.
        val: Held-out validation images.
        args: Experiment and reporting settings.

    Note:
        This is a development diagnostic, not the authors' kNN evaluation.
        The initial score is checkpointed so resume does not relabel a trained
        encoder as its initial state. Evaluation runs only on the global rank 0.
        If Lightning skips training, a fallback reports after teardown on CPU.
    """

    def __init__(
        self, train: DataLoader, val: DataLoader, args: argparse.Namespace
    ) -> None:
        self.train_loader, self.val_loader, self.args = train, val, args
        self.initial: dict[str, float] | None = None
        self.started = 0.0
        self._reported = False

    def state_dict(self) -> dict[str, Any]:
        """Return the original encoder's score for checkpoint resume."""
        return {"initial": self.initial}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the original encoder's score from a trusted checkpoint."""
        self.initial = state_dict.get("initial")

    def _evaluate(self, module: spt.Module) -> dict[str, float]:
        train_x, train_y = extract_features(
            module.backbone, self.train_loader, module.device
        )
        val_x, val_y = extract_features(module.backbone, self.val_loader, module.device)
        probe = make_pipeline(
            StandardScaler(), RidgeClassifier(alpha=self.args.probe_alpha)
        )
        probe.fit(train_x, train_y)
        return {
            "validation_accuracy": float(probe.score(val_x, val_y)),
            "mean_feature_std": float(train_x.std(axis=0).mean()),
        }

    def on_fit_start(self, trainer: pl.Trainer, pl_module: spt.Module) -> None:
        """Evaluate an untrained encoder once, then begin training timing."""
        self._reported = False
        if trainer.is_global_zero and self.args.probe and self.initial is None:
            if trainer.ckpt_path is None:
                self.initial = self._evaluate(pl_module)
        self.initial = trainer.strategy.broadcast(self.initial)
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
            torch.cuda.reset_peak_memory_stats(pl_module.device)
        self.started = time.perf_counter()

    def on_train_end(self, trainer: pl.Trainer, pl_module: spt.Module) -> None:
        """Evaluate and report before strategy teardown moves the model to CPU."""
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
        elapsed = time.perf_counter() - self.started
        peak = (
            torch.cuda.max_memory_allocated(pl_module.device)
            if pl_module.device.type == "cuda"
            else None
        )
        if trainer.is_global_zero:
            final = self._evaluate(pl_module) if self.args.probe else None
            git_options = {
                "cwd": Path(__file__).resolve().parents[1],
                "check": False,
                "capture_output": True,
                "text": True,
            }
            try:
                source = subprocess.run(["git", "rev-parse", "HEAD"], **git_options)
                status = subprocess.run(["git", "status", "--porcelain"], **git_options)
                source_commit = (
                    source.stdout.strip() if source.returncode == 0 else None
                )
                source_dirty = (
                    bool(status.stdout.strip()) if status.returncode == 0 else None
                )
            except OSError:
                source_commit, source_dirty = None, None
            report = {
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(self.args).items()
                },
                "source_commit": source_commit,
                "source_dirty": source_dirty,
                "reference_commit": REFERENCE_COMMIT,
                "torch_version": torch.__version__,
                "lightning_version": pl.__version__,
                "world_size": trainer.world_size,
                "global_step": trainer.global_step,
                "training_seconds_this_invocation": elapsed,
                "rank_zero_peak_cuda_bytes": peak,
                "train_images": len(trainer.datamodule.train.dataset),
                "probe_train_images": len(self.train_loader.dataset),
                "validation_images": len(self.val_loader.dataset),
                "initial_frozen_ridge": self.initial,
                "final_frozen_ridge": final,
                "final_frozen_probe_device": (
                    str(pl_module.device) if self.args.probe else None
                ),
            }
            run_dir = next(
                logger.run_dir
                for logger in trainer.loggers
                if isinstance(logger, spt.RegistryLogger)
            )
            path = run_dir / "evaluation.json"
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"Frozen-probe results and settings: {path}\n{json.dumps(report, indent=2)}"
            )
        trainer.strategy.barrier()
        self._reported = True

    def on_fit_end(self, trainer: pl.Trainer, pl_module: spt.Module) -> None:
        """Keep reporting when a completed checkpoint or zero steps skips training."""
        if not self._reported:
            self.on_train_end(trainer, pl_module)


def main(argv: list[str] | None = None) -> None:
    """Train ADNCE or SimCLR and write a held-out evaluation report.

    Args:
        argv: Optional command-line arguments for programmatic invocation.
    """
    args = parse_args(argv)
    pl.seed_everything(args.seed, workers=True)
    train, probe_train, val = build_loaders(args)
    module = build_module(args)
    features = 2048 if args.backbone == "resnet50" else 512
    num_classes = NUM_CLASSES[args.dataset]
    probe = spt.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=nn.Linear(features, num_classes),
        loss=nn.CrossEntropyLoss(),
        metrics={
            "accuracy": torchmetrics.classification.MulticlassAccuracy(
                num_classes=num_classes, average="micro"
            )
        },
    )
    report = FrozenProbeReport(probe_train, val, args)
    spt.set(cache_dir=str(args.output_dir))
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=[probe, report],
        logger=False,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        default_root_dir=str(args.output_dir),
        log_every_n_steps=10,
        deterministic=True,
    )
    module.save_hyperparameters(
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train, val=val),
        seed=args.seed,
        ckpt_path=str(args.resume) if args.resume else None,
        weights_only=False,
    )
    manager()


if __name__ == "__main__":
    main()
