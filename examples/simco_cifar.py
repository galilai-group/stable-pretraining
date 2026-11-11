"""SimCo on CIFAR-10 and CIFAR-100
================================

Train SimCo through Module, OnlineProbe, and Manager.

This follows SimCo's two-view dual-temperature objective (CVPR 2022).
The reserved validation split and FP32 recipe are adaptations of author settings.
Run ``python examples/simco_cifar.py --help`` for all options.
"""  # noqa: D205, D415 -- Sphinx-Gallery requires a section title.

import argparse
import hashlib
import json
import time
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassAccuracy
from torchvision.datasets import CIFAR10, CIFAR100, FakeData

import stable_pretraining as spt


_AUGMENTATION = {
    "random_resized_crop": {
        "size": [32, 32],
        "scale": [0.2, 1.0],
        "ratio": [0.75, 4.0 / 3.0],
        "antialias": True,
    },
    "random_horizontal_flip": {"p": 0.5},
    "color_jitter": {
        "brightness": 0.4,
        "contrast": 0.4,
        "saturation": 0.2,
        "hue": 0.1,
        "p": 0.8,
    },
    "random_grayscale": {"p": 0.2},
}

_METRIC_DEFINITIONS = {
    "fit/loss": {
        "definition": "Symmetric dual-temperature cross-view loss before adding detached probe classifier loss",
        "direction": "minimize",
    },
    "train/linear_probe_loss": {
        "definition": "Mean classifier cross-entropy on detached augmented backbone features",
        "direction": "minimize",
    },
    "eval/linear_probe_top1_accuracy": {
        "definition": "Fraction of correct online classifier predictions on the reserved validation images, in [0, 1]",
        "direction": "maximize",
    },
}


def _fingerprint(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Wiring-check step budget; also caps batches per epoch so validation runs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Global batch size across all devices",
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--intra-temperature", type=float, default=0.1)
    parser.add_argument("--inter-temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--probe-learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--precision", choices=("32-true", "bf16-mixed"), default="32-true"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Tiny wiring check; no real-data accuracy claim",
    )
    return parser


def _datasets(args: argparse.Namespace):
    classes = 10 if args.dataset == "cifar10" else 100
    stats = (
        spt.data.dataset_stats.CIFAR10
        if classes == 10
        else spt.data.dataset_stats.CIFAR100
    )
    transforms = spt.data.transforms
    augment = transforms.Compose(
        transforms.RandomResizedCrop(**_AUGMENTATION["random_resized_crop"]),
        transforms.RandomHorizontalFlip(**_AUGMENTATION["random_horizontal_flip"]),
        transforms.ColorJitter(**_AUGMENTATION["color_jitter"]),
        transforms.RandomGrayscale(**_AUGMENTATION["random_grayscale"]),
        transforms.ToImage(dtype=torch.float32, scale=True, **stats),
    )
    if args.synthetic:
        train = FakeData(size=64, image_size=(3, 32, 32), num_classes=classes)
        val = FakeData(
            size=32, image_size=(3, 32, 32), num_classes=classes, random_offset=10000
        )
        split_indices = {
            "source": "separate torchvision FakeData datasets",
            "training_random_offset": 0,
            "validation_random_offset": 10000,
            "training_indices": list(range(len(train))),
            "validation_indices": list(range(len(val))),
        }
    else:
        dataset_class = CIFAR10 if classes == 10 else CIFAR100
        original_train = dataset_class(
            root=str(args.data_dir), train=True, download=False
        )
        indices = torch.randperm(
            len(original_train),
            generator=torch.Generator().manual_seed(args.split_seed),
        ).tolist()
        # Reserve validation examples before SSL; the official test split stays untouched.
        train = Subset(original_train, indices[5000:])
        val = Subset(original_train, indices[:5000])
        split_indices = {
            "source": args.dataset,
            "source_split": "official training split",
            "training_indices": train.indices,
            "validation_indices": val.indices,
        }
    train = spt.data.FromTorchDataset(
        train,
        names=["image", "label"],
        transform=transforms.MultiViewTransform([augment, augment]),
    )
    val = spt.data.FromTorchDataset(
        val,
        names=["image", "label"],
        transform=transforms.Compose(
            transforms.ToImage(dtype=torch.float32, scale=True, **stats)
        ),
    )
    return train, val, classes, split_indices


def _protocol(args: argparse.Namespace, classes: int, split_indices: dict) -> dict:
    if args.devices < 1 or args.batch_size < 2 or args.batch_size % args.devices:
        raise ValueError(
            "Require devices >= 1 and global batch-size >= 2 divisible by devices"
        )
    for split in ("training_indices", "validation_indices"):
        if len(split_indices[split]) % args.devices:
            raise ValueError(
                f"{split} size must be divisible by devices to avoid sampler padding"
            )
    stats = (
        spt.data.dataset_stats.CIFAR10
        if classes == 10
        else spt.data.dataset_stats.CIFAR100
    )
    batches_per_epoch = len(split_indices["training_indices"]) // args.batch_size
    if args.max_steps > 0:
        batches_per_epoch = min(batches_per_epoch, args.max_steps)
    planned_optimizer_steps = args.epochs * batches_per_epoch
    if args.max_steps > 0:
        planned_optimizer_steps = min(planned_optimizer_steps, args.max_steps)
    warmup_steps = args.warmup_epochs * batches_per_epoch
    if args.warmup_epochs < 0 or warmup_steps >= planned_optimizer_steps:
        raise ValueError(
            "Warmup must be shorter than training; use --warmup-epochs 0 for wiring checks"
        )
    return {
        "method": "SimCo",
        "paper": "https://arxiv.org/abs/2203.17248",
        "author_reference_commit": "e8d5e768255721b23643d948631f80c0b8a0851b",
        "recipe": "cifar_adaptation_not_paper_reproduction",
        "data": {
            "dataset": args.dataset,
            "synthetic": args.synthetic,
            "number_of_classes": classes,
            "training_examples": len(split_indices["training_indices"]),
            "validation_examples": len(split_indices["validation_indices"]),
            "split_seed": args.split_seed,
            "split_indices_file": "split_indices.json",
            "split_indices_sha256": _fingerprint(split_indices),
            "official_test_split_used": False,
        },
        "augmentation": {
            "number_of_independent_views": 2,
            "transform_order": [*_AUGMENTATION, "normalization"],
            "ordered_transforms": _AUGMENTATION,
            "resize_interpolation": "bilinear",
        },
        "normalization": {**stats, "dtype": "float32", "scale": True},
        "encoder": {"model_name": "resnet18", "low_resolution": True, "weights": None},
        "projector": {
            "dimensions": [512, args.hidden_dim, args.projection_dim],
            "hidden_blocks": "Linear with bias, ReLU",
            "final_linear_bias": True,
            "batch_normalization": False,
        },
        "loss": {
            "intra_temperature": args.intra_temperature,
            "inter_temperature": args.inter_temperature,
            "positive_pairs": "matching row across views",
            "negatives": "other rows of opposite view only",
            "reduction": "mean over global examples and both directions",
            "hardness_ratio_detached": True,
        },
        "optimization": {
            "optimizer": {
                "type": "SGD",
                "lr": args.learning_rate,
                "momentum": 0.9,
                "weight_decay": 5e-4,
                "exclude_bias_norm": False,
            },
            # Module advances the scheduler after each optimizer update.
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": warmup_steps,
                "max_steps": planned_optimizer_steps,
                "warmup_start_lr": 0.0,
                "eta_min": 0.0,
            },
            "interval": "step",
        },
        "probe": {
            "input_features": 512,
            "number_of_classes": classes,
            "loss": "CrossEntropyLoss with mean reduction",
            "optimizer": {
                "type": "SGD",
                "lr": args.probe_learning_rate,
                "momentum": 0.9,
            },
            "scheduler": {"type": "ConstantLR", "factor": 1.0},
            "features_detached": True,
            "accuracy_average": "micro",
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "maximum_optimizer_steps": args.max_steps,
            "planned_optimizer_steps": planned_optimizer_steps,
            "batches_per_epoch": batches_per_epoch,
            "positive_step_budget_caps_epoch_batches": True,
            "batch_size": args.batch_size,
            "global_batch_size": args.batch_size,
            "per_device_batch_size": args.batch_size // args.devices,
            "workers": args.workers,
            "training_shuffle": True,
            "training_drop_last": True,
            "validation_shuffle": False,
            "accelerator": args.accelerator,
            "devices": args.devices,
            "world_size": args.devices,
            "strategy": "ddp" if args.devices > 1 else "auto",
            "sync_batchnorm": args.devices > 1 and args.accelerator == "gpu",
            "distributed_sampler": args.devices > 1,
            "validation_sampler_padding": False,
            "precision": args.precision,
            "torch_cpu_threads": 1,
        },
        "validation_records": {
            "directory": "validation_records",
            "frequency": "every validation epoch",
            "names": ["sample_idx", "label", "linear_probe_preds"],
            "sample_idx": "Position within split_indices.json validation_indices; that list gives the source dataset index",
            "label": "Ground-truth integer class index",
            "linear_probe_preds": "Unnormalized classifier logits; argmax over classes yields the predicted label",
            "distributed_files": "Each rank writes its own device=LOCAL_RANK shards; concatenate all ranks for an epoch",
        },
        "environment_and_resources": "Built-in EnvironmentDumpCallback, HardwareMonitor and RegistryLogger artifacts in the output directory and Manager run directory",
    }


def _checkpoints(output_dir: Path) -> tuple[ModelCheckpoint, ModelCheckpoint]:
    best = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="epoch-{epoch:03d}",
        auto_insert_metric_name=False,
        monitor="eval/linear_probe_top1_accuracy",
        mode="max",
        save_top_k=1,
        save_last=False,
        save_on_train_epoch_end=False,
    )
    # A monitored callback's save_last can stay stale when its top-k is unchanged.
    latest = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="latest",
        monitor=None,
        save_top_k=1,
        save_last=False,
        save_on_train_epoch_end=True,
        enable_version_counter=False,
    )
    return best, latest


def _run(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("epochs must be positive; max-steps must be positive or -1")
    if (
        args.batch_size < 2
        or args.workers < 0
        or args.hidden_dim < 1
        or args.projection_dim < 2
    ):
        raise ValueError(
            "Require batch-size >= 2, workers >= 0, hidden-dim > 0, projection-dim > 1"
        )
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run inside a GPU allocation, or pass --accelerator cpu"
        )
    torch.set_num_threads(1)
    pl.seed_everything(args.seed, workers=True)
    output_dir = (
        (args.output_dir or Path("outputs") / f"simco-{args.dataset}")
        .expanduser()
        .resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    spt.set(cache_dir=str(output_dir), requeue_checkpoint=False)
    train, val, classes, split_indices = _datasets(args)
    if len(train) < args.batch_size:
        raise ValueError(
            "batch-size exceeds training set size; use --batch-size 16 for synthetic checks"
        )
    protocol = _protocol(args, classes, split_indices)
    loaders = {
        "batch_size": protocol["training"]["per_device_batch_size"],
        "num_workers": args.workers,
        "persistent_workers": args.workers > 0,
        "pin_memory": args.accelerator == "gpu",
    }
    data = spt.data.DataModule(
        train=DataLoader(train, shuffle=True, drop_last=True, **loaders),
        val=DataLoader(val, shuffle=False, **loaders),
    )
    backbone = spt.backbone.from_torchvision(**protocol["encoder"])
    input_dim, hidden_dim, projection_dim = protocol["projector"]["dimensions"]
    projector = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, projection_dim),
    )
    warnings = [
        "This CIFAR recipe is an untuned adaptation, not a reproduction of paper accuracy.",
        "Online probe accuracy is diagnostic; the official test split and frozen-encoder evaluation remain unused.",
    ]
    if args.synthetic:
        warnings.append(
            "Synthetic images check wiring only; accuracy has no real-data meaning."
        )
    if args.max_steps > 0:
        warnings.append(
            "The positive step budget shortens epochs; this is a wiring check, not a full training result."
        )
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    configuration.update(
        output_dir=str(output_dir),
        recipe="cifar_adaptation_not_paper_reproduction",
        train_examples=len(train),
        validation_examples=len(val),
        evaluation="online linear classifier on detached backbone features; accuracy higher is better",
        official_test_split_used=False,
        protocol=protocol,
        protocol_sha256=_fingerprint(protocol),
        split_indices_sha256=protocol["data"]["split_indices_sha256"],
        metric_definitions=_METRIC_DEFINITIONS,
        warnings=warnings,
    )
    module = spt.SimCo(
        backbone=backbone,
        projector=projector,
        intra_temperature=protocol["loss"]["intra_temperature"],
        inter_temperature=protocol["loss"]["inter_temperature"],
        hparams=configuration,
        optim=protocol["optimization"],
    )
    probe = spt.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=nn.Linear(
            protocol["probe"]["input_features"], protocol["probe"]["number_of_classes"]
        ),
        loss=nn.CrossEntropyLoss(),
        optimizer=protocol["probe"]["optimizer"],
        scheduler=protocol["probe"]["scheduler"],
        metrics={"top1_accuracy": MulticlassAccuracy(classes, average="micro")},
        verbose=False,
    )
    checkpoints, latest_checkpoint = _checkpoints(output_dir)
    writer = spt.OnlineWriter(
        names=protocol["validation_records"]["names"],
        path=output_dir / protocol["validation_records"]["directory"],
        during="validation",
        every_k_epochs=-1,
        all_gather=False,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=1,
        strategy=protocol["training"]["strategy"],
        sync_batchnorm=protocol["training"]["sync_batchnorm"],
        use_distributed_sampler=True,
        precision=args.precision,
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        limit_train_batches=protocol["training"]["batches_per_epoch"]
        if args.max_steps > 0
        else 1.0,
        default_root_dir=output_dir,
        logger=False,
        callbacks=[probe, checkpoints, latest_checkpoint, writer],
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data,
        seed=args.seed,
        ckpt_path=str(args.resume.expanduser().resolve()) if args.resume else None,
        weights_only=False,
    )
    if spt.utils.get_rank() == 0:
        with (output_dir / "config.json").open("w", encoding="utf-8") as stream:
            json.dump(configuration, stream, indent=2)
        with (output_dir / "split_indices.json").open("w", encoding="utf-8") as stream:
            json.dump(
                split_indices,
                stream,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        for warning in warnings:
            print(f"WARNING: {warning}")
    started = time.perf_counter()
    manager()
    if not trainer.is_global_zero:
        return
    if trainer.world_size != args.devices:
        raise RuntimeError(
            f"Expected {args.devices} training processes, got {trainer.world_size}"
        )
    summary = {
        "method": "SimCo",
        "dataset": args.dataset,
        "synthetic_wiring_check": args.synthetic,
        "optimizer_steps": trainer.global_step,
        "wall_time_seconds": time.perf_counter() - started,
        "distributed": {
            "world_size": trainer.world_size,
            "strategy": type(trainer.strategy).__name__,
            "global_batch_size": args.batch_size,
            "per_device_batch_size": protocol["training"]["per_device_batch_size"],
            "sync_batchnorm": protocol["training"]["sync_batchnorm"],
        },
        "best_checkpoint": checkpoints.best_model_path,
        "best_validation_online_top1_accuracy": checkpoints.best_model_score.item()
        if checkpoints.best_model_score is not None
        else None,
        "last_checkpoint": latest_checkpoint.best_model_path,
        "protocol_sha256": configuration["protocol_sha256"],
        "split_indices_sha256": configuration["split_indices_sha256"],
        "metric_definitions": _METRIC_DEFINITIONS,
        "warnings": warnings,
        "validation_records": protocol["validation_records"],
        "metrics": {
            name: value.item()
            for name, value in trainer.callback_metrics.items()
            if value.numel() == 1
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


def main() -> None:
    """Prepare data or run the CIFAR SimCo training example."""
    args = _parser().parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    if args.download_only:
        dataset_class = CIFAR10 if args.dataset == "cifar10" else CIFAR100
        for train in (True, False):
            dataset_class(root=str(args.data_dir), train=train, download=True)
        print(f"Prepared {args.dataset} at {args.data_dir}")
        return
    _run(args)


if __name__ == "__main__":
    main()
