"""CorInfoMax CIFAR-10 recipe: pretraining, frozen linear fitting, evaluation.

Run each phase separately; checkpoint paths printed by training are inputs to
the next phase. Checkpoints must be trusted local Lightning checkpoints.
The test split is evaluated through validation because OnlineProbe reports
metrics in that stage. Best test accuracy is descriptive, not a tuning criterion.
See CorInfoMax.md for commands and the unverified reproduction status.
"""

import argparse
import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision.datasets import CIFAR10
from torchvision.transforms import InterpolationMode

import stable_pretraining as spt


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.247, 0.243, 0.261)


def _ssl_view(blur_probability: float, solarize_probability: float) -> Any:
    transforms = spt.data.transforms
    return transforms.Compose(
        transforms.RandomResizedCrop(32, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILGaussianBlur(sigma=(0.1, 2.0), p=blur_probability),
        transforms.RandomSolarize(threshold=128, p=solarize_probability),
        transforms.ToImage(mean=MEAN, std=STD),
    )


def _transform(mode: str, train: bool) -> Any:
    transforms = spt.data.transforms
    if mode == "pretrain":
        return transforms.MultiViewTransform([_ssl_view(1.0, 0.0), _ssl_view(0.1, 0.2)])
    if train:
        return transforms.Compose(
            transforms.RandomResizedCrop(32, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(mean=MEAN, std=STD),
        )
    return transforms.Compose(
        transforms.Resize(36, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(32),
        transforms.ToImage(mean=MEAN, std=STD),
    )


def _loader(args: argparse.Namespace, train: bool) -> DataLoader:
    dataset = spt.data.FromTorchDataset(
        CIFAR10(root=str(args.data_dir), train=train, download=args.download),
        names=["image", "label"],
        transform=_transform(args.mode, train),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        drop_last=train and args.mode == "pretrain",
        num_workers=args.workers,
        pin_memory=args.accelerator != "cpu",
        persistent_workers=args.workers > 0,
    )


def _epoch_lr_factor(
    step: int,
    *,
    steps_per_epoch: int,
    epochs: int,
    peak_lr: float,
    min_lr: float,
    warmup_epochs: int,
    warmup_start_lr: float,
) -> float:
    # Module steps schedulers per batch, so hold the reference LR for an epoch.
    epoch = min(step // steps_per_epoch, epochs)
    if epoch < warmup_epochs:
        lr = warmup_start_lr + epoch * (peak_lr - warmup_start_lr) / (warmup_epochs - 1)
    else:
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
    return lr / peak_lr


def _scheduler(recipe: dict[str, Any]) -> partial:
    factor = partial(
        _epoch_lr_factor,
        **{
            key: recipe[key]
            for key in (
                "steps_per_epoch",
                "epochs",
                "peak_lr",
                "min_lr",
                "warmup_epochs",
                "warmup_start_lr",
            )
        },
    )
    return partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=factor)


def _checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Expected a Lightning checkpoint: {path}")
    return checkpoint


def _load_component(
    component: torch.nn.Module, checkpoint: dict[str, Any], prefix: str
) -> None:
    state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    if not state:
        raise ValueError(f"Checkpoint has no weights with prefix {prefix!r}")
    component.load_state_dict(state, strict=True)


def _features(self: Any, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
    return {"embedding": self.backbone(batch["image"]), "label": batch["label"]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pretrain", "linear", "evaluate"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, help="Trusted input phase checkpoint"
    )
    parser.add_argument(
        "--resume", type=Path, help="Trusted checkpoint of the same phase"
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    pretrain = args.mode == "pretrain"
    if args.epochs is None:
        args.epochs = 1000 if pretrain else 100
    if args.batch_size is None:
        args.batch_size = 512 if pretrain else 256
    if args.warmup_epochs is None:
        args.warmup_epochs = 10 if pretrain else 0
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs/batch-size must be positive and workers nonnegative")
    if args.warmup_epochs != 0 and not 2 <= args.warmup_epochs < args.epochs:
        parser.error("warmup-epochs must be zero or at least two and below epochs")
    if not pretrain and args.warmup_epochs != 0:
        parser.error("the linear recipe does not use warmup")
    if args.resume and (args.checkpoint or args.mode == "evaluate"):
        parser.error("--resume cannot be combined with --checkpoint or evaluate")
    if args.mode == "pretrain" and args.checkpoint:
        parser.error("use --resume to continue pretraining")
    if args.mode != "pretrain" and not (args.checkpoint or args.resume):
        parser.error("linear/evaluate requires --checkpoint (or --resume for linear)")
    for name in ("data_dir", "output_dir", "checkpoint", "resume"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.expanduser().resolve())
    for path in (args.checkpoint, args.resume):
        if path is not None and (not path.is_file() or path.suffix != ".ckpt"):
            parser.error(f"expected an existing .ckpt file: {path}")
    return args


def main() -> None:
    """Run the selected CIFAR-10 phase through the existing Manager lifecycle.

    Args:
        None. Configuration is read from command-line arguments.

    Returns:
        None. Training prints checkpoint paths; evaluation prints test metrics.
    """
    args = _parse_args()
    # Seed before constructing modules as Manager receives prebuilt objects.
    spt.utils.seed_everything(args.seed, workers=True)
    phase_dir = args.output_dir / args.mode
    spt.set(cache_dir=str(phase_dir))
    train_loader = _loader(args, train=True) if args.mode != "evaluate" else None
    val_loader = _loader(args, train=False) if args.mode != "pretrain" else None
    steps = len(train_loader) if train_loader is not None else 0
    if train_loader is not None and steps == 0:
        raise ValueError("The selected batch size leaves no training batches")
    pretrain = args.mode == "pretrain"
    recipe = {
        "version": 1,
        "mode": args.mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "steps_per_epoch": steps,
        "seed": args.seed,
        "peak_lr": 0.5 if pretrain else 0.2,
        "min_lr": 1e-6 if pretrain else 0.002,
        "warmup_epochs": args.warmup_epochs,
        "warmup_start_lr": 0.003,
    }
    if args.resume:
        saved_recipe = (
            _checkpoint(args.resume)
            .get("hyper_parameters", {})
            .get("corinfomax_recipe")
        )
        if saved_recipe != recipe:
            raise ValueError(
                "Resume requires the same phase, seed, batch size, and schedule"
            )
    backbone = spt.backbone.from_torchvision(
        "resnet18", low_resolution=True, weights=None, zero_init_residual=True
    )
    optimizer = {
        "type": "SGD",
        "lr": recipe["peak_lr"],
        "momentum": 0.9,
        "weight_decay": 1e-4 if pretrain else 0.0,
        "exclude_bias_norm": False,
    }
    callbacks = []
    if pretrain:
        module = spt.CorInfoMax(
            encoder_name=backbone,
            embed_dim=512,
            optim={"optimizer": optimizer, "scheduler": _scheduler(recipe)},
        )
    else:
        classifier = torch.nn.Linear(512, 10)
        if args.checkpoint:
            checkpoint = _checkpoint(args.checkpoint)
            if args.mode == "linear":
                if not any(
                    key.startswith("corinfomax_loss.")
                    for key in checkpoint["state_dict"]
                ):
                    raise ValueError(
                        "Linear fitting requires a CorInfoMax pretraining checkpoint"
                    )
                _load_component(backbone, checkpoint, "backbone.")
            else:
                _load_component(backbone, checkpoint, "backbone.backbone.")
                _load_component(
                    classifier, checkpoint, "callbacks_modules.linear_probe."
                )
        module = spt.Module(
            backbone=spt.backbone.EvalOnly(backbone), forward=_features, optim=None
        )
        callbacks.append(
            spt.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=classifier,
                loss=torch.nn.CrossEntropyLoss(),
                optimizer=optimizer,
                scheduler=_scheduler(recipe) if args.mode == "linear" else None,
                metrics={
                    "top1": MulticlassAccuracy(num_classes=10, average="micro"),
                    "top5": MulticlassAccuracy(
                        num_classes=10, top_k=5, average="micro"
                    ),
                },
            )
        )
    module.save_hyperparameters({"corinfomax_recipe": recipe})
    final_checkpoint = None
    best_checkpoint = None
    if args.mode != "evaluate":
        final_checkpoint = ModelCheckpoint(
            filename="final-{epoch:04d}",
            save_top_k=1,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
        )
        callbacks.append(final_checkpoint)
        if args.mode == "linear":
            best_checkpoint = ModelCheckpoint(
                filename="best-test-{epoch:04d}",
                monitor="eval/linear_probe_top1",
                mode="max",
                save_top_k=1,
                save_on_train_epoch_end=False,
                auto_insert_metric_name=False,
            )
            callbacks.append(best_checkpoint)
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        precision="32-true",
        max_epochs=args.epochs,
        callbacks=callbacks,
        default_root_dir=str(phase_dir),
        num_sanity_val_steps=0,
        enable_checkpointing=args.mode != "evaluate",
        logger=False,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        seed=args.seed,
        ckpt_path=str(args.resume) if args.resume else None,
        weights_only=False,
    )
    if args.mode == "evaluate":
        # Manager currently sets this reference only in its training entry point.
        manager._trainer = trainer
        manager.validate()
    else:
        manager()
    report = {"phase": args.mode}
    if final_checkpoint is not None:
        report["final_checkpoint"] = final_checkpoint.best_model_path
    if args.mode != "pretrain":
        report["evaluation_split"] = "CIFAR-10 test (10,000 images)"
        for metric in ("top1", "top5"):
            key = f"eval/linear_probe_{metric}"
            value = trainer.callback_metrics.get(key)
            if value is not None:
                name = f"final_{metric}" if args.mode == "linear" else metric
                report[name] = float(value)
    if best_checkpoint is not None and best_checkpoint.best_model_score is not None:
        report["best_observed_test_top1"] = float(best_checkpoint.best_model_score)
        report["best_test_checkpoint"] = best_checkpoint.best_model_path
        report["selection_note"] = (
            "Final metrics use the final epoch; best test score is descriptive."
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
