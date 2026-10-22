"""CIFAR-10 DCL benchmark with a matched SimCLR comparison."""

import argparse
from pathlib import Path

import lightning as pl
import torch
import torchvision
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the experiment configuration.

    Args:
        argv: Command-line arguments; None reads sys.argv.

    Returns:
        Configuration, including reduced CPU settings when --smoke is supplied.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("dcl", "simclr"), default="dcl")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--base-lr",
        type=float,
        default=0.03,
        help="SGD learning rate at batch size 256; scaled linearly.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--accelerator", choices=("cpu", "gpu", "mps", "auto"), default="auto"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cifar10-dcl"))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 32 synthetic training images and 16 validation images on CPU.",
    )
    args = parser.parse_args(argv)
    if args.smoke:
        args.epochs, args.batch_size, args.num_workers = 1, 8, 0
        args.accelerator = "cpu"
    if args.epochs < 1 or args.batch_size < 2 or args.num_workers < 0:
        parser.error("epochs >= 1, batch-size >= 2, and num-workers >= 0 are required")
    for name in ("temperature", "base_lr"):
        value = getattr(args, name)
        if not 0 < value < float("inf"):
            parser.error(f"{name} must be positive and finite")
    return args


def run_experiment(
    args: argparse.Namespace, callbacks: list[pl.Callback] | None = None
) -> tuple[pl.Trainer, spt.Module]:
    """Train and evaluate one method through the standard Manager pipeline.

    Args:
        args: Configuration returned by parse_args.
        callbacks: Optional extra callbacks, used by the standalone smoke check.

    Returns:
        Completed trainer and trained module.
    """
    pl.seed_everything(args.seed, workers=True)
    if args.smoke:
        torch.set_num_threads(2)
    spt.set(
        cache_dir=str(args.output_dir.expanduser().resolve()),
        default_callbacks={"hf_checkpoint": False},
    )

    # Keep the same two-view augmentation recipe as the existing SimCLR script.
    train_transform = transforms.MultiViewTransform(
        [
            transforms.Compose(
                transforms.RGB(),
                transforms.RandomResizedCrop((32, 32)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToImage(**spt.data.static.CIFAR10),
            )
            for _ in range(2)
        ]
    )
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((32, 32)),
        transforms.ToImage(**spt.data.static.CIFAR10),
    )
    if args.smoke:
        train_source = torchvision.datasets.FakeData(
            size=32,
            image_size=(3, 32, 32),
            num_classes=10,
            random_offset=0,
        )
        val_source = torchvision.datasets.FakeData(
            size=16,
            image_size=(3, 32, 32),
            num_classes=10,
            random_offset=10000,
        )
    else:
        root = str(args.data_dir.expanduser().resolve())
        train_source = torchvision.datasets.CIFAR10(root, train=True, download=True)
        val_source = torchvision.datasets.CIFAR10(root, train=False, download=True)
    train_dataset = spt.data.FromTorchDataset(
        train_source,
        names=["image", "label"],
        transform=train_transform,
    )
    val_dataset = spt.data.FromTorchDataset(
        val_source,
        names=["image", "label"],
        transform=val_transform,
    )
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    data = spt.data.DataModule(train=train_loader, val=val_loader)

    backbone = spt.backbone.from_torchvision(
        "resnet18", low_resolution=True, weights=None
    )
    backbone.fc = torch.nn.Identity()
    # Match the existing benchmark's projection head for both objectives.
    projector = torch.nn.Sequential(
        torch.nn.Linear(512, 2048),
        torch.nn.BatchNorm1d(2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 2048),
        torch.nn.BatchNorm1d(2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 256),
    )
    objective = (
        {"dcl_loss": spt.losses.DCLLoss(temperature=args.temperature)}
        if args.method == "dcl"
        else {"simclr_loss": spt.losses.NTXEntLoss(temperature=args.temperature)}
    )
    module = spt.Module(
        forward=getattr(forward, args.method),
        backbone=backbone,
        projector=projector,
        hparams={
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "learning_rate": args.base_lr * args.batch_size / 256,
        },
        optim={
            "optimizer": {
                "type": "SGD",
                "lr": args.base_lr * args.batch_size / 256,
                "momentum": 0.9,
                "weight_decay": 1e-4,
            },
            "scheduler": {
                "type": "CosineAnnealingLR",
                "T_max": args.epochs * len(train_loader),
            },
            "interval": "step",
        },
        **objective,
    )
    knn = spt.OnlineKNN(
        name="knn_probe",
        input="embedding",
        target="label",
        input_dim=512,
        num_classes=10,
        queue_length=64 if args.smoke else 20000,
        k=10,
        distance_metric="cosine",
        temperature=0.07,
        chunk_size=256,
        metrics={"accuracy": MulticlassAccuracy(num_classes=10, average="micro")},
    )
    linear_probe = spt.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=torch.nn.Linear(512, 10),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"top1": MulticlassAccuracy(num_classes=10, average="micro")},
    )
    checkpoint = ModelCheckpoint(
        save_top_k=1,
        every_n_epochs=args.epochs,
        save_on_train_epoch_end=False,
        filename="final",
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.epochs,
        precision="32-true",
        num_sanity_val_steps=0,
        log_every_n_steps=1 if args.smoke else 50,
        # Manager installs its local registry/CSV logger; no W&B account needed.
        logger=False,
        callbacks=[
            knn,
            linear_probe,
            checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
            *(callbacks or []),
        ],
    )
    spt.Manager(trainer=trainer, module=module, data=data, seed=args.seed)()
    print(f"Run outputs: {trainer.logger.log_dir}")
    print(
        f"Final online kNN accuracy: {trainer.callback_metrics['eval/knn_probe_accuracy']}"
    )
    print(
        f"Final online linear top-1: {trainer.callback_metrics['eval/linear_probe_top1']}"
    )
    print(f"Final checkpoint: {checkpoint.best_model_path}")
    return trainer, module


if __name__ == "__main__":
    run_experiment(parse_args())
