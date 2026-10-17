"""Train FroSSL on CIFAR-10 with an online linear probe.

This is an integration example, not a reproduction of the paper's full recipe.
It uses a held-out split of the CIFAR-10 training set for probe validation.
The official test set is evaluated only after training.

Examples::

    python examples/frossl_cifar10.py --smoke
    python examples/frossl_cifar10.py --epochs 100 --num-views 4
"""

import argparse
import json
from pathlib import Path
from typing import Any

import lightning as pl
import torch
import torchmetrics
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, FakeData

import stable_pretraining as spt


def build_data(
    args: argparse.Namespace,
) -> tuple[spt.data.DataModule, DataLoader, dict[str, Any]]:
    """Build aligned training views and disjoint evaluation datasets.

    Args:
        args: Parsed example settings.

    Returns:
        DataModule, held-out test loader, and split information to save.

    Note:
        The official test set is never passed to the DataModule, so its labels
        cannot affect training or validation model selection.
    """
    transforms = spt.data.transforms
    mean = spt.data.dataset_stats.CIFAR10["mean"]
    std = spt.data.dataset_stats.CIFAR10["std"]
    augment = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.ToImage(mean=mean, std=std),
    )
    train_transform = transforms.MultiViewTransform([augment] * args.num_views)
    eval_transform = transforms.Compose(
        transforms.RGB(), transforms.ToImage(mean=mean, std=std)
    )

    if args.smoke:
        train_raw = FakeData(size=16, image_size=(3, 32, 32), num_classes=10)
        val_raw = FakeData(
            size=8, image_size=(3, 32, 32), num_classes=10, random_offset=16
        )
        test_raw = FakeData(
            size=8, image_size=(3, 32, 32), num_classes=10, random_offset=24
        )
        split = {"dataset": "FakeData", "train_size": 16, "val_size": 8, "test_size": 8}
    else:
        cifar = CIFAR10(root=args.data_root, train=True, download=args.download)
        test_raw = CIFAR10(root=args.data_root, train=False, download=args.download)
        train_raw, val_raw = random_split(
            cifar, [45_000, 5_000], generator=torch.Generator().manual_seed(args.seed)
        )
        split = {
            "dataset": "CIFAR-10",
            "train_size": len(train_raw),
            "val_size": len(val_raw),
            "test_size": len(test_raw),
            "split_seed": args.seed,
            "train_indices": train_raw.indices,
            "val_indices": val_raw.indices,
        }

    train_ds = spt.data.FromTorchDataset(
        train_raw, names=("image", "label"), transform=train_transform
    )
    val_ds = spt.data.FromTorchDataset(
        val_raw, names=("image", "label"), transform=eval_transform
    )
    test_ds = spt.data.FromTorchDataset(
        test_raw, names=("image", "label"), transform=eval_transform
    )
    batch_size = 8 if args.smoke else args.batch_size
    loader_options = {"batch_size": batch_size, "num_workers": args.workers}
    data = spt.data.DataModule(
        train=DataLoader(train_ds, shuffle=True, drop_last=True, **loader_options),
        val=DataLoader(val_ds, **loader_options),
    )
    return data, DataLoader(test_ds, **loader_options), split


@torch.inference_mode()
def evaluate_online_probe(
    module: spt.FroSSL, test_loader: DataLoader
) -> dict[str, Any]:
    """Measure final online-probe accuracy on the untouched test set.

    Args:
        module: Trained FroSSL module with an attached ``linear_probe``.
        test_loader: Loader for the official CIFAR-10 test set.

    Returns:
        Exact test accuracy and sample count.

    Note:
        This evaluates the final online probe, not a new post-hoc linear
        classifier trained after freezing the backbone.
    """
    module.eval()
    probe = module.callbacks_modules["linear_probe"]
    correct = total = 0
    for batch in test_loader:
        images = batch["image"].to(module.device)
        labels = batch["label"].to(module.device)
        logits = probe(module.backbone(images))
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += labels.numel()
    return {"online_probe_test_accuracy": correct / total, "test_samples": total}


def main() -> None:
    """Train FroSSL and save the configuration, metrics, and split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/frossl"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--invariance-weight", type=float)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.num_views < 2:
        parser.error("--num-views must be at least 2")
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0:
        parser.error(
            "--epochs and --batch-size must be positive; --workers nonnegative"
        )

    args.data_root = args.data_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    pl.seed_everything(args.seed, workers=True)
    weight = args.invariance_weight
    if weight is None:
        weight = 1.4 if args.num_views == 2 else 2.0
    spt.set(cache_dir=str(args.cache_dir))
    data, test_loader, split = build_data(args)
    hidden_dim, projection_dim = (128, 64) if args.smoke else (2048, 1024)
    projector = torch.nn.Sequential(
        torch.nn.Linear(512, hidden_dim),
        torch.nn.BatchNorm1d(hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.BatchNorm1d(hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, projection_dim),
    )
    module = spt.FroSSL(
        backbone=spt.backbone.from_torchvision("resnet18", low_resolution=True),
        projector=projector,
        optim={"optimizer": {"type": "SGD", "lr": 0.03, "momentum": 0.9}},
        invariance_weight=weight,
    )
    probe = spt.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=torch.nn.Linear(512, 10),
        loss=torch.nn.CrossEntropyLoss(),
        metrics=torchmetrics.classification.MulticlassAccuracy(10, average="micro"),
    )
    trainer = pl.Trainer(
        max_epochs=1 if args.smoke else args.epochs,
        accelerator="cpu" if args.smoke else "auto",
        devices=1,
        logger=False,
        enable_checkpointing=not args.smoke,
        enable_progress_bar=not args.smoke,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[probe],
    )
    manager = spt.Manager(trainer=trainer, module=module, data=data, seed=args.seed)
    manager()

    result = evaluate_online_probe(module, test_loader)
    result["validation_online_probe_accuracy"] = float(
        trainer.callback_metrics["eval/linear_probe_MulticlassAccuracy"]
    )
    result["completed_epochs"] = trainer.current_epoch
    result["global_steps"] = trainer.global_step
    result["evaluation"] = (
        "final online linear probe; synthetic smoke test"
        if args.smoke
        else "final online linear probe; official held-out test set"
    )

    registry_logger = next(
        logger
        for logger in trainer.loggers
        if type(logger).__name__ == "RegistryLogger"
    )
    run_dir = Path(registry_logger.run_dir)
    config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "resolved_invariance_weight": weight,
        "backbone": "resnet18 (low_resolution=True)",
        "projector_dimensions": [512, hidden_dim, hidden_dim, projection_dim],
        "optimizer": {"type": "SGD", "lr": 0.03, "momentum": 0.9},
        "torch_version": torch.__version__,
        "lightning_version": pl.__version__,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (run_dir / "data_split.json").write_text(json.dumps(split, indent=2) + "\n")
    (run_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"run_dir": str(run_dir), **result}, indent=2))


if __name__ == "__main__":
    main()
