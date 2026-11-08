"""Linear evaluation of a frozen ReSSL CIFAR-10 backbone.

Evaluate a frozen student or teacher backbone with a CIFAR-10 linear probe.

Examples:
    python benchmarks/cifar10/ressl-resnet18-lineareval.py \
        --checkpoint /path/to/last.ckpt

    python benchmarks/cifar10/ressl-resnet18-lineareval.py \
        --checkpoint /path/to/last.ckpt \
        --branch teacher
"""

import argparse
import csv
import os
from pathlib import Path

import stable_pretraining as spt
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a pretrained ReSSL Lightning checkpoint.",
    )
    parser.add_argument(
        "--branch",
        choices=("student", "teacher"),
        default=os.environ.get("BRANCH", "student"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="CIFAR-10 root containing cifar-10-batches-py.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "256"))
    )
    parser.add_argument(
        "--epochs", type=int, default=int(os.environ.get("EPOCHS", "100"))
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=float(os.environ["LR"]) if os.environ.get("LR") else None,
    )
    parser.add_argument(
        "--schedule",
        choices=("cosine", "step"),
        default=os.environ.get("SCHEDULE", "cosine"),
    )
    parser.add_argument(
        "--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "4"))
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.lr is None:
        args.lr = 0.2 if args.schedule == "cosine" else 30.0
    output_dir = (
        (args.output_dir or Path.home() / f"ressl-{args.branch}-eval-exact-augment")
        .expanduser()
        .resolve()
    )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Locate checkpoints with: find ~/.cache/stable-pretraining/runs -name '*.ckpt'"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required to run this evaluation.")

    device = torch.device("cuda")
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print("Checkpoint epoch:", ckpt.get("epoch", "unknown"), flush=True)
    print("Checkpoint global_step:", ckpt.get("global_step", "unknown"), flush=True)
    weights = ckpt.get("state_dict", ckpt)
    prefix = f"backbone.{args.branch}."
    branch_weights = {
        key[len(prefix) :]: value
        for key, value in weights.items()
        if key.startswith(prefix)
    }
    if not branch_weights:
        raise RuntimeError(
            f"No {prefix}* weights found. Inspect checkpoint state_dict keys."
        )

    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
    backbone.fc = nn.Identity()
    backbone.load_state_dict(branch_weights, strict=True)
    backbone.requires_grad_(False)
    backbone.to(device).eval()
    print(
        f"Loaded and froze {args.branch} backbone:",
        len(branch_weights),
        "state entries",
        flush=True,
    )

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalize = transforms.Normalize(mean, std)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    if args.data_dir is not None:
        data_root = args.data_dir.expanduser()
    elif os.environ.get("DATA_DIR"):
        data_root = Path(os.environ["DATA_DIR"]).expanduser()
    else:
        base = Path(
            os.environ.get(
                "STABLE_PRETRAINING_DATA_DIR",
                str(Path.home() / ".cache" / "stable-pretraining" / "data"),
            )
        ).expanduser()
        data_root = base / "cifar10"
    print("CIFAR-10 root:", data_root, flush=True)
    if not (data_root / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(
            f"Existing CIFAR-10 dataset not found at {data_root}. "
            "Pass --data-dir pointing to the parent of cifar-10-batches-py."
        )

    train_set = torchvision.datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=False,
        transform=train_transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=False,
        transform=test_transform,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    classifier = nn.Linear(512, 10).to(device)
    nn.init.normal_(classifier.weight, mean=0.0, std=0.01)
    nn.init.zeros_(classifier.bias)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0,
        nesterov=True,
    )
    if args.schedule == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[60, 80],
            gamma=0.1,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs * len(train_loader),
        )

    criterion = nn.CrossEntropyLoss()
    history_path = output_dir / "metrics.csv"
    best_accuracy = (
        -1.0
    )  # Diagnostic only; do not select checkpoints using test accuracy.

    with history_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "test_accuracy", "lr"])
        for epoch in range(args.epochs):
            classifier.eval()  # Match the reference evaluation procedure.
            total_loss = 0.0
            total_seen = 0
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.no_grad():
                    features = backbone(images)
                logits = classifier(features)
                loss = criterion(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if args.schedule == "cosine":
                    scheduler.step()
                total_loss += loss.item() * labels.size(0)
                total_seen += labels.size(0)

            classifier.eval()
            correct = 0
            tested = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    predictions = classifier(backbone(images)).argmax(dim=1)
                    correct += (predictions == labels).sum().item()
                    tested += labels.size(0)

            accuracy = correct / tested
            current_lr = optimizer.param_groups[0]["lr"]
            writer.writerow([epoch + 1, total_loss / total_seen, accuracy, current_lr])
            file.flush()
            best_accuracy = max(best_accuracy, accuracy)
            print(
                f"Epoch {epoch + 1:3d}/{args.epochs} | "
                f"loss={total_loss / total_seen:.4f} | "
                f"test_top1={accuracy:.2%} | best={best_accuracy:.2%} | "
                f"lr={current_lr:.5g}",
                flush=True,
            )
            if args.schedule == "step":
                scheduler.step()

    torch.save(
        {
            "classifier": classifier.state_dict(),
            "epoch": args.epochs,
            "accuracy": accuracy,
            "source_checkpoint": str(args.checkpoint),
        },
        output_dir / "last_linear_probe.pt",
    )
    print(f"Final test accuracy: {accuracy:.2%}")
    print(f"Best observed test accuracy: {best_accuracy:.2%} (diagnostic only)")
    print(f"Results: {history_path}")


if __name__ == "__main__":
    main()
