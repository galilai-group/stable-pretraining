"""BayesNCL training on CIFAR10.

Adapt the official CIFAR recipe to CIFAR10 using the repository's data and
evaluation callbacks. This recipe has not been benchmarked against the paper.

Run one training and validation batch on CPU:
    python benchmarks/cifar10/bayesncl-resnet18.py --fast-dev-run

Run ten training and validation batches on CPU:
    python benchmarks/cifar10/bayesncl-resnet18.py --fast-dev-run 10

Run the 200-epoch recipe:
    python benchmarks/cifar10/bayesncl-resnet18.py

Reference: https://github.com/Peng-Cui-2003/BayesNCL
"""

import argparse
import sys
from pathlib import Path

import lightning as pl
import torch
import torchmetrics
import torchvision
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from torch import nn

import stable_pretraining as spt


def main(fast_dev_run: int = 0) -> None:
    """Train BayesNCL or check a few training and validation batches.

    Args:
        fast_dev_run: Number of small training and validation batches to run
            on CPU without data-loading workers. Zero runs the full recipe.
    """
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_data_dir

    spt.set(requeue_checkpoint=False)
    pl.seed_everything(42, workers=True)
    batch_size = 8 if fast_dev_run else 256
    num_workers = 0 if fast_dev_run else 8
    max_epochs = 200
    transforms = spt.data.transforms

    train_transform = transforms.MultiViewTransform(
        [
            transforms.Compose(
                transforms.RGB(),
                transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.PILGaussianBlur(p=0.5),
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

    data_dir = get_data_dir("cifar10")
    train_dataset = spt.data.FromTorchDataset(
        torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=True),
        names=["image", "label"],
        transform=train_transform,
    )
    val_dataset = spt.data.FromTorchDataset(
        torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True),
        names=["image", "label"],
        transform=val_transform,
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
    backbone.fc = nn.Identity()
    projector = nn.Sequential(
        nn.Linear(512, 2048),
        nn.ReLU(),
        nn.Linear(2048, 256),
    )
    gating_head = nn.Sequential(
        nn.Linear(512, 2048),
        nn.ReLU(),
        nn.Linear(2048, 256),
    )

    # The reference LARS implementation handles bias/norm exclusions internally.
    optimizer_config = {
        "type": "LARS",
        "weight_decay": 1e-4,
        "clip_lr": True,
        "eta": 0.02,
        "exclude_bias_n_norm": True,
        "exclude_bias_norm": False,
    }
    scheduler_config = {
        "type": "LinearWarmupCosineAnnealingLR",
        "warmup_steps": 10 * len(train_dataloader),
        "max_steps": max_epochs * len(train_dataloader),
        "warmup_start_lr": 3e-5,
    }
    module = spt.Module(
        backbone=backbone,
        projector=projector,
        gating_head=gating_head,
        forward=spt.forward.bayesncl,
        bayesncl_loss=spt.losses.BayesNCLLoss(
            temperature=0.2, kl_weight=3e-5, prior_probability=0.8
        ),
        gate_temperature=1.0,
        gate_threshold=0.5,
        optim={
            "representation": {
                "modules": "backbone|projector",
                "optimizer": dict(optimizer_config, lr=0.4),
                "scheduler": scheduler_config,
                "interval": "step",
            },
            "gates": {
                "modules": "gating_head",
                "optimizer": dict(optimizer_config, lr=0.1),
                "scheduler": scheduler_config,
                "interval": "step",
            },
        },
    )

    linear_probe = spt.callbacks.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=nn.Linear(512, 10),
        loss=nn.CrossEntropyLoss(),
        metrics={
            "top1": torchmetrics.classification.MulticlassAccuracy(10),
            "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
        },
    )
    knn_probe = spt.callbacks.OnlineKNN(
        name="knn_probe",
        input="embedding",
        target="label",
        queue_length=20000,
        metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
        input_dim=512,
        num_classes=10,
        k=10,
    )
    callbacks = [knn_probe, linear_probe]
    if not fast_dev_run:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    accelerator = "gpu" if torch.cuda.is_available() and not fast_dev_run else "cpu"
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        fast_dev_run=fast_dev_run,
        accelerator=accelerator,
        devices=1,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        precision="16-mixed" if accelerator == "gpu" else "32-true",
        logger=CSVLogger(save_dir="logs", name="bayesncl-cifar10"),
        enable_checkpointing=False,
    )
    manager = spt.Manager(trainer=trainer, module=module, data=data, seed=42)
    manager()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast-dev-run",
        nargs="?",
        type=int,
        const=1,
        default=0,
        metavar="BATCHES",
        help="Run this many small training and validation batches on CPU (default: 1).",
    )
    args = parser.parse_args()
    main(fast_dev_run=args.fast_dev_run)
