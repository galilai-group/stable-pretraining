"""ReSA training on CIFAR-10 or CIFAR-100 with a ResNet-18.

Follows the official CIFAR recipe (https://github.com/winci-ai/resa_cifar).
Paper (Table 1, ResNet-18, 1000 epochs): CIFAR-10 93.53 linear / 93.02 k-NN,
CIFAR-100 72.21 / 66.83.

Usage:
    python benchmarks/cifar10/resa-resnet18.py
    python benchmarks/cifar10/resa-resnet18.py --dataset cifar100
    python benchmarks/cifar10/resa-resnet18.py --epochs 2 --limit-train-batches 20  # smoke test
"""

import argparse
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision.transforms import InterpolationMode

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.forward import resa

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

# Normalization statistics from the official code (src/transform.py).
STATS = {
    "cifar10": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
    "cifar100": {"mean": (0.5071, 0.4865, 0.4409), "std": (0.2673, 0.2564, 0.2762)},
}
DATASETS = {
    "cifar10": (torchvision.datasets.CIFAR10, 10),
    "cifar100": (torchvision.datasets.CIFAR100, 100),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="cifar10")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--lr", type=float, default=0.3, help="base learning rate, scaled by batch/256"
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--ema", type=float, default=0.996, help="base EMA momentum")
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="also keep a checkpoint every N epochs (0 to disable)",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="cap batches per epoch, for smoke tests",
    )
    parser.add_argument("--cache-dir", default=None, help="where runs are written")
    return parser.parse_args()


class ReSADiagnostics(pl.Callback):
    """Log per-epoch statistics of the soft labels.

    Each row of the soft-label matrix is one image's target distribution over
    the batch.

    - ``diagonal_mass``: weight an image puts on itself (1 for one-hot targets).
    - ``same_class_share``: fraction of the weight on other images that goes to
      images of the same class (chance is about 1 / num_classes).
    - ``target_entropy``: row entropy, a lower bound on the cross-entropy loss.

    Labels are used only for logging. Assumes a single process, so the soft
    labels' columns are this batch's images.
    """

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        assignment = outputs.get("resa_assignment")
        if assignment is None or "label" not in outputs:
            return
        n = assignment.shape[0]
        labels = outputs["label"][:n]
        idx = torch.arange(n, device=assignment.device)
        off_diagonal = assignment.clone()
        off_diagonal[idx, idx] = 0
        same_class = (labels[:, None] == labels[None, :]).to(assignment.dtype)
        share = (off_diagonal * same_class).sum(1) / off_diagonal.sum(1).clamp_min(
            1e-12
        )
        pl_module.log_dict(
            {
                "train/resa_diagonal_mass": assignment[idx, idx].mean(),
                "train/resa_same_class_share": share.mean(),
                "train/resa_target_entropy": torch.special.entr(assignment)
                .sum(1)
                .mean(),
            },
            on_step=False,
            on_epoch=True,
        )


def main():
    args = parse_args()
    if args.cache_dir:
        spt.set(cache_dir=args.cache_dir)
    stats = STATS[args.dataset]
    dataset_cls, num_classes = DATASETS[args.dataset]

    weak = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(**stats),
    )
    strong = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop(
            (32, 32), scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomSolarize(threshold=128, p=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(**stats),
    )
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((32, 32)),
        transforms.ToImage(**stats),
    )

    data_dir = get_data_dir(args.dataset)
    train_dataset = spt.data.FromTorchDataset(
        dataset_cls(root=str(data_dir), train=True, download=True),
        names=["image", "label"],
        # The weak view comes first: ReSA builds its soft labels from view 0.
        transform=transforms.MultiViewTransform([weak, strong]),
    )
    val_dataset = spt.data.FromTorchDataset(
        dataset_cls(root=str(data_dir), train=False, download=True),
        names=["image", "label"],
        transform=val_transform,
    )
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, drop_last=True, shuffle=True, **loader_kwargs
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, **loader_kwargs)
    data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

    backbone = spt.backbone.from_torchvision(
        "resnet18", low_resolution=True, weights=None
    )
    backbone.fc = nn.Identity()
    projector = nn.Sequential(
        nn.Linear(512, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512, affine=False),
    )
    # TeacherStudentCallback updates the teacher every step but recomputes the
    # momentum once per epoch; the official code recomputes it every iteration.
    # They differ by at most ~6e-6 at 1000 epochs, so we keep the library's callback.
    ema = dict(warm_init=True, base_ema_coefficient=args.ema, final_ema_coefficient=1.0)

    steps_per_epoch = len(train_dataloader)
    if args.limit_train_batches:
        steps_per_epoch = min(steps_per_epoch, args.limit_train_batches)
    max_steps = args.epochs * steps_per_epoch
    # The cosine phase divides by (max_steps - warmup_steps), so keep it non-empty.
    warmup_steps = min(args.warmup_epochs * steps_per_epoch, max_steps - 1)
    lr = args.lr * args.batch_size / 256
    module = spt.Module(
        backbone=spt.TeacherStudentWrapper(backbone, **ema),
        projector=spt.TeacherStudentWrapper(projector, **ema),
        forward=resa,
        resa_loss=spt.losses.ReSALoss(temperature=args.temperature),
        optim={
            "optimizer": {
                "type": "SGD",
                "lr": lr,
                "momentum": 0.9,
                "weight_decay": args.weight_decay,
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
                "warmup_start_lr": 0.0,
                "eta_min": 0.1 * lr,
            },
            "interval": "step",
        },
    )

    callbacks = [
        spt.callbacks.OnlineProbe(
            module,
            name="linear_probe",
            input="embedding",
            target="label",
            probe=nn.Linear(512, num_classes),
            loss=nn.CrossEntropyLoss(),
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
                "top5": torchmetrics.classification.MulticlassAccuracy(
                    num_classes, top_k=5
                ),
            },
        ),
        spt.callbacks.OnlineKNN(
            name="knn_probe",
            input="embedding",
            target="label",
            queue_length=20000,
            metrics={
                "accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes)
            },
            input_dim=512,
            k=10,
        ),
        ReSADiagnostics(),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if args.checkpoint_every > 0:
        callbacks.append(
            ModelCheckpoint(
                filename="epoch{epoch:04d}",
                auto_insert_metric_name=False,
                every_n_epochs=args.checkpoint_every,
                save_top_k=-1,
            )
        )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        precision=args.precision,
        limit_train_batches=args.limit_train_batches,
    )
    manager = spt.Manager(trainer=trainer, module=module, data=data, seed=args.seed)
    manager()


if __name__ == "__main__":
    main()
