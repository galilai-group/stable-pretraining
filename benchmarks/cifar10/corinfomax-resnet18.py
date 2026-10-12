"""CorInfoMax training on CIFAR-10.

Reproduces the CIFAR-10 / ResNet-18 recipe of Ozsoy et al. (NeurIPS 2022),
https://github.com/serdarozsoy/corinfomax-ssl (reported linear top-1: 93.18%):
1000 epochs, batch 512, SGD (lr 0.5, momentum 0.9, wd 1e-4), 10 warmup epochs
then cosine decay to 1e-6, projector 2048-2048-64 without batch norm,
alpha = 250, EMA weights la_R = la_mu = 0.01, eps = 1e-8, and asymmetric
augmentations (blur 1.0 / 0.1, solarize 0.0 / 0.2).

Usage:
    python benchmarks/cifar10/corinfomax-resnet18.py
    python benchmarks/cifar10/corinfomax-resnet18.py --epochs 2  # smoke run
"""

import argparse
import sys
from pathlib import Path

import lightning as pl
import torch
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger
from torchvision.transforms import InterpolationMode

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms
from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealing

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--ckpt-path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
# The reference implementation trains in full fp32 ("32-true").
parser.add_argument("--precision", type=str, default="16-mixed")
parser.add_argument("--wandb", action="store_true")
args = parser.parse_args()


def corinfomax_view(blur_p, solarize_p):
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop(
            (32, 32), scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILGaussianBlur(p=blur_p),
        # Applied to PIL images, so the threshold is in pixel units (0-255).
        transforms.RandomSolarize(threshold=128, p=solarize_p),
        transforms.ToImage(**spt.data.static.CIFAR10),
    )


corinfomax_transform = transforms.MultiViewTransform(
    [
        corinfomax_view(blur_p=1.0, solarize_p=0.0),
        corinfomax_view(blur_p=0.1, solarize_p=0.2),
    ]
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.ToImage(**spt.data.static.CIFAR10),
)

data_dir = get_data_dir("cifar10")
cifar_train = torchvision.datasets.CIFAR10(
    root=str(data_dir), train=True, download=True
)
cifar_val = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True)

train_dataset = spt.data.FromTorchDataset(
    cifar_train,
    names=["image", "label"],
    transform=corinfomax_transform,
)
val_dataset = spt.data.FromTorchDataset(
    cifar_val,
    names=["image", "label"],
    transform=val_transform,
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    drop_last=True,
    shuffle=True,
    persistent_workers=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=256,
    num_workers=args.num_workers,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

backbone = spt.backbone.from_torchvision(
    "resnet18",
    low_resolution=True,
    zero_init_residual=True,
)
backbone.fc = torch.nn.Identity()

# Linear-ReLU MLP without batch norm, bias-free output layer (paper recipe).
projector = torch.nn.Sequential(
    torch.nn.Linear(512, 2048),
    torch.nn.ReLU(inplace=True),
    torch.nn.Linear(2048, 2048),
    torch.nn.ReLU(inplace=True),
    torch.nn.Linear(2048, 64, bias=False),
)

module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=forward.corinfomax,
    corinfomax_loss=spt.losses.CorInfoMaxLoss(
        proj_dim=64,
        sim_coeff=250.0,
        cov_coeff=1.0,
        cov_momentum=0.01,
        mean_momentum=0.01,
        init_scale=1.0,
        epsilon=1e-8,
    ),
    optim={
        "optimizer": {
            "type": "SGD",
            "lr": 0.5,
            "momentum": 0.9,
            "weight_decay": 1e-4,
        },
        # Custom kwargs disable the scheduler's default total_steps, so build
        # it from the trainer: 1% linear warmup (10 of 1000 epochs) from
        # lr 3e-3, then cosine decay to 1e-6.
        "scheduler": lambda optimizer, module: LinearWarmupCosineAnnealing(
            optimizer,
            total_steps=module.trainer.estimated_stepping_batches,
            peak_step=0.01,
            start_factor=3e-3 / 0.5,
            end_lr=1e-6,
        ),
        "interval": "step",
    },
)

linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(512, 10),
    loss=torch.nn.CrossEntropyLoss(),
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
    k=10,
    num_classes=10,
)

logger = None
if args.wandb:
    logger = WandbLogger(
        project="cifar10-corinfomax",
        name="corinfomax-resnet18",
        log_model=False,
    )

trainer = pl.Trainer(
    max_epochs=args.epochs,
    num_sanity_val_steps=0,
    callbacks=[knn_probe, linear_probe],
    precision=args.precision,
    logger=logger,
    enable_checkpointing=False,
)

manager = spt.Manager(
    trainer=trainer,
    module=module,
    data=data,
    ckpt_path=args.ckpt_path,
    seed=args.seed,
)
manager()
