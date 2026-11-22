"""DirectPred training on CIFAR-10.

Follows the CIFAR-10 / ResNet-18 setting of Tian, Chen and Ganguli,
"Understanding self-supervised learning dynamics without contrastive pairs"
(ICML 2021), Table 8: rho = 0.3 and eps = 0.1 (Table 6 shows that a smaller
eps such as 0.01 is weaker), batch 128, SGD (lr 0.03, momentum 0.9, wd 4e-4)
at a constant learning rate, EMA 0.996, a projector 512-128 with non-affine
batch norm, and the DirectPred predictor recomputed every step.

The paper reports linear accuracy of the frozen encoder (Table 8: 85.21 +- 0.23
after 100 epochs, 88.88 +- 0.15 after 300 epochs). Evaluate a checkpoint of
this script that way with ``directpred-linear-eval.py``; the online probes
logged during training are only an indicator.

The symmetrized loss sums the two directions while spt's forward function
averages them, so the loss is scaled by 2 to keep the learning rate and weight
decay above.

``--predictor linear`` and ``--predictor linear-sym`` run the paper's baseline
instead: the same recipe with a linear predictor (regular or symmetric)
trained by gradient descent (Table 8: 84.49 +- 0.20 after 100 epochs).

Usage:
    python benchmarks/cifar10/directpred-resnet18.py
    python benchmarks/cifar10/directpred-resnet18.py --epochs 300
    python benchmarks/cifar10/directpred-resnet18.py --predictor linear
    python benchmarks/cifar10/directpred-resnet18.py --predictor linear-sym
    python benchmarks/cifar10/directpred-linear-eval.py --ckpt <run>/checkpoints/last.ckpt
"""

import argparse
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--ckpt-path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--precision", type=str, default="16-mixed")
parser.add_argument("--wandb", action="store_true")
parser.add_argument(
    "--predictor",
    choices=["direct", "linear", "linear-sym"],
    default="direct",
    help="direct: DirectPred; linear / linear-sym: baseline predictor trained by SGD",
)
args = parser.parse_args()

# CIFAR-10 normalization statistics.
CIFAR10_STATS = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])


def directpred_view():
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToImage(**CIFAR10_STATS),
    )


class ScaledLoss(nn.Module):
    """Multiply a two-view loss by a constant."""

    def __init__(self, loss, scale):
        super().__init__()
        self.loss, self.scale = loss, scale

    def forward(self, x, y):
        return self.scale * self.loss(x, y)


class SymmetricLinear(nn.Module):
    """Bias-free linear map with a symmetric weight, trained by gradient."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(nn.Linear(dim, dim, bias=False).weight.detach())

    def forward(self, x):
        return nn.functional.linear(x, (self.weight + self.weight.T) / 2)


data_dir = get_data_dir("cifar10")
cifar_train = torchvision.datasets.CIFAR10(
    root=str(data_dir), train=True, download=True
)
cifar_val = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True)

train_dataset = spt.data.FromTorchDataset(
    cifar_train,
    names=["image", "label"],
    transform=transforms.MultiViewTransform([directpred_view(), directpred_view()]),
)
val_dataset = spt.data.FromTorchDataset(
    cifar_val,
    names=["image", "label"],
    transform=transforms.Compose(
        transforms.RGB(),
        transforms.Resize((32, 32)),
        transforms.ToImage(**CIFAR10_STATS),
    ),
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

backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
backbone.fc = nn.Identity()

projector = nn.Sequential(
    nn.Linear(512, 512),
    nn.BatchNorm1d(512, affine=False),
    nn.ReLU(inplace=True),
    nn.Linear(512, 128),
)

if args.predictor == "direct":
    predictor = spt.backbone.DirectPredictor(128, rho=0.3, eps=0.1)
    forward_fn = forward.directpred
else:
    # Baseline: a linear predictor (regular or symmetric) trained by gradient.
    if args.predictor == "linear":
        predictor = nn.Linear(128, 128, bias=False)
    else:
        predictor = SymmetricLinear(128)
    forward_fn = forward.byol

module = spt.Module(
    backbone=spt.backbone.TeacherStudentWrapper(
        backbone, base_ema_coefficient=0.996, final_ema_coefficient=0.996
    ),
    projector=spt.backbone.TeacherStudentWrapper(
        projector, base_ema_coefficient=0.996, final_ema_coefficient=0.996
    ),
    predictor=predictor,
    forward=forward_fn,
    byol_loss=ScaledLoss(spt.losses.BYOLLoss(), 2.0),
    optim={
        "optimizer": {
            "type": "SGD",
            "lr": 0.03,
            "momentum": 0.9,
            "weight_decay": 4e-4,
        },
        # Constant learning rate.
        "scheduler": {"type": "ConstantLR", "factor": 1.0, "total_iters": 0},
        "interval": "step",
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
    k=10,
    num_classes=10,
)

logger = None
if args.wandb:
    logger = WandbLogger(
        project="cifar10-directpred",
        name="directpred-resnet18",
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
