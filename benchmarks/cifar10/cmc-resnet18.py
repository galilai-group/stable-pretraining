"""CMC training on CIFAR10."""

import lightning as pl
import torch
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

cmc_transform = transforms.Compose(
    transforms.RGB(),
    transforms.RandomResizedCrop((32, 32)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToImage(),
    transforms.RGBToLabViews(),
)

# CMC's two encoders are channel-specific, so even eval must split RGB into
# L/ab views (there is no single-view eval path, unlike SimCLR/BYOL/etc.).
val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.ToImage(),
    transforms.RGBToLabViews(),
)

data_dir = get_data_dir("cifar10")
cifar_train = torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=True)
cifar_val = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True)

train_dataset = spt.data.FromTorchDataset(
    cifar_train,
    names=["image", "label"],
    transform=cmc_transform,
)
val_dataset = spt.data.FromTorchDataset(
    cifar_val,
    names=["image", "label"],
    transform=val_transform,
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=256,
    num_workers=8,
    drop_last=True,
    shuffle=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=256,
    num_workers=10,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

backbone_l = spt.backbone.from_torchvision(
    "resnet18", low_resolution=True, in_channels=1
)
backbone_ab = spt.backbone.from_torchvision(
    "resnet18", low_resolution=True, in_channels=2
)

projector_l = torch.nn.Sequential(
    torch.nn.Linear(512, 128),
    torch.nn.BatchNorm1d(128),
    torch.nn.ReLU(inplace=True),
    torch.nn.Linear(128, 128),
)
projector_ab = torch.nn.Sequential(
    torch.nn.Linear(512, 128),
    torch.nn.BatchNorm1d(128),
    torch.nn.ReLU(inplace=True),
    torch.nn.Linear(128, 128),
)

module = spt.Module(
    backbone_l=backbone_l,
    backbone_ab=backbone_ab,
    projector_l=projector_l,
    projector_ab=projector_ab,
    forward=forward.cmc,
    cmc_loss=spt.losses.CLIPLoss(temperature=0.07),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": 5,
            "weight_decay": 1e-6,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
    },
)

linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(1024, 10),
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
    input_dim=1024,
    k=10,
)

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="cifar10-cmc",
    log_model=False,
)

lr_monitor = LearningRateMonitor(logging_interval="step")

trainer = pl.Trainer(
    max_epochs=1000,
    num_sanity_val_steps=0,
    callbacks=[knn_probe, linear_probe, lr_monitor],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
