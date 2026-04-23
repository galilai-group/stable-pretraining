import lightning as pl
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from torch import nn

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms
from stable_pretraining.callbacks.rankme import RankMe
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils import get_data_dir

simclr_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=1.0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
    ]
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToImage(**spt.data.static.ImageNet),
)

data_dir = get_data_dir("imagenet100")

train_dataset = spt.data.HFDataset(
    "clane9/imagenet-100",
    split="train",
    cache_dir=str(data_dir),
    transform=simclr_transform,
)
val_dataset = spt.data.HFDataset(
    "clane9/imagenet-100",
    split="validation",
    cache_dir=str(data_dir),
    transform=val_transform,
)

batch_size = 256
world_size = 1
total_batch_size = batch_size * world_size
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    shuffle=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

backbone = spt.backbone.from_torchvision(
    "resnet50",
    low_resolution=False,
)
backbone.fc = torch.nn.Identity()

projector = nn.Sequential(
    nn.Linear(2048, 2048, bias=False),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 128, bias=False),
    spt.utils.BatchNorm1dNoBias(128),
)

module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=forward.simclr_forward,
    simclr_loss=spt.losses.NTXEntLoss(temperature=0.5),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": 0.3 * batch_size / 256,
            "weight_decay": 1e-4,
            "clip_lr": True,
            "eta": 0.02,
            "exclude_bias_n_norm": True,
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
    probe=torch.nn.Linear(2048, 100),
    loss=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(100),
        "top5": torchmetrics.classification.MulticlassAccuracy(100, top_k=5),
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(100)},
    input_dim=2048,
    k=20,
)


# RankMe for comparison
rankme = RankMe(
    name="rankme",
    target="embedding",
    queue_length=2048,
    target_shape=2048,
)

wandb_logger = WandbLogger(
    project="imagenet100-simclr",
    name="simclr-resnet50",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=400,
    num_sanity_val_steps=0,
    callbacks=[knn_probe, linear_probe, rankme],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
    devices=1,
    accelerator="gpu",
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
