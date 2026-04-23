"""SwAV training on ImageNet-100 with ResNet-50."""

import lightning as pl
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from torch import nn

import stable_pretraining as spt
from stable_pretraining.forward import swav_forward
from stable_pretraining.data import transforms
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir


class FreezePrototypesCallback(pl.Callback):
    """Freeze the prototypes layer for the first few epochs."""

    def __init__(self, freeze_epochs=1):
        self.freeze_epochs = freeze_epochs

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Zero out the gradients for the prototypes layer if within freeze_epochs."""
        if trainer.current_epoch < self.freeze_epochs:
            for param in pl_module.prototypes.parameters():
                if param.grad is not None:
                    param.grad.zero_()


BATCH_SIZE = 512
MAX_EPOCHS = 500
WARMUP_EPOCHS = 10
END_LR = 6e-4 if BATCH_SIZE <= 256 else 4.8e-3  # Parameters used in the original paper
LR = 0.6 if BATCH_SIZE <= 256 else 4.8  # Parameters used in the original paper
# Queue parameters from the paper
USE_QUEUE = True if BATCH_SIZE < 256 else False
QUEUE_LENGTH = 3840
START_QUEUE_AT_EPOCH = 15


swav_transform = transforms.MultiViewTransform(
    {
        # Global crop 1: 224x224 with strong augmentations
        "global_1": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, p=1.0),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        # Global crop 2: 224x224
        "global_2": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, p=0.1),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        # Local crops (6 total): 96x96 to capture finer details
        "local_1": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(96, scale=(0.05, 0.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        "local_2": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(96, scale=(0.05, 0.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        "local_3": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(96, scale=(0.05, 0.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        "local_4": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(96, scale=(0.05, 0.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        "local_5": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(96, scale=(0.05, 0.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        "local_6": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(96, scale=(0.05, 0.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
    }
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
    transform=swav_transform,
)
val_dataset = spt.data.HFDataset(
    "clane9/imagenet-100",
    split="validation",
    cache_dir=str(data_dir),
    transform=val_transform,
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    shuffle=True,
)

val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=8,
    persistent_workers=True,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

backbone = spt.backbone.from_torchvision("resnet50", low_resolution=False, weights=None)
backbone.fc = torch.nn.Identity()

projector = nn.Sequential(
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 128),
)

prototypes = nn.Linear(128, 3000, bias=False)

steps_per_epoch = len(train_dataloader)
total_steps = MAX_EPOCHS * steps_per_epoch
peak_step_for_warmup = WARMUP_EPOCHS * steps_per_epoch

module = spt.Module(
    backbone=backbone,
    projector=projector,
    prototypes=prototypes,
    forward=swav_forward,
    use_queue=USE_QUEUE,
    queue_length=QUEUE_LENGTH,
    start_queue_at_epoch=START_QUEUE_AT_EPOCH,
    projection_dim=128,
    swav_loss=spt.losses.joint_embedding.SwAVLoss(
        temperature=0.1,
        sinkhorn_iterations=3,
    ),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": LR,
            "weight_decay": 1e-6,
            "clip_lr": True,
            "eta": 0.02,
            "exclude_bias_n_norm": True,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "peak_step": peak_step_for_warmup,
            "total_steps": total_steps,
            "end_lr": END_LR,
        },
        "interval": "step",
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

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="imagenet100-swav-resnet50",
    log_model=False,
)

lr_monitor = LearningRateMonitor(logging_interval="step")

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    num_sanity_val_steps=0,
    callbacks=[
        knn_probe,
        linear_probe,
        FreezePrototypesCallback(freeze_epochs=2),
        lr_monitor,
    ],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
