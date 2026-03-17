"""SwAV training on CIFAR10."""

import lightning as pl
import torch
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger
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
MAX_EPOCHS = 50
WARMUP_EPOCHS = 10
END_LR = 6e-4 if BATCH_SIZE <= 256 else 4.8e-3  # Parameters used in the original paper
LR = 0.6 if BATCH_SIZE <= 256 else 4.8  # Parameters used in the original paper
# Queue parameters from the paper
USE_QUEUE = True if BATCH_SIZE < 256 else False
QUEUE_LENGTH = 3840
START_QUEUE_AT_EPOCH = 15
PROJECTION_DIM = 128  # The output of the projector

swav_transform = transforms.MultiViewTransform(
    {
        "global_1": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
        "global_2": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, p=0.5),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
        "local_1": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(16, scale=(0.05, 0.2)),  # Small crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, p=0.5),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
        "local_2": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(16, scale=(0.05, 0.2)),  # Small crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, p=0.5),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
        "local_3": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(16, scale=(0.05, 0.2)),  # Small crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, p=0.5),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
        "local_4": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(16, scale=(0.05, 0.2)),  # Small crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, p=0.5),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
        "local_5": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(16, scale=(0.05, 0.2)),  # Small crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, p=0.5),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
        "local_6": transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(16, scale=(0.05, 0.2)),  # Small crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, p=0.5),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
    }
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
    transform=swav_transform,
)
val_dataset = spt.data.FromTorchDataset(
    cifar_val,
    names=["image", "label"],
    transform=val_transform,
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    # Updated to 8 views to match the multi-crop transform
    sampler=spt.data.sampler.RepeatedRandomSampler(train_dataset, n_views=8),
    batch_size=256,
    num_workers=8,
    drop_last=True,
)

val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=256,
    num_workers=10,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

backbone = spt.backbone.from_torchvision(
    "resnet18",
    low_resolution=True,
)
backbone.fc = torch.nn.Identity()

projector = nn.Sequential(
    nn.Linear(512, 2048),
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
    projection_dim=PROJECTION_DIM,
    swav_loss=spt.losses.joint_embedding.SwAVLoss(
        temperature=0.1,
        sinkhorn_iterations=3,
    ),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": LR,
            "weight_decay": 1e-6,
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
)

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="cifar10-swav",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    num_sanity_val_steps=0,
    callbacks=[knn_probe, linear_probe, FreezePrototypesCallback(freeze_epochs=2)],
    precision="16-mixed",  # Can be unstable
    logger=wandb_logger,
    enable_checkpointing=False,
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
