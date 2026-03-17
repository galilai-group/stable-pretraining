#!/usr/bin/env python
"""Multi-GPU test script for SimCLR training with online probing.

This script demonstrates distributed training across 2 GPUs using DDP.
Assumes GPUs are already allocated (e.g., via SLURM).
"""

import lightning as pl
import torch
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoConfig, AutoModelForImageClassification

import stable_pretraining as ssl
from stable_pretraining.data import transforms
from stable_pretraining.data.datasets import Dataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose(
    transforms.RGB(),
    transforms.RandomResizedCrop((32, 32)),  # CIFAR-10 is 32x32
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=(5, 5), p=1.0),
    transforms.ToImage(mean=mean, std=std),
)

# Use torchvision CIFAR-10
cifar_train = torchvision.datasets.CIFAR10(
    root="/tmp/cifar10", train=True, download=True
)

# Create a custom wrapper that adds sample_idx


class IndexedDataset(Dataset):
    """Custom dataset wrapper that adds sample_idx to each sample."""

    def __init__(self, dataset, transform=None):
        super().__init__(transform)
        self.dataset = dataset

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        sample = {"image": image, "label": label, "sample_idx": idx}
        return self.process_sample(sample)

    def __len__(self):
        return len(self.dataset)


train_dataset = IndexedDataset(cifar_train, transform=train_transform)

# Increase batch size for multi-GPU training (64 per GPU * 2 GPUs = 128 total)
train = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=ssl.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
    batch_size=256,  # This is per-GPU batch size
    num_workers=8,  # Reduced workers per GPU to avoid overload
    drop_last=True,
    persistent_workers=True,  # Keep workers alive between epochs
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.CenterCrop((32, 32)),
    transforms.ToImage(mean=mean, std=std),
)

# Use torchvision CIFAR-10 for validation
cifar_val = torchvision.datasets.CIFAR10(
    root="/tmp/cifar10", train=False, download=True
)
val_dataset = IndexedDataset(cifar_val, transform=val_transform)
val = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=128,  # Per-GPU batch size
    num_workers=4,
    persistent_workers=True,
)
data = ssl.data.DataModule(train=train, val=val)


def forward(self, batch, stage):
    output = {}
    output["embedding"] = self.backbone(batch["image"])["logits"]
    if self.training:
        proj = self.projector(output["embedding"])
        views = ssl.data.fold_views(proj, batch["sample_idx"])
        output["loss"] = self.simclr_loss(views[0], views[1])
    return output


config = AutoConfig.from_pretrained("microsoft/resnet-18")
backbone = AutoModelForImageClassification.from_config(config)
projector = torch.nn.Linear(512, 128)
backbone.classifier[1] = torch.nn.Identity()

module = ssl.Module(
    backbone=backbone,
    projector=projector,
    forward=forward,
    simclr_loss=ssl.losses.NTXEntLoss(temperature=0.5),
    accumulate_grad_batches=2,  # Pass as module parameter
)

# Configure callbacks
linear_probe = ssl.callbacks.OnlineProbe(
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

knn_probe = ssl.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
    input_dim=512,
    k=10,
)

wandb_logger = WandbLogger(project="simclr-cifar10_multigpu")

# Configure multi-GPU trainer
trainer = pl.Trainer(
    max_epochs=10,  # Increased epochs for better results
    num_sanity_val_steps=0,  # Skip sanity check as queues need to be filled first
    callbacks=[linear_probe, knn_probe],
    precision="16-mixed",  # Use mixed precision for faster training
    logger=wandb_logger,
    enable_checkpointing=False,
    # Multi-GPU settings
    accelerator="gpu",
    devices=2,  # Use 2 GPUs
    strategy="ddp",  # Distributed Data Parallel
    sync_batchnorm=True,  # Synchronize batch norm across GPUs
    # Performance optimizations
    log_every_n_steps=50,
    val_check_interval=0.5,  # Validate twice per epoch
)

if __name__ == "__main__":
    # Important: Guard the main execution for multiprocessing
    print("Starting multi-GPU training on 2 devices...")
    print(f"Total effective batch size: {64 * 2} (batch_size * devices)")

    # Initialize model with dummy batch to avoid uninitialized buffer errors
    print("Initializing model with dummy batch...")
    # Create dummy batch with 2 views per sample (for SimCLR)
    dummy_batch = {
        "image": torch.randn(8, 3, 32, 32),  # 4 samples * 2 views
        "label": torch.randint(0, 10, (8,)),
        "sample_idx": torch.tensor(
            [0, 0, 1, 1, 2, 2, 3, 3]
        ),  # Repeated indices for views
    }
    # Set module to training mode for proper initialization
    module.train()
    with torch.no_grad():
        _ = module(dummy_batch, stage="fit")
    print("Model initialized successfully!")

    manager = ssl.Manager(trainer=trainer, module=module, data=data)
    manager()

    print("Training completed!")
