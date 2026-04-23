#!/usr/bin/env python
"""Multi-GPU test script for supervised training with online probing.

This script demonstrates distributed training across 2 GPUs using DDP.
Assumes GPUs are already allocated (e.g., via SLURM).
"""

import hydra
from functools import partial


@hydra.main()
def main(cfg):
    import lightning as pl
    import torch
    import torchmetrics
    import torchvision
    from lightning.pytorch.loggers import WandbLogger
    from transformers import AutoConfig, AutoModelForImageClassification

    import stable_pretraining as ssl
    from stable_pretraining.data import transforms
    from stable_pretraining.data.datasets import Dataset

    pl.seed_everything(0)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        transforms.RGB(),
        transforms.RandomHorizontalFlip(p=0.5),
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
        batch_size=32,  # This is per-GPU batch size
        num_workers=8,  # Reduced workers per GPU to avoid overload
        drop_last=True,
        persistent_workers=True,  # Keep workers alive between epochs
    )

    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((32, 32)),
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
            output["loss"] = torch.nn.functional.cross_entropy(proj, batch["label"])
        return output

    config = AutoConfig.from_pretrained("microsoft/resnet-18")
    backbone = AutoModelForImageClassification.from_config(config)
    projector = torch.nn.Linear(512, 10)
    backbone.classifier[1] = torch.nn.Identity()

    module = ssl.Module(
        backbone=backbone,
        projector=projector,
        forward=forward,
        optim=partial(torch.optim.AdamW, lr=1e-5),
    )

    # Configure callbacks
    linear_probe = ssl.callbacks.OnlineProbe(
        module=module,
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

    wandb_logger = WandbLogger(project="simclr-cifar10_multigpu")

    # Configure multi-GPU trainer
    trainer = pl.Trainer(
        max_epochs=100000,  # Increased epochs for better results
        callbacks=[linear_probe],
        precision="16-mixed",  # Use mixed precision for faster training
        logger=wandb_logger,
        enable_checkpointing=False,
        sync_batchnorm=True,
    )

    manager = ssl.Manager(
        trainer=trainer, module=module, data=data, ckpt_path="restart"
    )
    manager()

    print("Training completed!")


if __name__ == "__main__":
    # run with
    # python train_supervised_multigpu_requeue.py --multirun hydra/launcher=submitit_slurm hydra.launcher.timeout_min=7 hydra.launcher.partition=scavenge hydra.launcher.max_num_timeout=10 hydra.launcher.gpus_per_node=8 hydra.launcher.tasks_per_node=8 ++batch_size=256 ++signal_delay_s=240
    # python train_supervised_multigpu_requeue.py --multirun hydra/launcher=submitit_slurm hydra.launcher.timeout_min=4 hydra.launcher.partition=learnfair hydra.launcher.max_num_timeout=10 hydra.launcher.gpus_per_node=1 hydra.launcher.tasks_per_node=1 ++batch_size=512

    main()
