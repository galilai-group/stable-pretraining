"""DIET ResNet-18 on CIFAR-100.

Datum Index as Target (DIET) predicts the stable dataset index during
single-view training. Semantic labels are used only by the online probes.
Set ``MAX_EPOCHS`` and ``NUM_GPUS`` to override the benchmark defaults.
"""

import os
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods import DIET


def _transform(train: bool):
    ops = [transforms.RGB()]
    if train:
        ops.extend(
            [
                transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.2,
                    hue=0.1,
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
    else:
        ops.append(transforms.Resize((32, 32)))
    ops.append(
        transforms.ToImage(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        )
    )
    return transforms.Compose(*ops)


def diet_forward(self, batch, stage):
    """Run DIET's index objective and expose semantic labels to probes."""
    output = self.model(batch, stage=stage)
    return {
        "loss": output["loss"],
        "embedding": output["embedding"],
        "label": batch.get("label"),
    }


def main():
    max_epochs = int(os.environ.get("MAX_EPOCHS", 1000))
    num_gpus = int(os.environ.get("NUM_GPUS", torch.cuda.device_count() or 1))
    batch_size = 256 // num_gpus
    num_workers = 8

    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.FromTorchDataset(
                torchvision.datasets.CIFAR100(
                    root=Path("~/data").expanduser(), train=True, download=True
                ),
                names=["image", "label"],
                transform=_transform(train=True),
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=num_workers > 0,
            shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=spt.data.FromTorchDataset(
                torchvision.datasets.CIFAR100(
                    root=Path("~/data").expanduser(), train=False, download=True
                ),
                names=["image", "label"],
                transform=_transform(train=False),
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        ),
    )

    model = DIET(
        encoder_name="resnet18",
        num_samples=50_000,
        label_smoothing=0.8,
        low_resolution=True,
        image_size=32,
    )
    module = spt.Module(
        model=model,
        forward=diet_forward,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": 1e-3,
                "weight_decay": 0.05,
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "peak_step": 10 / max_epochs,
                "start_factor": 0.01,
                "end_lr": 1e-5,
                "total_steps": (len(data.train) // num_gpus) * max_epochs,
            },
            "interval": "step",
        },
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        callbacks=[
            spt.callbacks.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=nn.Linear(model.embed_dim, 100),
                loss=nn.CrossEntropyLoss(),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(100),
                    "top5": torchmetrics.classification.MulticlassAccuracy(
                        100, top_k=5
                    ),
                },
                optimizer={"type": "AdamW", "lr": 1e-3, "weight_decay": 1e-6},
            ),
            spt.callbacks.OnlineKNN(
                name="knn_probe",
                input="embedding",
                target="label",
                queue_length=10_000,
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(100)
                },
                input_dim=model.embed_dim,
                k=20,
            ),
            spt.callbacks.RankMe(
                name="rankme",
                target="embedding",
                queue_length=1_000,
                target_shape=model.embed_dim,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        precision="32-true",
        devices=num_gpus,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp_find_unused_parameters_true" if num_gpus > 1 else "auto",
        sync_batchnorm=num_gpus > 1,
    )
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
