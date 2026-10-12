"""TWIST ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B.

Two-view TWIST with the paper's ViT optimizer settings (AdamW, lr 3e-4 per
256 samples, weight decay 0.06, 10 warmup epochs then cosine) and the default
loss weights (sharpness 1.0, diversity 1.0). The momentum encoder and
multi-crop of the paper's ViT recipe are not used, and without the momentum
encoder the paper's ViT sharpness weight of 0.4 collapses to uniform
predictions within 20 epochs (loss = (0.4 - 1) * ln C, probes at chance).

Imagenette is loaded through ``torchvision.datasets.Imagenette`` (the official
fastai archive, 9,469 train / 3,925 val) with the shared two-view transforms.

Environment variables: ``MAX_EPOCHS`` (default 20; use 200 for the RESULTS.md
table), ``NUM_WORKERS`` (8), ``BASE_LR`` (3e-4) and ``SHARPNESS_WEIGHT`` (1.0).
"""

import os
import sys
from pathlib import Path

import torch
import torchvision
from two_view import (
    attach_forward_and_optim,
    standard_callbacks,
    standard_trainer,
    two_view_train_transform,
    val_transform,
)

import stable_pretraining as spt
from stable_pretraining.methods.twist import TWIST

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir  # noqa: E402

WARMUP_EPOCHS = 10


def make_imagenette_data(batch_size: int, num_workers: int) -> spt.data.DataModule:
    data_dir = str(get_data_dir("imagenet10"))
    train_dataset = spt.data.FromTorchDataset(
        torchvision.datasets.Imagenette(root=data_dir, split="train", download=True),
        names=["image", "label"],
        transform=two_view_train_transform(),
    )
    val_dataset = spt.data.FromTorchDataset(
        torchvision.datasets.Imagenette(root=data_dir, split="val", download=True),
        names=["image", "label"],
        transform=val_transform(),
    )
    return spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=num_workers > 0,
            shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        ),
    )


def main():
    batch_size = 256
    max_epochs = int(os.environ.get("MAX_EPOCHS", 20))
    num_workers = int(os.environ.get("NUM_WORKERS", 8))
    base_lr = float(os.environ.get("BASE_LR", 3e-4))
    sharpness_weight = float(os.environ.get("SHARPNESS_WEIGHT", 1.0))

    data = make_imagenette_data(batch_size=batch_size, num_workers=num_workers)
    steps_per_epoch = len(data.train)

    module = TWIST(
        encoder_name="vit_small_patch16_224",
        projector_dims=(4096, 4096, 4096),
        sharpness_weight=sharpness_weight,
        diversity_weight=1.0,
    )
    # ViT-S/16: AdamW as in the paper's ViT recipe (LARS + lr 0.5 is the
    # ResNet50 recipe). Weight decay is held at the paper's starting value.
    attach_forward_and_optim(
        module,
        TWIST,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": base_lr * batch_size / 256,
                "weight_decay": 0.06,
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "peak_step": WARMUP_EPOCHS * steps_per_epoch,
                "total_steps": max_epochs * steps_per_epoch,
            },
            "interval": "step",
        },
    )

    callbacks = standard_callbacks(module, embed_dim=module.embed_dim)
    trainer = standard_trainer(
        callbacks, max_epochs=max_epochs, log_name="twist-vits-inet10"
    )
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
