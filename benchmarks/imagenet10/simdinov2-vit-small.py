"""SimDINOv2 ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B."""

import os
from pathlib import Path
import sys

import torch
import torchvision.datasets as dset

sys.path.append(str(Path(__file__).parent))
from multicrop import (
    _global_t,
    _local_t,
    attach_forward_and_optim,
    standard_callbacks,
    standard_trainer,
    val_transform,
)
from utils import get_data_dir

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.simdinov2 import SimDINOv2


def make_simdinov2_data(
    batch_size: int = 128, num_workers: int = 8, n_global: int = 2, n_local: int = 6
):
    data_dir = str(get_data_dir("imagenet10"))
    train_t = transforms.MultiViewTransform(
        {
            **{f"global_{i}": _global_t() for i in range(n_global)},
            **{f"local_{i}": _local_t() for i in range(n_local)},
        }
    )
    train_ds = spt.data.FromTorchDataset(
        dset.Imagenette(data_dir, split="train", size="320px", download=True),
        names=["image", "label"],
        transform=train_t,
        add_sample_idx=False,
    )
    val_ds = spt.data.FromTorchDataset(
        dset.Imagenette(data_dir, split="val", size="320px", download=True),
        names=["image", "label"],
        transform=val_transform(),
        add_sample_idx=False,
    )

    return spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=num_workers > 0,
            shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        ),
    )


def main():
    batch_size = int(os.environ.get("BATCH_SIZE", 32))
    max_epochs = int(os.environ.get("MAX_EPOCHS", 20))
    num_workers = int(os.environ.get("NUM_WORKERS", 4))

    data = make_simdinov2_data(
        batch_size=batch_size, num_workers=num_workers, n_global=2, n_local=6
    )

    module = SimDINOv2(
        encoder_name="vit_small_patch16_224",
        projector_hidden_dim=2048,
        projector_bottleneck_dim=256,
        cls_loss_weight=1.0,
        patch_loss_weight=1.0,
        mask_ratio_min_max=(0.1, 0.5),
        mask_sample_probability=0.5,
        ema_decay_start=0.996,
        ema_decay_end=1.0,
    )
    attach_forward_and_optim(
        module,
        SimDINOv2,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": 5e-4,
                "weight_decay": 0.05,
                "betas": (0.9, 0.95),
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        },
        use_local=True,
    )

    callbacks = [
        spt.callbacks.TeacherStudentCallback(
            update_frequency=1, update_after_backward=True
        ),
        *standard_callbacks(module, embed_dim=module.embed_dim),
    ]
    trainer = standard_trainer(
        callbacks, max_epochs=max_epochs, log_name="simdinov2-vits-inet10"
    )
    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
