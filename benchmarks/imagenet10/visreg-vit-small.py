"""VISReg ViT-S/16 on ImageNet-10 (Imagenette).

Multi-view invariance + VISReg sliced-Gaussianity regularizer (the drop-in
alternative to LeJEPA's Epps-Pulley SIGReg). Settings mirror
``lejepa-vit-small.py``; the VISReg-specific knobs are ``lamb`` (convex mixing
weight) and ``num_projections``. Uses 2 global views (224x224) + 6 local views
(96x96) matching the official multi-view augmentation strategy.

Epoch count is read from the ``MAX_EPOCHS`` env var (default 200, matching the
README benchmark table).
"""

import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics

# Multi-worker DataLoaders pass sample tensors between processes via file
# descriptors by default; with the multi-view (8 crops/sample) pipeline and
# many workers this overflows the open-fd limit on some nodes, raising
# "RuntimeError: received 0 items of ancdata". The file_system strategy passes
# shared-memory files by name instead, avoiding the fd limit.
torch.multiprocessing.set_sharing_strategy("file_system")

import stable_pretraining as spt  # noqa: E402
from stable_pretraining.data import transforms  # noqa: E402
from stable_pretraining.methods.visreg import VISReg, VISRegOutput  # noqa: E402


def _photometric_transforms() -> list:
    return [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.5),
        transforms.RandomSolarize(threshold=128, p=0.2),
    ]


def _global_transform():
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((224, 224), scale=(0.3, 1.0)),
        *_photometric_transforms(),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


def _local_transform():
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.3)),
        *_photometric_transforms(),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


def visreg_forward(self, batch, stage):
    """VISReg forward: multi-view invariance + sliced-Gaussianity regularizer.

    Batch format:
        - Training: dict of named views (``"global_0"``, ``"local_2"``, etc.)
        - Eval: single dict with ``"image"`` key

    Returns:
        Dictionary with ``"loss"``, ``"embedding"``, and ``"label"``.
    """
    out = {}

    images = batch.get("image")
    if stage == "fit":
        global_views = [
            batch[key]["image"] for key in batch if key.startswith("global")
        ]
        local_views = [batch[key]["image"] for key in batch if key.startswith("local")]
        labels = next(
            batch[key]["label"]
            for key in batch
            if key.startswith("global") or key.startswith("local")
        )

        output: VISRegOutput = self.model.forward(
            global_views=global_views, local_views=local_views, images=images
        )
        out["label"] = labels.repeat(len(global_views))
    else:
        output: VISRegOutput = self.model.forward(images=images)
        out["label"] = batch["label"].long()

    out["loss"] = output.loss
    out["embedding"] = output.embedding

    self.log(
        f"{stage}/visreg",
        output.visreg_loss,
        on_step=True,
        on_epoch=True,
        sync_dist=True,
    )
    self.log(
        f"{stage}/inv", output.inv_loss, on_step=True, on_epoch=True, sync_dist=True
    )
    self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True)
    return out


def main():
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import imagenette

    # Disable spt.Manager's automatic requeue checkpoint (run_dir/checkpoints/
    # last.ckpt, saved every epoch). Combined with enable_checkpointing=False on
    # the Trainer, this run saves NO checkpoints at all.
    spt.set(requeue_checkpoint=False)

    num_gpus = 1
    batch_size = 128
    num_workers = 16
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 200))
    global_views = 2
    all_views = 8

    # 2 global views (blur p=1.0, p=0.1) + 6 local views
    train_transform = transforms.MultiViewTransform(
        {
            **{f"global_{i}": _global_transform() for i in range(global_views)},
            **{
                f"local_{i}": _local_transform()
                for i in range(all_views - global_views)
            },
        }
    )

    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(**spt.data.static.ImageNet),
    )

    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=imagenette(
                "train",
                transform=train_transform,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=num_workers > 0,
            shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=imagenette(
                "validation",
                transform=val_transform,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        ),
    )

    model = VISReg(
        encoder_name="vit_small_patch16_224",
        lamb=0.6,
        num_projections=1024,
    )

    module = spt.Module(
        model=model,
        forward=visreg_forward,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": (lr := 1e-3),
                "weight_decay": 0.05,
                "betas": (0.9, 0.999),
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "peak_step": 10 / max_epochs,
                "start_factor": 0.01,
                "end_lr": lr / 1000,
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
                probe=nn.Linear(model.embed_dim, 10),
                loss=nn.CrossEntropyLoss(),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(10),
                    "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
                },
                optimizer={"type": "AdamW", "lr": 0.03, "weight_decay": 1e-6},
            ),
            spt.callbacks.OnlineKNN(
                name="knn_probe",
                input="embedding",
                target="label",
                queue_length=10000,
                metrics={"top1": torchmetrics.classification.MulticlassAccuracy(10)},
                input_dim=model.embed_dim,
                k=20,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=pl.pytorch.loggers.CSVLogger(
            save_dir=str(Path(__file__).parent / "logs"),
            name="visreg-vits-inet10",
        ),
        precision="16-mixed",
        devices=num_gpus,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true" if num_gpus > 1 else "auto",
        enable_checkpointing=False,
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()


if __name__ == "__main__":
    main()
