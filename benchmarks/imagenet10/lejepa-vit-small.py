"""LeJEPA ViT-Base on ImageNet-10 (Imagenette).

Multi-view invariance + Epps-Pulley goodness-of-fit (SIGReg).
Uses 2 global views (224x224) + 6 local views (96x96) matching
the official LeJEPA augmentation strategy.
"""

import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.lejepa import LeJEPA, LeJEPAOutput


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


def lejepa_forward(self, batch, stage):
    """LeJEPA forward: multi-view invariance + Epps-Pulley goodness-of-fit (SIGReg).

    Expects ``self`` to have attributes:
        - ``backbone``: Feature extraction network
        - ``projector``: Projection head
        - ``sigreg``: :class:`SlicedEppsPulley` module
        - ``lamb``: SIGReg weight λ

    Batch format:
        - Training: dict of named views (``"global_0"``, ``"local_2"``, etc.)
        - Eval: single dict with ``"image"`` key

    Args:
        self: Module instance (automatically bound).
        batch: Named view dict or single-image dict.
        stage: Training stage ('train', 'val', or 'test').

    Returns:
        Dictionary with ``"loss"``, ``"embedding"``, and optionally ``"label"``.
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

        output: LeJEPAOutput = self.model.forward(
            global_views=global_views, local_views=local_views, images=images
        )
        out["label"] = labels.repeat(len(global_views))
    else:
        output: LeJEPAOutput = self.model.forward(images=images)
        out["label"] = batch["label"].long()

    out["loss"] = output.loss
    out["embedding"] = output.embedding

    self.log(
        f"{stage}/sigreg",
        output.sigreg_loss,
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
    from utils import get_data_dir

    num_gpus = 1
    batch_size = 128
    num_workers = 16
    max_epochs = 600
    global_views = 2
    all_views = 8

    data_dir = str(get_data_dir("imagenet10"))

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
            dataset=spt.data.HFDataset(
                "frgfm/imagenette",
                split="train",
                revision="refs/convert/parquet",
                cache_dir=data_dir,
                transform=train_transform,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=num_workers > 0,
            shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "frgfm/imagenette",
                split="validation",
                revision="refs/convert/parquet",
                cache_dir=data_dir,
                transform=val_transform,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        ),
    )

    model = LeJEPA(
        encoder_name="vit_small_patch16_224",
        lamb=0.02,
        n_slices=1024,
        n_points=17,
    )

    module = spt.Module(
        model=model,
        forward=lejepa_forward,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": (lr := 4e-4),
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
            spt.callbacks.RankMe(
                name="rankme",
                target="embedding",
                queue_length=1000,
                target_shape=model.embed_dim,
            ),
            pl.pytorch.callbacks.ModelCheckpoint(
                dirpath=str(Path(__file__).parent / "checkpoints" / "lejepa-vits"),
                filename="lejepa-vits-{epoch:03d}",
                save_top_k=-1,
                every_n_epochs=300,
                save_last=True,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=pl.pytorch.loggers.WandbLogger(
            entity="stable-ssl",
            project="imagenet10-methods",
            name="lejepa-vits-inet10",
            log_model=False,
        ),
        precision="16-mixed",
        devices=num_gpus,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true" if num_gpus > 1 else "auto",
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()


if __name__ == "__main__":
    main()
