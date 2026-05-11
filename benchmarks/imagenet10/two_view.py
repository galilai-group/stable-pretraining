"""Shared helpers for two-view (SimCLR/VICReg/Barlow/BYOL) ImageNet-10 recipes.

Centralises the dataloader, augmentations, callbacks, and forward dispatcher so
each method script is a thin configuration file. Designed for short
verification runs (20 epochs, 1 GPU, no W&B).
"""

from pathlib import Path
import sys
import types

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics

import stable_pretraining as spt
from stable_pretraining.data import transforms

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir  # noqa: E402


def two_view_train_transform():
    """Asymmetric SimCLR-style two-view augmentation."""
    return transforms.MultiViewTransform(
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


def val_transform():
    return transforms.Compose(
        transforms.RGB(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


def make_imagenette_data(batch_size: int = 256, num_workers: int = 8):
    data_dir = str(get_data_dir("imagenet10"))
    return spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "frgfm/imagenette",
                split="train",
                revision="refs/convert/parquet",
                cache_dir=data_dir,
                transform=two_view_train_transform(),
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
                transform=val_transform(),
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        ),
    )


def make_two_view_forward(method_cls):
    """Build a Lightning forward function for any two-view method.

    The method class must implement ``forward(view1, view2=None) -> ModelOutput``
    with attributes ``loss`` and ``embedding``. The wrapper handles MultiViewTransform
    output (dict with ``"views"`` list or named view dict) and single-view eval.
    """

    def fwd(self, batch, stage):
        if "image" in batch:  # eval / single view
            output = method_cls.forward(self, batch["image"])
            out = {"embedding": output.embedding}
            if "label" in batch:
                out["label"] = batch["label"].long()
            return out

        if "views" in batch:
            views = batch["views"]
        else:
            views = list(batch.values())
        if len(views) != 2:
            raise ValueError(f"{method_cls.__name__} expects 2 views, got {len(views)}")
        v1, v2 = views[0]["image"], views[1]["image"]
        output = method_cls.forward(self, v1, v2)
        embedding = output.embedding.detach()
        out = {"loss": output.loss, "embedding": embedding}
        if "label" in views[0]:
            B = views[0]["label"].shape[0]
            n_repeat = max(embedding.shape[0] // B, 1)
            label_views = [views[i]["label"] for i in range(n_repeat)]
            out["label"] = torch.cat(label_views, dim=0).long()
        self.log(
            f"{stage}/loss",
            output.loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return out

    return fwd


def standard_callbacks(module, embed_dim, num_classes=10):
    return [
        spt.callbacks.OnlineProbe(
            module,
            name="linear_probe",
            input="embedding",
            target="label",
            probe=nn.Linear(embed_dim, num_classes),
            loss=nn.CrossEntropyLoss(),
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
                "top5": torchmetrics.classification.MulticlassAccuracy(num_classes, top_k=5),
            },
            optimizer={"type": "AdamW", "lr": 0.03, "weight_decay": 0.0},
        ),
        spt.callbacks.OnlineKNN(
            name="knn_probe",
            input="embedding",
            target="label",
            queue_length=10000,
            metrics={"top1": torchmetrics.classification.MulticlassAccuracy(num_classes)},
            input_dim=embed_dim,
            k=20,
        ),
        pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
    ]


def standard_trainer(callbacks, max_epochs, log_name):
    return pl.Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=pl.pytorch.loggers.CSVLogger(
            save_dir=str(Path(__file__).parent / "logs"),
            name=log_name,
        ),
        precision="16-mixed",
        enable_checkpointing=False,
        devices=torch.cuda.device_count() or 1,
        accelerator="gpu",
    )


def attach_forward_and_optim(module, method_cls, optim):
    module.forward = types.MethodType(make_two_view_forward(method_cls), module)
    module.optim = optim
    return module
