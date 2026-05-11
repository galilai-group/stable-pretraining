"""Shared helpers for single-view masked-image methods (SimMIM/Data2Vec/iGPT/BEiT/MAE)."""

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


def masked_train_transform():
    """Light augmentation for masked / autoregressive methods (no view duplication)."""
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


def val_transform():
    return transforms.Compose(
        transforms.RGB(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


def make_imagenette_data(batch_size: int = 128, num_workers: int = 8):
    data_dir = str(get_data_dir("imagenet10"))
    return spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "frgfm/imagenette",
                split="train",
                revision="refs/convert/parquet",
                cache_dir=data_dir,
                transform=masked_train_transform(),
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


def make_masked_forward(method_cls):
    """For methods whose forward takes a single image tensor and returns
    ``ModelOutput(loss, embedding)``."""

    def fwd(self, batch, stage):
        out = method_cls.forward(self, batch["image"])
        result = {"embedding": out.embedding}
        if self.training:
            result["loss"] = out.loss
            self.log(
                f"{stage}/loss",
                out.loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        if "label" in batch:
            result["label"] = batch["label"].long()
        return result

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
            optimizer={"type": "AdamW", "lr": 0.025, "weight_decay": 0.0},
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


def attach_forward_and_optim(module, method_cls, optim, extra_callbacks=()):
    module.forward = types.MethodType(make_masked_forward(method_cls), module)
    module.optim = optim
    return module
