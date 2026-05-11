"""Shared helpers for multi-crop methods (DINO/iBOT/DINOv2)."""

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


def _photometric():
    return [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.5),
        transforms.RandomSolarize(threshold=128, p=0.2),
    ]


def _global_t():
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((224, 224), scale=(0.4, 1.0)),
        *_photometric(),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


def _local_t():
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.4)),
        *_photometric(),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


def val_transform():
    return transforms.Compose(
        transforms.RGB(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


def make_imagenette_data(
    batch_size: int = 128, num_workers: int = 8, n_global: int = 2, n_local: int = 6
):
    data_dir = str(get_data_dir("imagenet10"))
    train_t = transforms.MultiViewTransform(
        {
            **{f"global_{i}": _global_t() for i in range(n_global)},
            **{f"local_{i}": _local_t() for i in range(n_local)},
        }
    )
    return spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "frgfm/imagenette",
                split="train",
                revision="refs/convert/parquet",
                cache_dir=data_dir,
                transform=train_t,
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


def make_multicrop_forward(method_cls, use_local: bool = True):
    """Adapter for multi-crop methods.

    Expects the method's ``forward(global_views, local_views, images)`` to
    return a ``ModelOutput`` with ``loss`` (training) and ``embedding``.
    """

    def fwd(self, batch, stage):
        # Eval / single-image path
        if "image" in batch:
            output = method_cls.forward(self, images=batch["image"])
            out = {"embedding": output.embedding}
            if "label" in batch:
                out["label"] = batch["label"].long()
            return out

        # Training: dict of named views
        global_imgs = [batch[k]["image"] for k in batch if k.startswith("global")]
        local_imgs = (
            [batch[k]["image"] for k in batch if k.startswith("local")]
            if use_local
            else []
        )
        first_global = next(k for k in batch if k.startswith("global"))
        labels = batch[first_global].get("label")

        if use_local and "local_views" in method_cls.forward.__code__.co_varnames:
            output = method_cls.forward(
                self, global_views=global_imgs, local_views=local_imgs
            )
        else:
            output = method_cls.forward(self, global_views=global_imgs)

        # Embedding shape varies: DINO returns [n_global * B, D]; iBOT/DINOv2 returns [n_global * B, D] too.
        embedding = output.embedding.detach()
        n_emb = embedding.shape[0]
        out = {"loss": output.loss, "embedding": embedding}
        if labels is not None:
            B = labels.shape[0]
            n_repeat = n_emb // B
            out["label"] = labels.repeat(n_repeat).long()
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
                "top5": torchmetrics.classification.MulticlassAccuracy(
                    num_classes, top_k=5
                ),
            },
            optimizer={"type": "AdamW", "lr": 0.03, "weight_decay": 1e-6},
        ),
        spt.callbacks.OnlineKNN(
            name="knn_probe",
            input="embedding",
            target="label",
            queue_length=10000,
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(num_classes)
            },
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


def attach_forward_and_optim(module, method_cls, optim, use_local: bool = True):
    module.forward = types.MethodType(
        make_multicrop_forward(method_cls, use_local=use_local), module
    )
    module.optim = optim
    return module
