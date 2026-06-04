"""GPU-augmented two-view pipeline for ImageNet-10 (Imagenette).

Drop-in replacement for :mod:`two_view` that moves all randomized
augmentation off the DataLoader workers and onto the GPU via
:mod:`stable_pretraining.data.gpu_transforms` (kornia + ``torch.compile``).

What lives where
----------------
- **CPU workers** (DataLoader): PIL → uint8 tensor → resize to 256×256 →
  ToImage (uint8→float, no normalize). This is the cheapest CPU work that
  still produces a same-shape batchable tensor. No per-sample random aug.
- **GPU** (via ``Module.on_after_batch_transfer``): RandomResizedCrop,
  HorizontalFlip, ColorJitter, GaussianBlur, RandomGrayscale, optional
  RandomSolarize, and final Normalize. Each view goes through one chain
  call, so per-sample randomness is independent across views.

Multi-view shape
----------------
The CPU dataloader emits a SINGLE 256×256 image per sample (no
MultiViewTransform fan-out on CPU). The GPU pipeline duplicates the
batch tensor across views and applies the augmentation chain N times to
produce N independent views. This is fundamentally cheaper than the CPU
two-view path, which loaded → cropped → augmented twice per sample.
"""

from pathlib import Path
import sys
import types

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics

import stable_pretraining as spt
from stable_pretraining.data import gpu_transforms as gt, transforms

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir  # noqa: E402


def cpu_train_transform():
    """Minimal CPU-side prep: PIL → 256×256 RGB float tensor. No random aug."""
    return transforms.Compose(
        transforms.Resize((256, 256)),
        transforms.ToImage(rgb=True, scale=True),
    )


def val_transform():
    return transforms.Compose(
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(rgb=True, **spt.data.static.ImageNet),
    )


def gpu_two_view_transform_stacked(compile: bool = True):
    """Symmetric-SSL fast path: one chain call on a stacked ``(2B, ...)`` tensor.

    Thin convenience wrapper over :class:`gt.StackedMultiView` with the
    BarlowTwins/SimCLR-style augmentation recipe. Valid for symmetric
    SSL only (Barlow Twins, SimCLR, VICReg, NNCLR). For asymmetric
    methods use :func:`gpu_two_view_transform`.
    """
    imnet_mean = [0.485, 0.456, 0.406]
    imnet_std = [0.229, 0.224, 0.225]
    chain = gt.GPUCompose(
        [
            gt.GPURandomResizedCrop(size=224, scale=(0.08, 1.0)),
            gt.GPURandomHorizontalFlip(p=0.5),
            gt.GPUColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            gt.GPURandomGrayscale(p=0.2),
            gt.GPUGaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.5),
            gt.GPUNormalize(mean=imnet_mean, std=imnet_std),
        ],
        compile=compile,
    )
    return gt.StackedMultiView(chain, n_views=2)


def gpu_two_view_transform(compile: bool = True):
    """Build the GPU augmentation chain that produces two views per sample.

    Returns a callable ``batch_dict -> batch_dict`` that:
      1. Reads ``batch["image"]`` (CPU or GPU tensor, shape ``(B, 3, 256, 256)``),
      2. Produces two augmented views with independent random params,
      3. Writes ``batch["views"]`` as a list of two dicts, each with
         key ``"image"``, matching the schema expected by
         :func:`two_view.make_two_view_forward`.
    """
    # ImageNet normalize constants matching spt.data.static.ImageNet.
    imnet_mean = [0.485, 0.456, 0.406]
    imnet_std = [0.229, 0.224, 0.225]

    # Per-view chains. View 1: standard SimCLR; View 2: SimCLR + solarize.
    view1 = gt.GPUCompose(
        [
            gt.GPURandomResizedCrop(size=224, scale=(0.08, 1.0)),
            gt.GPURandomHorizontalFlip(p=0.5),
            gt.GPUColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            gt.GPURandomGrayscale(p=0.2),
            gt.GPUGaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=1.0),
            gt.GPUNormalize(mean=imnet_mean, std=imnet_std),
        ],
        compile=compile,
    )
    view2 = gt.GPUCompose(
        [
            gt.GPURandomResizedCrop(size=224, scale=(0.08, 1.0)),
            gt.GPURandomHorizontalFlip(p=0.5),
            gt.GPUColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            gt.GPURandomGrayscale(p=0.2),
            gt.GPUGaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.1),
            gt.GPURandomSolarize(p=0.2),
            gt.GPUNormalize(mean=imnet_mean, std=imnet_std),
        ],
        compile=compile,
    )
    return gt.MultiView([view1, view2])


def make_imagenette_data_gpu(
    batch_size: int = 256, num_workers: int = 8, gpu_transform=None
):
    """DataModule with minimal CPU transforms; ``gpu_transform`` is attached to the train dataset."""
    data_dir = str(get_data_dir("imagenet10"))
    train_ds = spt.data.HFDataset(
        "frgfm/imagenette",
        split="train",
        revision="refs/convert/parquet",
        cache_dir=data_dir,
        transform=cpu_train_transform(),
        gpu_transform=gpu_transform,
    )
    val_ds = spt.data.HFDataset(
        "frgfm/imagenette",
        split="validation",
        revision="refs/convert/parquet",
        cache_dir=data_dir,
        transform=val_transform(),
    )
    return spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=num_workers > 0,
            shuffle=True,
            pin_memory=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        ),
    )


def attach_forward_and_optim(module, method_cls, optim):
    """Attach the two-view forward + optim to a Module (GPU aug lives on the dataset)."""
    from two_view import make_two_view_forward

    module.forward = types.MethodType(make_two_view_forward(method_cls), module)
    module.optim = optim
    return module


def build_gpu_transform(compile: bool = True, stacked: bool = True):
    """Construct the GPU transform that the user attaches to the train dataset.

    Args:
        compile: Forward to :class:`GPUCompose` (``torch.compile`` wrapper).
        stacked: True → :func:`gpu_two_view_transform_stacked` (symmetric SSL),
            False → :func:`gpu_two_view_transform` (asymmetric).
    """
    builder = gpu_two_view_transform_stacked if stacked else gpu_two_view_transform
    return builder(compile=compile)


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
            optimizer={"type": "AdamW", "lr": 0.03, "weight_decay": 0.0},
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
