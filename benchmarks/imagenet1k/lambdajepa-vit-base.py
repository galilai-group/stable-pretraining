"""λ-JEPA ViT-Base on ImageNet-1k.

Multi-view invariance to an EMA target + SACReg on the projector and
backbone. Uses 6 views (224x224) and a global batch of 128 matching the
paper's ImageNet-1k ViT-B/16 setting (the ``LambdaJEPA`` defaults).
"""

import os
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.lambda_jepa import LambdaJEPA, LambdaJEPAOutput


def _view_transform():
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
        transforms.ColorJitter(
            brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0), p=0.5),
        transforms.RandomSolarize(threshold=128, p=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


def lambdajepa_forward(self, batch, stage):
    """λ-JEPA forward: multi-view invariance + SACReg.

    Expects ``self`` to have attributes:
        - ``model``: :class:`LambdaJEPA` module

    Batch format:
        - Training: dict of named views (``"view_0"``, ``"view_1"``, etc.)
        - Eval: single dict with ``"image"`` key

    Args:
        self: Module instance (automatically bound).
        batch: Named view dict or single-image dict.
        stage: Training stage ('train', 'val', or 'test').

    Returns:
        Dictionary with ``"loss"``, ``"embedding"``, and optionally ``"label"``.
    """
    out = {}
    if stage == "fit":
        views = [batch[key]["image"] for key in batch if key.startswith("view")]
        labels = next(batch[key]["label"] for key in batch if key.startswith("view"))
        output: LambdaJEPAOutput = self.model.forward(global_views=views)
        out["label"] = labels.repeat(len(views))
    else:
        output: LambdaJEPAOutput = self.model.forward(images=batch["image"])
        out["label"] = batch["label"].long()

    out["loss"] = output.loss
    out["embedding"] = output.embedding

    self.log(
        f"{stage}/inv", output.inv_loss, on_step=True, on_epoch=True, sync_dist=True
    )
    self.log(
        f"{stage}/sacreg_z",
        output.sacreg_z_loss,
        on_step=True,
        on_epoch=True,
        sync_dist=True,
    )
    self.log(
        f"{stage}/sacreg_h",
        output.sacreg_h_loss,
        on_step=True,
        on_epoch=True,
        sync_dist=True,
    )
    self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True)
    return out


def main():
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_data_dir

    seed = int(os.environ.get("SEED", 0))
    max_epochs = int(os.environ.get("MAX_EPOCHS", 100))
    num_gpus = torch.cuda.device_count() or 1
    assert 128 % num_gpus == 0, (
        "the global batch of 128 must split evenly over the GPUs"
    )
    batch_size = 128 // num_gpus
    num_workers = 16
    num_views = 6
    num_classes = 1000

    data_dir = str(get_data_dir("imagenet1k"))

    train_transform = transforms.MultiViewTransform(
        {f"view_{i}": _view_transform() for i in range(num_views)}
    )
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(**spt.data.static.ImageNet),
    )

    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "ILSVRC/imagenet-1k",
                split="train",
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
                "ILSVRC/imagenet-1k",
                split="validation",
                cache_dir=data_dir,
                transform=val_transform,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        ),
    )

    # λ-JEPA paper recipe (ImageNet-1k, ViT-B/16) = the LambdaJEPA defaults: loss
    # weights 22.81 / 222.26 / 4.054, slices of 128 / 256, queues of 3 / 7 steps,
    # invariance to an EMA target.
    model = LambdaJEPA()

    module = spt.Module(
        model=model,
        forward=lambdajepa_forward,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": (lr := 1e-3),
                "weight_decay": 0.05,
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "peak_step": 10 / max_epochs,
                "start_factor": 0.01,
                "end_lr": lr / 100,
                "total_steps": (len(data.train) // num_gpus) * max_epochs,
            },
            "interval": "step",
        },
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        callbacks=[
            spt.callbacks.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=nn.Linear(model.embed_dim, num_classes),
                loss=nn.CrossEntropyLoss(),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
                    "top5": torchmetrics.classification.MulticlassAccuracy(
                        num_classes, top_k=5
                    ),
                },
                optimizer={"type": "AdamW", "lr": 1e-3, "weight_decay": 1e-6},
            ),
            spt.callbacks.OnlineKNN(
                name="knn_probe",
                input="embedding",
                target="label",
                queue_length=10000,
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(num_classes)
                },
                input_dim=model.embed_dim,
                k=20,
            ),
            spt.callbacks.RankMe(
                name="rankme",
                target="embedding",
                queue_length=1000,
                target_shape=model.embed_dim,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        precision="bf16-mixed",
        devices=num_gpus,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true" if num_gpus > 1 else "auto",
        sync_batchnorm=num_gpus > 1,
    )

    spt.Manager(trainer=trainer, module=module, data=data, seed=seed)()


if __name__ == "__main__":
    main()
