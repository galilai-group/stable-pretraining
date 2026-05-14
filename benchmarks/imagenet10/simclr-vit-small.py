"""SimCLR ViT-S/16 on ImageNet-10 (Imagenette).

Short verification recipe: 20 epochs, batch 256, single GPU, no W&B.
Two-view contrastive learning with NT-Xent loss; checks that the SimCLR
class + online linear/KNN probes converge on a small dataset.
"""

import sys
import types
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.simclr import SimCLR


def main():
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_data_dir

    num_gpus = torch.cuda.device_count() or 1
    batch_size = 256
    num_workers = 8
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))

    train_transform = transforms.MultiViewTransform(
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

    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(**spt.data.static.ImageNet),
    )

    data_dir = str(get_data_dir("imagenet10"))

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

    module = SimCLR(
        encoder_name="vit_small_patch16_224",
        projector_dims=(2048, 2048, 256),
        temperature=0.2,
    )

    def simclr_forward(self, batch, stage):
        # Eval / single-view: batch has top-level "image"
        if "image" in batch:
            output = SimCLR.forward(self, batch["image"])
            out = {"embedding": output.embedding}
            if "label" in batch:
                out["label"] = batch["label"].long()
            return out

        # Training: MultiViewTransform yields {"views": [v1_dict, v2_dict]} for
        # the list form, or {"name1": v1_dict, "name2": v2_dict} for the dict form.
        if "views" in batch:
            views = batch["views"]
        else:
            views = list(batch.values())
        if len(views) != 2:
            raise ValueError(f"SimCLR expects 2 views, got {len(views)}")
        v1, v2 = views[0]["image"], views[1]["image"]
        output = SimCLR.forward(self, v1, v2)
        out = {"loss": output.loss, "embedding": output.embedding.detach()}
        if "label" in views[0]:
            out["label"] = torch.cat(
                [views[0]["label"], views[1]["label"]], dim=0
            ).long()
        self.log(
            f"{stage}/loss",
            output.loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return out

    module.forward = types.MethodType(simclr_forward, module)
    module.optim = {
        "optimizer": {
            "type": "LARS",
            "lr": 0.3 * batch_size / 256,
            "weight_decay": 1e-4,
            "clip_lr": True,
            "eta": 0.02,
            "exclude_bias_n_norm": True,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
    }

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        callbacks=[
            spt.callbacks.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=nn.Linear(module.embed_dim, 10),
                loss=nn.CrossEntropyLoss(),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(10),
                    "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
                },
                optimizer={"type": "AdamW", "lr": 0.03, "weight_decay": 0.0},
            ),
            spt.callbacks.OnlineKNN(
                name="knn_probe",
                input="embedding",
                target="label",
                queue_length=10000,
                metrics={"top1": torchmetrics.classification.MulticlassAccuracy(10)},
                input_dim=module.embed_dim,
                k=20,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=pl.pytorch.loggers.CSVLogger(
            save_dir=str(Path(__file__).parent / "logs"),
            name="simclr-vits-inet10",
        ),
        precision="16-mixed",
        enable_checkpointing=False,
        devices=num_gpus,
        accelerator="gpu",
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()


if __name__ == "__main__":
    main()
