"""MAE ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B.

MAE linear probes are notoriously slow to converge — the 50% bar is
unlikely without much longer schedules and a fine-tuning probe. This
script is for plumbing/regression checks, not SOTA.
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
from stable_pretraining.methods.mae import MAE


def main():
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_data_dir

    num_gpus = torch.cuda.device_count() or 1
    batch_size = 128
    max_epochs = int(__import__("os").environ.get("MAX_EPOCHS", 20))

    def mae_forward(self, batch, stage):
        output = MAE.forward(self, batch["image"])
        with torch.no_grad():
            features = self.encoder.forward_features(batch["image"])
        self.log(
            f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True
        )
        return {
            "loss": output.loss,
            "embedding": features[:, 1:].mean(dim=1).detach(),
            **({"label": batch["label"].long()} if "label" in batch else {}),
        }

    data_dir = str(get_data_dir("imagenet10"))
    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "frgfm/imagenette",
                split="train",
                revision="refs/convert/parquet",
                cache_dir=data_dir,
                transform=transforms.Compose(
                    transforms.RGB(),
                    transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToImage(**spt.data.static.ImageNet),
                ),
            ),
            batch_size=batch_size,
            num_workers=8,
            drop_last=True,
            persistent_workers=True,
            shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "frgfm/imagenette",
                split="validation",
                revision="refs/convert/parquet",
                cache_dir=data_dir,
                transform=transforms.Compose(
                    transforms.RGB(),
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToImage(**spt.data.static.ImageNet),
                ),
            ),
            batch_size=batch_size,
            num_workers=8,
            persistent_workers=True,
        ),
    )

    module = MAE(
        model_or_model_name="vit_small_patch16_224",
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=6,
        mask_ratio=0.75,
        block_size=1,
        norm_pix_loss=True,
        loss_type="mse",
        pretrained=False,
    )
    module.forward = types.MethodType(mae_forward, module)
    module.optim = {
        "optimizer": {
            "type": "AdamW",
            "lr": 5e-4,
            "weight_decay": 0.05,
            "betas": (0.9, 0.95),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "peak_step": 2 / max_epochs,
            "start_factor": 0.01,
            "end_lr": 5e-5,
            "total_steps": (len(data.train) // num_gpus) * max_epochs,
        },
        "interval": "step",
    }

    embed_dim = 384  # ViT-S/16
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        callbacks=[
            spt.callbacks.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=nn.Linear(embed_dim, 10),
                loss=nn.CrossEntropyLoss(),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(10),
                    "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
                },
                optimizer={"type": "AdamW", "lr": 0.025, "weight_decay": 0.0},
            ),
            spt.callbacks.OnlineKNN(
                name="knn_probe",
                input="embedding",
                target="label",
                queue_length=10000,
                metrics={"top1": torchmetrics.classification.MulticlassAccuracy(10)},
                input_dim=embed_dim,
                k=20,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=pl.pytorch.loggers.CSVLogger(
            save_dir=str(Path(__file__).parent / "logs"),
            name="mae-vits-inet10",
        ),
        precision="16-mixed",
        enable_checkpointing=False,
        devices=num_gpus,
        accelerator="gpu",
    )

    spt.Manager(trainer=trainer, module=module, data=data)()


if __name__ == "__main__":
    main()
