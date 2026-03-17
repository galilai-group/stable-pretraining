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
    batch_size = 256
    scaled_lr = 1.5e-4 * (batch_size * num_gpus / 4096)

    def mae_forward(self, batch, stage):
        output = MAE.forward(self, batch["image"])
        with torch.no_grad():
            features = self.encoder.forward_features(batch["image"])

        self.log(
            f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True
        )

        return {
            "loss": output.loss,
            "embedding": features[:, 1:].mean(dim=1).detach(),  # skip cls
            **({"label": batch["label"].long()} if "label" in batch else {}),
        }

    data = spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "clane9/imagenet-100",
                split="train",
                cache_dir=str(get_data_dir("imagenet100")),
                transform=transforms.Compose(
                    transforms.RGB(),
                    transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToImage(**spt.data.static.ImageNet),
                ),
            ),
            batch_size=batch_size,
            num_workers=(num_workers := 16),
            drop_last=True,
            persistent_workers=num_workers > 0,
            shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "clane9/imagenet-100",
                split="validation",
                cache_dir=str(get_data_dir("imagenet100")),
                transform=transforms.Compose(
                    transforms.RGB(),
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToImage(**spt.data.static.ImageNet),
                ),
            ),
            batch_size=batch_size,
            num_workers=(num_workers := 16),
            persistent_workers=num_workers > 0,
        ),
    )

    module = MAE(
        encoder_name="vit_base_patch16_224",
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mask_ratio=0.75,
        block_size=1,  # random masking (block_size=1)
        norm_pix_loss=True,  # normalize pixel targets per patch
        loss_type="mse",
        pretrained=False,
    )

    # Bind spt.Module-compatible forward and optimizer config
    module.forward = types.MethodType(mae_forward, module)
    module.optim = {
        "optimizer": {
            "type": "AdamW",
            "lr": scaled_lr,
            "weight_decay": 0.05,
            "betas": (0.9, 0.95),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
    }

    trainer = pl.Trainer(
        max_epochs=600,
        num_sanity_val_steps=0,
        callbacks=[
            spt.callbacks.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=nn.Linear(768, 100),
                loss=nn.CrossEntropyLoss(),
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(100),
                    "top5": torchmetrics.classification.MulticlassAccuracy(
                        100, top_k=5
                    ),
                },
                optimizer={"type": "AdamW", "lr": 3e-3, "weight_decay": 1e-4},
            ),
            spt.callbacks.OnlineKNN(
                name="knn_probe",
                input="embedding",
                target="label",
                queue_length=10000,
                metrics={"top1": torchmetrics.classification.MulticlassAccuracy(100)},
                input_dim=768,
                k=20,
            ),
            spt.callbacks.RankMe(
                name="rankme",
                target="embedding",
                queue_length=1000,
                target_shape=768,
            ),
            pl.pytorch.callbacks.ModelCheckpoint(
                dirpath=str(Path(__file__).parent / "checkpoints" / "mae-vitb"),
                filename="mae-vitb-{epoch:03d}",
                save_top_k=-1,
                every_n_epochs=200,
                save_last=True,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=pl.pytorch.loggers.WandbLogger(
            entity="stable-ssl",
            project="imagenet100-mae-ijepa",
            name="mae-vitb-inet100",
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
