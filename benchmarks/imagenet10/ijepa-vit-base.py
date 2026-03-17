import sys
import types
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.ijepa import IJEPA


def main():
    sys.path.append(str(Path(__file__).parent.parent))
    from utils import get_data_dir

    num_gpus = torch.cuda.device_count() or 1
    batch_size = 64

    def ijepa_forward(self, batch, stage):
        output = IJEPA.forward(self, batch["image"], embedding_source="student")
        embedding = output.embedding.mean(dim=1)
        if self.training:
            embedding = embedding.detach()

        self.log(
            f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True
        )

        return {
            "loss": output.loss,
            "embedding": embedding,
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
                    transforms.RandomResizedCrop((224, 224), scale=(0.3, 1.0)),
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
            num_workers=(num_workers := 16),
            persistent_workers=num_workers > 0,
        ),
    )

    module = IJEPA(
        encoder_name="vit_base_patch16_224",
        predictor_embed_dim=384,
        predictor_depth=12,
        num_targets=4,
        target_scale=(0.15, 0.2),
        target_aspect_ratio=(0.75, 1.5),
        context_scale=(0.85, 1.0),
        ema_decay_start=0.996,
        ema_decay_end=1.0,
        pretrained=False,
    )

    module.forward = types.MethodType(ijepa_forward, module)
    module.optim = {
        "optimizer": {
            "type": "AdamW",
            "lr": 6e-4,
            "weight_decay": 0.05,
            "betas": (0.9, 0.95),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "peak_step": 300 / 600,
            "start_factor": 0.01,
            "end_lr": 6e-4 / 10,
            "total_steps": (len(data.train) // num_gpus) * 600,
        },
        "interval": "step",
    }

    trainer = pl.Trainer(
        max_epochs=600,
        num_sanity_val_steps=0,
        callbacks=[
            spt.callbacks.TeacherStudentCallback(
                update_frequency=1,
                update_after_backward=True,
            ),
            spt.callbacks.OnlineProbe(
                module,
                name="linear_probe",
                input="embedding",
                target="label",
                probe=nn.Linear(768, 10),
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
                dirpath=str(Path(__file__).parent / "checkpoints" / "ijepa-vitb"),
                filename="ijepa-vitb-{epoch:03d}",
                save_top_k=-1,
                every_n_epochs=300,
                save_last=True,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=pl.pytorch.loggers.WandbLogger(
            entity="stable-ssl",
            project="imagenet10-methods",
            name="ijepa-vitb-inet10",
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
