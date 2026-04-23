import sys
import os
import types
from pathlib import Path

import lightning as pl
import time
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from torch import nn

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.salt import SALT

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

train_transform = transforms.Compose(
    transforms.RGB(),
    transforms.RandomResizedCrop((224, 224), scale=(0.4, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToImage(**spt.data.static.ImageNet),
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToImage(**spt.data.static.ImageNet),
)

data_dir = get_data_dir("imagenet100")

train_dataset = spt.data.HFDataset(
    "clane9/imagenet-100",
    split="train",
    cache_dir=str(data_dir),
    transform=train_transform,
)
val_dataset = spt.data.HFDataset(
    "clane9/imagenet-100",
    split="validation",
    cache_dir=str(data_dir),
    transform=val_transform,
)

batch_size = 256
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=4,
    drop_last=True,
    persistent_workers=True,
    shuffle=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)


def salt_forward(self, batch, stage):
    output = SALT.forward(self, batch["image"])

    self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True)

    return {
        "loss": output.loss,
        "embedding": output.embedding,
        **({"label": batch["label"].long()} if "label" in batch else {}),
    }


# Create SALT model — optionally from Stage 1 checkpoint
ckpt_path = os.environ.get("SALT_TEACHER_CKPT")
if ckpt_path is not None:
    module = SALT.from_checkpoint(
        ckpt_path,
        encoder_name="vit_tiny_patch16_224",
        predictor_embed_dim=384,
        predictor_depth=12,
        predictor_num_heads=16,
    )
else:
    module = SALT(
        encoder_name="vit_tiny_patch16_224",
        predictor_embed_dim=384,
        predictor_depth=12,
        predictor_num_heads=16,
    )

module.forward = types.MethodType(salt_forward, module)
module.optim = {
    "main": {
        "modules": "student|predictor",
        "optimizer": {
            "type": "AdamW",
            "lr": 0.000625,
            "weight_decay": 0.04,
            "betas": (0.9, 0.95),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
    },
}

linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=nn.Linear(192, 100),
    loss=nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(100),
        "top5": torchmetrics.classification.MulticlassAccuracy(100, top_k=5),
    },
    optimizer={
        "type": "AdamW",
        "lr": 3e-3,
        "weight_decay": 1e-4,
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(100)},
    input_dim=192,
    k=20,
)

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="imagenet100-salt",
    name=f"salt-stage2-vit-tiny-{time.time()}",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=400,
    num_sanity_val_steps=0,
    callbacks=[linear_probe, knn_probe],
    precision="16-mixed",
    logger=wandb_logger,
    devices=1,
    accelerator="gpu",
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
