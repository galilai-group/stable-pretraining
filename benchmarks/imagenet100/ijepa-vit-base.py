import time
import types
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.ijepa import IJEPA

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir


IMAGE_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 768  # ViT-Base hidden dimension

# Predictor (paper: depth=12, width=384 for ViT-Base)
PRED_DEPTH = 12
PRED_DIM = 384

# Optimizer (paper: AdamW, lr=1.5e-4, wd=0.05, betas=(0.9, 0.95))
# Paper uses batch_size=2048 with lr=1.5e-4. Scale linearly with effective batch.
BASE_LR = 5e-4
BATCH_SIZE = 256
NUM_GPUS = torch.cuda.device_count() or 1
EFFECTIVE_BATCH = BATCH_SIZE * NUM_GPUS
SCALED_LR = BASE_LR * (EFFECTIVE_BATCH / 2048)
WEIGHT_DECAY = 0.05
BETAS = (0.9, 0.95)

# EMA schedule (paper: 0.996 → 1.0 cosine)
BASE_EMA = 0.996
FINAL_EMA = 1.0

MAX_EPOCHS = 300
WARMUP_EPOCHS = 15
NUM_CLASSES = 100  # ImageNet100

SAVE_EVERY_N_EPOCHS = 25
CKPT_DIR = str(Path(__file__).parent / "checkpoints" / "ijepa-vitb")


def ijepa_forward(self, batch, stage):
    output = IJEPA.forward(self, batch["image"])
    out = {
        "loss": output.loss,
        "embedding": output.embedding.mean(dim=1).detach()
        if self.training
        else output.embedding.mean(dim=1),
    }

    if "label" in batch:
        out["label"] = batch["label"].long()

    if self.training:
        self.log(
            f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True
        )
    return out


# I-JEPA uses only random resized crop + horizontal flip (no color jitter)
train_transform = transforms.Compose(
    transforms.RGB(),
    transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.3, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToImage(**spt.data.static.ImageNet),
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
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

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=(num_workers := 16),
    drop_last=True,
    persistent_workers=num_workers > 0,
    shuffle=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=(num_workers := 16),
    persistent_workers=num_workers > 0,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)


module = IJEPA(
    encoder_name="vit_base_patch16_224",
    predictor_embed_dim=PRED_DIM,
    predictor_depth=PRED_DEPTH,
    num_targets=4,
    target_scale=(0.15, 0.2),
    target_aspect_ratio=(0.75, 1.5),
    context_scale=(0.85, 1.0),
    ema_decay_start=BASE_EMA,
    ema_decay_end=FINAL_EMA,
    pretrained=False,
)

module.forward = types.MethodType(ijepa_forward, module)
module.optim = {
    "optimizer": {
        "type": "AdamW",
        "lr": SCALED_LR,
        "weight_decay": WEIGHT_DECAY,
        "betas": BETAS,
    },
    "scheduler": {
        "type": "LinearWarmupCosineAnnealing",
    },
    "interval": "epoch",
}


teacher_student_callback = spt.callbacks.TeacherStudentCallback(
    update_frequency=1,
    update_after_backward=False,
)

linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=nn.Linear(EMBED_DIM, NUM_CLASSES),
    loss=nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES),
        "top5": torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES, top_k=5),
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
    queue_length=10000,
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES),
    },
    input_dim=EMBED_DIM,
    k=20,
)

rankme = spt.callbacks.RankMe(
    name="rankme",
    target="embedding",
    queue_length=1000,
    target_shape=EMBED_DIM,
)

checkpoint_callback = ModelCheckpoint(
    dirpath=CKPT_DIR,
    filename="ijepa-vitb-{epoch:03d}",
    save_top_k=-1,
    every_n_epochs=SAVE_EVERY_N_EPOCHS,
    save_last=True,
)

lr_monitor = LearningRateMonitor(logging_interval="step")

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="imagenet100-mae-ijepa",
    name=f"new-ijepa-vitb-inet100-{time.time():.0f}",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    num_sanity_val_steps=0,
    callbacks=[
        teacher_student_callback,
        linear_probe,
        knn_probe,
        rankme,
        checkpoint_callback,
        lr_monitor,
    ],
    precision="16-mixed",
    logger=wandb_logger,
    devices=NUM_GPUS,
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true" if NUM_GPUS > 1 else "auto",
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
