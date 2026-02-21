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
from stable_pretraining.backbone.probe import AutoLinearClassifier
from stable_pretraining.data import transforms
from stable_pretraining.methods.mae import MAE
from stable_datasets.images.imagenette import Imagenette

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir


IMAGE_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 768

DECODER_DEPTH = 8
DECODER_DIM = 512
DECODER_HEADS = 16

MASK_RATIO = 0.75
BASE_LR = 5e-4
BATCH_SIZE = 64
NUM_GPUS = torch.cuda.device_count() or 1

WEIGHT_DECAY = 0.05
BETAS = (0.9, 0.95)

# Training
MAX_EPOCHS = 600
WARMUP_EPOCHS = 40
NUM_CLASSES = 10  # ImageNette

# Checkpointing
SAVE_EVERY_N_EPOCHS = 300
CKPT_DIR = str(Path(__file__).parent / "checkpoints" / "mae-vitb")

PROBE_LR_SCALING = 100


def mae_forward(self, batch, stage):
    images = batch["image"]
    output = MAE.forward(self, images)

    # Extract encoder features for online probing (no masking, stop gradients)
    with torch.no_grad():
        features = self.encoder.forward_features(images)
    # Pool over patch tokens, skipping the CLS token at index 0
    embedding = features[:, 1:].mean(dim=1).detach()

    out = {
        "loss": output.loss,
        "embedding": embedding,
    }

    if "label" in batch:
        out["label"] = batch["label"].long()

    if self.training:
        self.log(
            f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True
        )
    return out


# MAE uses only random resized crop + horizontal flip (no color jitter)
train_transform = transforms.Compose(
    transforms.RGB(),
    transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToImage(**spt.data.static.ImageNet),
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToImage(**spt.data.static.ImageNet),
)

data_dir = get_data_dir("imagenet10")


class _HFDataset(spt.data.Dataset):
    def __init__(self, hf_dataset, transform):
        super().__init__(transform)
        self.dataset = hf_dataset

    def __getitem__(self, idx):
        return self.process_sample(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


_builder = Imagenette(config_name="imagenette", cache_dir=str(data_dir))
_builder.download_and_prepare()
train_dataset = _HFDataset(_builder.as_dataset(split="train"), train_transform)
val_dataset = _HFDataset(_builder.as_dataset(split="test"), val_transform)

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


module = MAE(
    encoder_name="vit_base_patch16_224",
    decoder_embed_dim=DECODER_DIM,
    decoder_depth=DECODER_DEPTH,
    decoder_num_heads=DECODER_HEADS,
    mask_ratio=MASK_RATIO,
    block_size=1,  # random masking
    norm_pix_loss=True,  # normalize pixel targets per patch
    loss_type="mse",
    pretrained=False,
)

# Bind spt.Module-compatible forward and optimizer config
module.forward = types.MethodType(mae_forward, module)
module.optim = {
    "optimizer": {
        "type": "AdamW",
        "lr": BASE_LR,
        "weight_decay": WEIGHT_DECAY,
        "betas": BETAS,
    },
    "scheduler": {
        "type": "LinearWarmupCosineAnnealing",
        "peak_step": WARMUP_EPOCHS / MAX_EPOCHS,
        "start_factor": 0.01,
        "end_lr": BASE_LR / 10,
        "total_steps": (len(train_dataloader) // NUM_GPUS) * MAX_EPOCHS,
    },
    "interval": "step",
}


class AutoProbeWrapper(nn.Module):
    """Wrapper for AutoLinearClassifier to work with OnlineProbe."""

    def __init__(self, classifier, input_key="embedding", target_key="label"):
        super().__init__()
        self.classifier = classifier
        self.input_key = input_key
        self.target_key = target_key

    def forward(self, batch, outputs, pl_module):
        x = outputs[self.input_key].detach()
        y = batch[self.target_key].long()
        loss = self.classifier(x, y=y, pl_module=pl_module)
        outputs["loss"] = outputs.get("loss", 0) + loss
        return outputs


auto_probe = spt.callbacks.OnlineProbe(
    module,
    name="auto_probe",
    input=None,
    target=None,
    probe=AutoProbeWrapper(
        AutoLinearClassifier(
            name="probe",
            embedding_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
            pooling="mean",
            normalization=["bn"],
            lr_scaling=[PROBE_LR_SCALING, PROBE_LR_SCALING // 2, PROBE_LR_SCALING // 4],
            weight_decay=[0.0],
            dropout=[0],
            label_smoothing=[0],
        ),
    ),
    loss=None,
    metrics=None,
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
    filename="mae-vitb-{epoch:03d}",
    save_top_k=-1,
    every_n_epochs=SAVE_EVERY_N_EPOCHS,
    save_last=True,
)

lr_monitor = LearningRateMonitor(logging_interval="step")

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="imagenet10-mae-ijepa",
    name="mae-vitb-inet10",
    log_model=False,
    config={
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "embed_dim": EMBED_DIM,
        "base_lr": BASE_LR,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "betas": BETAS,
        "max_epochs": MAX_EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "num_classes": NUM_CLASSES,
        "probe_lr_scaling": PROBE_LR_SCALING,
        "num_gpus": NUM_GPUS,
        "encoder": "vit_base_patch16_224",
        "method": "mae",
        "dataset": "imagenet10",
    },
)


trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    num_sanity_val_steps=0,
    callbacks=[
        auto_probe,
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
