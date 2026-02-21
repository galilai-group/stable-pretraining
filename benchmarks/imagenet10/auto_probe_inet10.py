"""Offline linear probe evaluation for I-JEPA checkpoints on ImageNet-10.

Sweeps LRs and weight decays using AutoLinearClassifier.

Usage:
    python benchmarks/imagenet10/linear-probe-inet10.py --ckpt-dir /path/to/ckpt-dir
    python benchmarks/imagenet10/linear-probe-inet10.py --single /path/to/ckpt.ckpt
"""

import argparse
import sys
import time
from glob import glob
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining.backbone.probe import AutoLinearClassifier
from stable_pretraining.data import transforms
from stable_pretraining.methods.ijepa import IJEPA

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir


# Config
IMAGE_SIZE = 224
EMBED_DIM = 768
NUM_CLASSES = 10
PROBE_EPOCHS = 100
BATCH_SIZE = 512
PROBE_LR_SCALING = 200


# Data
def get_data():
    train_transform = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.2, 1.0)),
        transforms.ToImage(**spt.data.static.ImageNet),
    )
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToImage(**spt.data.static.ImageNet),
    )

    data_dir = get_data_dir("imagenet10")
    from stable_datasets.images.imagenette import Imagenette

    # We do this cause HF transforms apply to entire batches at a time
    # so we need to inject them here
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

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=16,
        drop_last=True,
        persistent_workers=True,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=16,
        persistent_workers=True,
    )
    return spt.data.DataModule(train=train_loader, val=val_loader)


def probe_forward(self, batch, stage):
    with torch.no_grad():
        output = IJEPA.forward(self.backbone, batch["image"])
        embedding = output.embedding.mean(dim=1)  # avg pool patches, matches paper

    labels = batch["label"].long()
    total_loss = self.head(embedding, y=labels, pl_module=self)
    return {"loss": total_loss}


def load_backbone(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone = IJEPA(
        encoder_name="vit_base_patch16_224",
        predictor_embed_dim=384,
        predictor_depth=12,
        pretrained=False,
    )
    missing, unexpected = backbone.load_state_dict(
        checkpoint["state_dict"], strict=False
    )
    if missing:
        raise RuntimeError(f"Missing keys loading checkpoint: {missing[:5]}")
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone


def make_probe(ckpt_path):
    backbone = load_backbone(ckpt_path)
    head = AutoLinearClassifier(
        name="probe",
        embedding_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
        pooling="mean",
        normalization=["none", "bn"],
        lr_scaling=[10, 50, 200],
        weight_decay=[0.0, 5e-4],
        dropout=[0],
        label_smoothing=[0],
    )
    return spt.Module(
        backbone=backbone,
        head=head,
        forward=probe_forward,
        optim={
            "optimizer": {
                "type": "SGD",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0,
            },
            "scheduler": {"type": "StepLR", "step_size": 15, "gamma": 0.1},
        },
    )


# Main
def extract_epoch(ckpt_path: str) -> int:
    name = Path(ckpt_path).stem
    for part in name.split("-"):
        if part.startswith("epoch="):
            return int(part.split("=")[1])
    return -1


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ckpt-dir", type=str)
    group.add_argument("--single", type=str)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    if args.single:
        ckpt_paths = [args.single]
    else:
        ckpt_dir = Path(args.ckpt_dir)
        ckpt_paths = sorted(glob(str(ckpt_dir / "*.ckpt")))
        ckpt_paths = [p for p in ckpt_paths if "last" not in Path(p).stem]
        if not ckpt_paths:
            ckpt_paths = sorted(glob(str(ckpt_dir / "*.ckpt")))

    if not ckpt_paths:
        print(f"No checkpoints found in {args.ckpt_dir}")
        sys.exit(1)

    data = get_data()

    for ckpt_path in ckpt_paths:
        epoch_num = extract_epoch(ckpt_path)
        print(f"\nProbing: {Path(ckpt_path).name} (epoch {epoch_num})")

        probe = make_probe(ckpt_path)

        logger = (
            WandbLogger(
                entity="stable-ssl",
                project="imagenet10-ijepa-probe",
                name=f"probe-inet10-ep{epoch_num:03d}-{time.time():.0f}",
                log_model=False,
            )
            if not args.no_wandb
            else False
        )

        trainer = pl.Trainer(
            max_epochs=PROBE_EPOCHS,
            num_sanity_val_steps=0,
            callbacks=[LearningRateMonitor(logging_interval="epoch")],
            precision="16-mixed",
            logger=logger,
            devices=1,
            accelerator="gpu",
            enable_checkpointing=False,
        )

        spt.Manager(trainer=trainer, module=probe, data=data)()

        if logger:
            logger.experiment.finish()


if __name__ == "__main__":
    main()
