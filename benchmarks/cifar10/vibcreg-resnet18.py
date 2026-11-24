"""VIbCReg training on CIFAR-10, with the VICReg baseline and IterNorm ablation.

Follows the solo-learn recipes behind the reference results (ResNet-18,
1000 epochs, online linear top-1):

- ``--method vibcreg`` (91.18%): 2048-d GELU projector ending in IterNorm,
  loss weights 25/25/200, LARS with lr 0.3, 10 warmup epochs.
- ``--method vicreg`` (92.07%): 2048-d ReLU projector, loss weights 25/25/1,
  same optimizer with LARS ``clip_lr``.

Both share the augmentations, backbone, optimizer and schedule, so they differ
only where the two methods do.

Usage:
    python benchmarks/cifar10/vibcreg-resnet18.py
    python benchmarks/cifar10/vibcreg-resnet18.py --method vicreg
    python benchmarks/cifar10/vibcreg-resnet18.py --no-iternorm  # ablation
    python benchmarks/cifar10/vibcreg-resnet18.py --seed 1  # repeat with another seed
    python benchmarks/cifar10/vibcreg-resnet18.py --iternorm-lars --probe-schedule warmup-cosine
    python benchmarks/cifar10/vibcreg-resnet18.py --epochs 2 --limit-batches 5  # smoke check
    python benchmarks/cifar10/vibcreg-resnet18.py --ckpt-path /abs/path/last.ckpt  # resume
"""

import argparse
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

WARMUP_EPOCHS = 10
PROJECTOR_DIM = 2048


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["vibcreg", "vicreg"], default="vibcreg")
    parser.add_argument(
        "--no-iternorm",
        action="store_true",
        help="VIbCReg without the IterNorm layer (ablation).",
    )
    parser.add_argument(
        "--iternorm-lars",
        action="store_true",
        help="Apply LARS adaptation and weight decay to IterNorm's affine "
        "parameters, as solo-learn does.",
    )
    parser.add_argument(
        "--probe-schedule",
        choices=["constant", "warmup-cosine"],
        default="constant",
        help="Learning-rate schedule of the online linear probe; solo-learn "
        "decays it with the main warmup-cosine schedule.",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Cap train/val batches per epoch, for a quick end-to-end check.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Absolute path to a last.ckpt to resume training from.",
    )
    args = parser.parse_args()
    if args.method != "vibcreg" and (args.no_iternorm or args.iternorm_lars):
        parser.error("--no-iternorm / --iternorm-lars only apply to --method vibcreg")
    if args.no_iternorm and args.iternorm_lars:
        parser.error("--iternorm-lars needs IterNorm; drop --no-iternorm")
    return args


def _view_transform():
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomSolarize(threshold=128, p=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(**spt.data.static.CIFAR10),
    )


def _warmup_cosine(optimizer, module, epochs):
    # Warmup starts at 3e-5 (= 1e-4 * lr) as in solo-learn; total steps are
    # only known once the trainer is attached.
    total_steps = module.trainer.estimated_stepping_batches
    return spt.optim.LinearWarmupCosineAnnealing(
        optimizer,
        total_steps=total_steps,
        start_factor=1e-4,
        end_lr=0.0,
        peak_step=max(1, int(total_steps * WARMUP_EPOCHS / epochs)),
    )


def _cifar_resnet18():
    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
    backbone.fc = nn.Identity()
    return backbone


def _make_iternorm_affine_2d(module):
    # LARS with exclude_bias_n_norm skips adaptation and weight decay for 1-D
    # parameters. solo-learn stores IterNorm's affine parameters as [1, D], so
    # they are adapted like weight matrices; reshaping reproduces that.
    for m in module.modules():
        if isinstance(m, spt.backbone.IterNorm) and m.affine:
            m.weight = nn.Parameter(m.weight.detach().reshape(1, -1))
            m.bias = nn.Parameter(m.bias.detach().reshape(1, -1))


def _build_module(args):
    if args.method == "vibcreg":
        module = spt.VIbCReg(
            encoder_name=_cifar_resnet18(),
            projector_dims=(PROJECTOR_DIM,) * 3,
            sim_coeff=25.0,
            std_coeff=25.0,
            cov_coeff=200.0,
            iternorm=not args.no_iternorm,
        )
        if args.iternorm_lars:
            _make_iternorm_affine_2d(module)
        clip_lr = False
    else:
        # The VICReg method class predates the dict forward(batch, stage)
        # contract, so the baseline is assembled from the forward function.
        module = spt.Module(
            backbone=_cifar_resnet18(),
            projector=nn.Sequential(
                nn.Linear(512, PROJECTOR_DIM),
                nn.BatchNorm1d(PROJECTOR_DIM),
                nn.ReLU(),
                nn.Linear(PROJECTOR_DIM, PROJECTOR_DIM),
                nn.BatchNorm1d(PROJECTOR_DIM),
                nn.ReLU(),
                nn.Linear(PROJECTOR_DIM, PROJECTOR_DIM),
            ),
            forward=forward.vicreg,
            vicreg_loss=spt.losses.VICRegLoss(
                sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0
            ),
        )
        clip_lr = True

    module.optim = {
        "optimizer": {
            "type": "LARS",
            "lr": 0.3,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "eta": 0.02,
            "clip_lr": clip_lr,
            "exclude_bias_n_norm": True,
        },
        "scheduler": lambda optimizer, module: _warmup_cosine(
            optimizer, module, args.epochs
        ),
        "interval": "step",
    }
    return module


def _run_name(args):
    name = args.method + ("-no-iternorm" if args.no_iternorm else "")
    name += "-iternorm-lars" if args.iternorm_lars else ""
    name += "-probe-cosine" if args.probe_schedule == "warmup-cosine" else ""
    return f"{name}-resnet18-{args.epochs}ep-seed{args.seed}"


def main():
    args = _parse_args()

    train_transform = transforms.MultiViewTransform(
        [_view_transform(), _view_transform()]
    )
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((32, 32)),
        transforms.ToImage(**spt.data.static.CIFAR10),
    )

    data_dir = get_data_dir("cifar10")
    cifar_train = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True
    )
    cifar_val = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True
    )

    train_dataset = spt.data.FromTorchDataset(
        cifar_train,
        names=["image", "label"],
        transform=train_transform,
    )
    val_dataset = spt.data.FromTorchDataset(
        cifar_val,
        names=["image", "label"],
        transform=val_transform,
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
        persistent_workers=args.num_workers > 0,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)
    module = _build_module(args)

    linear_probe = spt.callbacks.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=torch.nn.Linear(512, 10),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={
            "top1": torchmetrics.classification.MulticlassAccuracy(10),
            "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
        },
        scheduler=(
            (lambda optimizer, module: _warmup_cosine(optimizer, module, args.epochs))
            if args.probe_schedule == "warmup-cosine"
            else None
        ),
    )

    knn_probe = spt.callbacks.OnlineKNN(
        name="knn_probe",
        input="embedding",
        target="label",
        queue_length=20000,
        metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
        input_dim=512,
        k=10,
    )

    logger = True
    if args.wandb_project:
        logger = WandbLogger(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=_run_name(args),
            log_model=False,
        )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        callbacks=[knn_probe, linear_probe],
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=logger,
        limit_train_batches=args.limit_batches,
        limit_val_batches=args.limit_batches,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data,
        seed=args.seed,
        ckpt_path=args.ckpt_path,
        weights_only=False,
    )
    manager()


if __name__ == "__main__":
    main()
