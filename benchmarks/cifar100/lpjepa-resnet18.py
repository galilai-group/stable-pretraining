r"""Reproduce Rectified LpJEPA on CIFAR-100 with ResNet-18.

Defaults match the 1,000-epoch reproduction: global batch 256, two GPUs,
FP16, SyncBatchNorm, p=1, mu=0, and 8,192 random projections. Prepare the
dataset once before launching multiple workers::

    python benchmarks/cifar100/lpjepa-resnet18.py --download-only --data-dir ./data
    torchrun --standalone --nproc-per-node=2 benchmarks/cifar100/lpjepa-resnet18.py \
        --data-dir ./data --output-dir ./outputs/lpjepa-seed5 --seed 5

Add --wandb-project PROJECT and optionally --wandb-entity ENTITY to log to W&B.
Registry metrics and best-encoder/last checkpoints are saved beneath
<output-dir>/spt-cache/runs/. Use --resume PATH for full-state checkpoint loading;
manual resume creates a new registry run and restarts best-checkpoint selection.
Exact replay of random data augmentation after resume is not guaranteed.

For CPU checks, use --accelerator cpu --devices 1 --precision 32-true.
"""

import argparse
import os
from pathlib import Path
from typing import Iterable

import lightning as pl
import torch
import torchvision
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins.environments import TorchElasticEnvironment
from lpjepa_precision import JointStepGuard, JointStepMixedPrecision
from timm.optim import Lars
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision.transforms import InterpolationMode

import stable_pretraining as spt
from stable_pretraining.data import transforms as T
from stable_pretraining.methods.lpjepa import RDMReg


def lpjepa_scheduler(
    optimizer: torch.optim.Optimizer, module: spt.Module
) -> SequentialLR:
    """Use ten epochs of linear warmup followed by per-step cosine decay.

    Args:
        optimizer: SSL or probe optimizer.
        module: Training module attached to its Trainer.

    Returns:
        The scheduler used for the CIFAR-100 reproduction.
    """
    total_steps = int(module.trainer.estimated_stepping_batches)
    warmup_steps = int(10 * total_steps / module.trainer.max_epochs)
    # The official schedule reaches peak LR one step before the transition.
    warmup = LinearLR(
        optimizer,
        start_factor=3e-5 / optimizer.param_groups[0]["lr"],
        total_iters=warmup_steps - 1,
    )
    decay = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=0.0
    )
    return SequentialLR(optimizer, [warmup, decay], milestones=[warmup_steps])


def lpjepa_probe_optimizer(parameters: Iterable[nn.Parameter]) -> Lars:
    """Match the official linear probes' LARS updates with zero weight decay.

    Args:
        parameters: Trainable parameters of one linear probe.

    Returns:
        A LARS optimizer that adapts weight updates and leaves biases unscaled.
    """
    parameters = list(parameters)
    # SPT's LARS disables adaptation at zero weight decay; the authors' does not.
    groups = [
        {"params": [p for p in parameters if p.ndim != 1], "always_adapt": True},
        {"params": [p for p in parameters if p.ndim == 1], "always_adapt": False},
    ]
    return Lars(
        groups,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0,
        trust_coeff=0.02,
        eps=1e-8,
        trust_clip=True,
    )


def main(argv: list[str] | None = None) -> None:
    """Build the CIFAR-100 recipe and train through Manager.

    Args:
        argv: Command-line arguments; defaults to the process arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/lpjepa"))
    parser.add_argument("--devices", type=int, default=2, help="GPUs on one node.")
    parser.add_argument("--accelerator", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument(
        "--precision", choices=("16-mixed", "32-true"), default="16-mixed"
    )
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=Path, help="Full-state checkpoint to resume.")
    parser.add_argument("--wandb-project", help="Enable W&B logging to this project.")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--download-only", action="store_true", help="Download CIFAR-100 and exit."
    )
    args = parser.parse_args(argv)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.download_only:
        for train in (True, False):
            torchvision.datasets.CIFAR100(
                root=args.data_dir, train=train, download=True
            )
        return

    batch_size = 256
    if args.devices < 1 or batch_size % args.devices or batch_size // args.devices < 2:
        parser.error(
            "--devices must divide global batch 256 and leave at least two images per device"
        )
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) != args.devices:
        parser.error("--devices must match torchrun's world size on a single node")
    if args.accelerator == "cpu" and args.precision != "32-true":
        parser.error("CPU runs require --precision 32-true")
    per_rank_batch_size = batch_size // args.devices
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(args.seed, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    spt.set(
        cache_dir=str(args.output_dir / "spt-cache"),
        requeue_checkpoint=True,
        default_loggers={"registry": True},
        default_callbacks={
            "hf_checkpoint": False,
            "sklearn_checkpoint": False,
            "wandb_checkpoint": False,
            "trackio_checkpoint": False,
            "swanlab_checkpoint": False,
        },
    )

    view_transform = T.Compose(
        T.RandomResizedCrop(
            size=32,
            scale=(0.2, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
            p=0.8,
        ),
        T.RandomGrayscale(p=0.2),
        T.RandomSolarize(threshold=128, p=0.1),
        T.RandomHorizontalFlip(p=0.5),
        T.ToImage(
            mean=(0.5071, 0.4865, 0.4409),
            std=(0.2673, 0.2564, 0.2762),
        ),
    )

    train_transform = T.MultiViewTransform([view_transform, view_transform])

    train_dataset = spt.data.FromTorchDataset(
        torchvision.datasets.CIFAR100(
            root=args.data_dir,
            train=True,
            download=False,
        ),
        names=["image", "label"],
        transform=train_transform,
    )

    # Match the official code's shared CIFAR evaluation normalization.
    eval_transform = T.ToImage(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.247, 0.243, 0.261),
    )

    eval_dataset = spt.data.FromTorchDataset(
        torchvision.datasets.CIFAR100(
            root=args.data_dir,
            train=False,
            download=False,
        ),
        names=["image", "label"],
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_rank_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=args.accelerator == "gpu",
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=per_rank_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=args.accelerator == "gpu",
    )

    data = spt.data.DataModule(
        train=train_loader,
        val=eval_loader,
    )

    backbone = spt.backbone.from_torchvision(
        "resnet18", low_resolution=True, weights=None
    )
    # The official paper CIFAR stem uses padding 2; SPT's low-resolution default is 1.
    backbone.conv1.padding = (2, 2)

    # The shared forward applies the final ReLU used by the official rectified MLP.
    projector = nn.Sequential(
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 512),
    )

    module = spt.Module(
        backbone=backbone,
        projector=projector,
        forward=spt.forward.lpjepa,
        rdmreg=RDMReg(num_slices=8192, p=1.0, mu=0.0, sigma=None),
        inv_weight=25.0,
        rdmreg_weight=250.0,
        hparams={
            "seed": args.seed,
            "max_epochs": args.max_epochs,
            "batch_size": batch_size,
            "per_rank_batch_size": per_rank_batch_size,
            "devices": args.devices,
            "precision": args.precision,
        },
        optim={
            "optimizer": {
                "type": "LARS",
                "lr": 0.03,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "eta": 0.02,
                "clip_lr": True,
                "exclude_bias_n_norm": True,
                # Let LARS handle 1-D exclusions, as in the official implementation.
                "exclude_bias_norm": False,
            },
            "scheduler": lpjepa_scheduler,
            "interval": "step",
        },
    )

    encoder_probe = spt.callbacks.OnlineProbe(
        module,
        name="encoder_probe",
        input="embedding",
        target="label",
        probe=nn.Linear(512, 100),
        loss=nn.CrossEntropyLoss(),
        metrics={"top1": MulticlassAccuracy(num_classes=100, top_k=1, average="micro")},
        optimizer=lpjepa_probe_optimizer,
        scheduler=lpjepa_scheduler,
    )

    projector_probe = spt.callbacks.OnlineProbe(
        module,
        name="projector_probe",
        input="projection",
        target="label",
        probe=nn.Linear(512, 100),
        loss=nn.CrossEntropyLoss(),
        metrics={"top1": MulticlassAccuracy(num_classes=100, top_k=1, average="micro")},
        optimizer=lpjepa_probe_optimizer,
        scheduler=lpjepa_scheduler,
    )

    callbacks = [
        encoder_probe,
        projector_probe,
        spt.callbacks.Sparsity(
            name="sparsity",
            target="projection",
            every_n_steps=50,
            train_first_view_only=True,
        ),
    ]

    best = ModelCheckpoint(
        monitor="eval/encoder_probe_top1_epoch",
        mode="max",
        filename="best-encoder",
        save_top_k=1,
        save_last=False,
        save_weights_only=False,
        save_on_train_epoch_end=False,
        enable_version_counter=False,
    )
    callbacks.append(best)
    plugins = []
    if TorchElasticEnvironment.detect():
        plugins.append(TorchElasticEnvironment())
    if args.precision == "16-mixed":
        plugins.append(JointStepMixedPrecision(optimizers_per_step=3))
        callbacks.append(JointStepGuard())

    logger = False
    if args.wandb_project:
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"lpjepa-resnet18-p1-seed{args.seed}",
            save_dir=str(args.output_dir),
            log_model=False,
        )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy="ddp" if args.devices > 1 else "auto",
        sync_batchnorm=args.accelerator == "gpu" and args.devices > 1,
        plugins=plugins,
        precision="32-true" if args.precision == "32-true" else None,
        max_epochs=args.max_epochs,
        use_distributed_sampler=True,
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        logger=logger,
        log_every_n_steps=50,
        default_root_dir=str(args.output_dir),
        benchmark=False,
        deterministic=False,
        callbacks=callbacks,
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=data,
        seed=args.seed,
        ckpt_path=str(args.resume.expanduser().resolve()) if args.resume else None,
        weights_only=False,
        compile=False,
    )()
    if logger and trainer.is_global_zero:
        logger.experiment.finish(exit_code=0)


if __name__ == "__main__":
    main()
