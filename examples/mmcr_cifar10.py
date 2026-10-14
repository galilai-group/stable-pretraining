# ruff: noqa: D205 -- Sphinx-Gallery requires a title underline.
"""MMCR and SimCLR on CIFAR-10.
============================================================

Run from the checkout with ``python examples/mmcr_cifar10.py --download``.
Use ``--fake-data --train-samples 32 --val-samples 16 --batch-size 8`` for an
offline installation check. This resource-conscious recipe is not a reproduction
of the authors' ResNet-50 / 40-view / 500-epoch CIFAR experiment.
"""

import argparse
import json
import time
from pathlib import Path

import lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassAccuracy
from torchvision.datasets import CIFAR10, FakeData

import stable_pretraining as spt


def _data(args):
    if args.fake_data:
        train = FakeData(
            size=args.train_samples or 128, image_size=(3, 32, 32), num_classes=10
        )
        val = FakeData(
            size=args.val_samples or 64,
            image_size=(3, 32, 32),
            num_classes=10,
            random_offset=100000,
        )
    else:
        dataset = CIFAR10(args.data_dir, train=True, download=args.download)
        order = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(args.seed)
        ).tolist()
        train = Subset(dataset, order[5000:][: args.train_samples or None])
        val = Subset(dataset, order[:5000][: args.val_samples or None])
    t = spt.data.transforms
    norm = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
    augment = t.Compose(
        t.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
        t.RandomHorizontalFlip(),
        t.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        t.RandomGrayscale(p=0.2),
        t.ToImage(**norm),
    )
    train = spt.data.FromTorchDataset(
        train,
        names=["image", "label"],
        transform=t.MultiViewTransform([augment for _ in range(args.views)]),
    )
    val = spt.data.FromTorchDataset(
        val, names=["image", "label"], transform=t.ToImage(**norm)
    )
    return train, val


def main() -> None:
    """Train through Manager and save an explicit experiment summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("mmcr", "simclr"), default="mmcr")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("./mmcr-cifar10"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--val-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lmbda", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--accelerator", choices=("cpu", "gpu", "mps", "auto"), default="auto"
    )
    parser.add_argument(
        "--precision", choices=("32-true", "bf16-mixed"), default="32-true"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--save-backbones",
        action="store_true",
        help="Save encoder weights before and after training for fixed-protocol evaluation.",
    )
    args = parser.parse_args()
    if args.batch_size < 2 or args.epochs < 1 or args.views < 2:
        parser.error("batch-size >= 2, epochs >= 1, views >= 2 are required")
    if min(args.train_samples, args.val_samples, args.workers) < 0:
        parser.error("sample limits and workers must be nonnegative")
    if args.method == "simclr" and args.views != 2:
        parser.error("SimCLR needs exactly two views")
    args.cache_dir = args.cache_dir.resolve()
    if args.save_backbones:
        initial_backbone = args.cache_dir / "initial-backbone.pt"
        final_backbone = args.cache_dir / "final-backbone.pt"
        if final_backbone.exists() or (not args.resume and initial_backbone.exists()):
            parser.error(
                "--save-backbones would overwrite existing snapshots; use a dedicated "
                "cache directory for each fresh run and preserve its original baseline"
            )
        if args.resume and not initial_backbone.is_file():
            parser.error(
                "--resume with --save-backbones requires the original "
                "initial-backbone.pt in --cache-dir; preserve the original baseline"
            )
    torch.set_num_threads(4)
    pl.seed_everything(args.seed, workers=True)
    spt.set(
        cache_dir=str(args.cache_dir),
        requeue_checkpoint=False,
        default_callbacks={
            "env_dump": False,
            "trainer_info": False,
            "module_summary": False,
            "slurm_info": False,
        },
    )
    train, val = _data(args)
    if len(train) < args.batch_size:
        parser.error("training dataset must contain at least one full batch")
    backbone = spt.backbone.from_torchvision(
        "resnet18", low_resolution=True, weights=None
    )
    projector = nn.Sequential(
        nn.Linear(512, 512, bias=False),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 128, bias=False),
    )
    optim = {"optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": 1e-4}}
    if args.method == "mmcr":
        model = spt.MMCR(backbone, projector, lmbda=args.lmbda, optim=optim)
    else:
        model = spt.Module(
            forward=spt.forward.simclr,
            backbone=backbone,
            projector=projector,
            simclr_loss=spt.losses.NTXEntLoss(temperature=0.5),
            optim=optim,
            hparams={"method": "simclr", "temperature": 0.5},
        )
    model.save_hyperparameters(
        {**vars(args), "train_size": len(train), "val_size": len(val)}
    )
    if args.save_backbones:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            torch.save(
                model.backbone.state_dict(), args.cache_dir / "initial-backbone.pt"
            )
    probe = spt.OnlineProbe(
        model,
        name="probe",
        input="embedding",
        target="label",
        probe=nn.Linear(512, 10),
        loss=nn.CrossEntropyLoss(),
        optimizer={"type": "AdamW", "lr": 1e-3},
        metrics={"accuracy": MulticlassAccuracy(10, average="micro")},
        verbose=False,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        precision=args.precision,
        max_epochs=args.epochs,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        default_root_dir=str(args.cache_dir),
        callbacks=[probe],
    )
    data = spt.data.DataModule(
        train=DataLoader(
            train,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.workers,
        ),
        val=DataLoader(val, batch_size=args.batch_size, num_workers=args.workers),
    )
    if trainer.strategy.root_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    spt.Manager(
        trainer=trainer,
        module=model,
        data=data,
        seed=args.seed,
        ckpt_path=str(args.resume.resolve()) if args.resume else None,
        weights_only=False,
    )()
    if trainer.strategy.root_device.type == "cuda":
        torch.cuda.synchronize()
    elif trainer.strategy.root_device.type == "mps":
        torch.mps.synchronize()
    if args.save_backbones:
        torch.save(
            {
                key: value.detach().cpu()
                for key, value in model.backbone.state_dict().items()
            },
            args.cache_dir / "final-backbone.pt",
        )
    metrics = {
        key: float(value)
        for key, value in trainer.callback_metrics.items()
        if isinstance(value, torch.Tensor) and value.numel() == 1
    }
    if not all(torch.isfinite(torch.tensor(v)) for v in metrics.values()):
        raise FloatingPointError("Nonfinite training or probe metric")
    summary = {
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "train_size": len(train),
        "val_size": len(val),
        "global_step": trainer.global_step,
        "seconds": time.perf_counter() - start,
        "metrics": metrics,
        "checkpoint": trainer.checkpoint_callback.best_model_path,
        "device": str(trainer.strategy.root_device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20
        if trainer.strategy.root_device.type == "cuda"
        else None,
        "evaluation": "online probe on held-out training split; not frozen linear evaluation",
    }
    summary_path = (
        Path(trainer.checkpoint_callback.dirpath).parent / "experiment-summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Experiment summary: {summary_path}")


if __name__ == "__main__":
    main()
