"""Compare FlatNCE and SimCLR on CIFAR-10 with an otherwise identical recipe."""

import argparse
import json
import math
import os
from pathlib import Path
import time

import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassAccuracy
import torchvision

import stable_pretraining as spt


class ContrastiveDiagnostics(pl.Callback):
    """Track model gradients and local backbone geometry alongside probe accuracy."""

    def on_after_backward(self, trainer: pl.Trainer, pl_module: spt.Module) -> None:
        gradients = [
            parameter.grad.detach().float().norm()
            for component in (pl_module.backbone, pl_module.projector)
            for parameter in component.parameters()
            if parameter.grad is not None
        ]
        if gradients:
            pl_module.log(
                "diagnostics/gradient_norm",
                torch.stack(gradients).norm(),
                sync_dist=True,
            )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: spt.Module,
        outputs: dict[str, torch.Tensor],
        batch: dict,
        batch_idx: int,
    ) -> None:
        embeddings = outputs["embedding"].detach().float()
        first, second = nn.functional.normalize(embeddings, dim=-1).chunk(2)
        pl_module.log(
            "diagnostics/positive_cosine",
            (first * second).sum(-1).mean(),
            sync_dist=True,
        )
        pl_module.log(
            "diagnostics/embedding_std",
            embeddings.std(dim=0, unbiased=False).mean(),
            sync_dist=True,
        )


def _checkpoint_callback(dirpath: Path | None = None) -> ModelCheckpoint:
    """Save resumable state at every completed epoch and at train end."""
    return ModelCheckpoint(
        dirpath=dirpath,
        save_last=True,
        save_top_k=1,
        every_n_epochs=1,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("flatnce", "simclr"), default="flatnce")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--accelerator", choices=("auto", "cpu", "gpu", "mps"), default="auto"
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--precision", choices=("32-true", "16-mixed", "bf16-mixed"), default="32-true"
    )
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument(
        "--eval-split", choices=("validation", "test"), default="validation"
    )
    parser.add_argument(
        "--train-samples", type=int, default=0, help="Optional training subset size."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use synthetic images without downloading CIFAR-10.",
    )
    parser.add_argument(
        "--download", action="store_true", help="Download CIFAR-10 if it is missing."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("STABLE_PRETRAINING_DATA_DIR", "data")) / "cifar10",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/flatnce"))
    parser.add_argument(
        "--resume", type=Path, help="Resume a complete training checkpoint."
    )
    args = parser.parse_args()
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("temperature must be finite and strictly positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and strictly positive")
    if args.batch_size < 2 or args.epochs < 1 or args.devices < 1:
        parser.error("batch-size must be >= 2; epochs and devices must be >= 1")
    if args.num_workers < 0 or args.train_samples < 0:
        parser.error("num-workers and train-samples must be nonnegative")
    if not 0 < args.validation_size < 50000:
        parser.error("validation-size must be between 1 and 49999")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("max-steps must be -1 or a positive integer")
    return args


def _datasets(
    args: argparse.Namespace,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if args.smoke_test:
        train = torchvision.datasets.FakeData(
            size=max(4 * args.batch_size, 32), image_size=(3, 32, 32), num_classes=10
        )
        evaluation = torchvision.datasets.FakeData(
            size=max(2 * args.batch_size, 16),
            image_size=(3, 32, 32),
            num_classes=10,
            random_offset=len(train),
        )
    else:
        full_train = torchvision.datasets.CIFAR10(
            root=str(args.data_dir.resolve()), train=True, download=args.download
        )
        # A fixed split independent of training seed keeps every comparison on
        # the same held-out images, including multi-seed sweeps.
        indices = torch.randperm(
            len(full_train), generator=torch.Generator().manual_seed(0)
        ).tolist()
        train = Subset(full_train, indices[args.validation_size :])
        if args.eval_split == "validation":
            evaluation = Subset(full_train, indices[: args.validation_size])
        else:
            evaluation = torchvision.datasets.CIFAR10(
                root=str(args.data_dir.resolve()), train=False, download=args.download
            )
    if args.train_samples:
        train = Subset(train, range(min(args.train_samples, len(train))))
    if len(train) < args.batch_size * args.devices:
        raise ValueError("training set must contain at least one full batch per device")
    return train, evaluation


def main() -> None:
    """Run a seeded contrastive experiment and save local metrics and checkpoints."""
    args = _parse_args()
    args.output_dir = args.output_dir.resolve()
    pl.seed_everything(args.seed, workers=True)
    spt.set(
        cache_dir=str(args.output_dir / "registry"),
        requeue_checkpoint=False,
        default_loggers={"registry": True},
        default_callbacks={
            "env_dump": False,
            "hf_checkpoint": False,
            "sklearn_checkpoint": False,
        },
    )
    transforms = spt.data.transforms

    def training_transform() -> nn.Module:
        return transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.CIFAR10),
        )

    train, evaluation = _datasets(args)
    train_dataset = spt.data.FromTorchDataset(
        train,
        names=["image", "label"],
        transform=transforms.MultiViewTransform(
            [training_transform(), training_transform()]
        ),
    )
    val_dataset = spt.data.FromTorchDataset(
        evaluation,
        names=["image", "label"],
        transform=transforms.Compose(
            transforms.RGB(),
            transforms.Resize((32, 32)),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
    )
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    data = spt.data.DataModule(
        train=DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs),
        val=DataLoader(val_dataset, **loader_kwargs),
    )
    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
    backbone.fc = nn.Identity()
    projector = nn.Sequential(
        nn.Linear(512, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, 256),
    )
    optim = {"optimizer": {"type": "LARS", "lr": args.lr, "weight_decay": 1e-6}}
    # A one-update smoke run has no room for both warmup and cosine decay.
    if args.max_steps != 1:
        optim["scheduler"] = {"type": "LinearWarmupCosineAnnealing"}
        optim["interval"] = "step"
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    objective = (
        {"flatnce_loss": spt.losses.FlatNCELoss(args.temperature)}
        if args.method == "flatnce"
        else {"simclr_loss": spt.losses.NTXEntLoss(args.temperature)}
    )
    module = spt.Module(
        backbone=backbone,
        projector=projector,
        forward=getattr(spt.forward, args.method),
        optim=optim,
        hparams=settings,
        **objective,
    )
    probe = spt.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=nn.Linear(512, 10),
        loss=nn.CrossEntropyLoss(),
        metrics={"top1": MulticlassAccuracy(10)},
    )
    knn = spt.OnlineKNN(
        name="knn_probe",
        input="embedding",
        target="label",
        queue_length=20000,
        input_dim=512,
        k=10,
        metrics={"accuracy": MulticlassAccuracy(10)},
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        num_sanity_val_steps=0,
        logger=False,
        log_every_n_steps=1,
        callbacks=[
            knn,
            probe,
            ContrastiveDiagnostics(),
            LearningRateMonitor(logging_interval="step"),
            _checkpoint_callback(),
        ],
        default_root_dir=str(args.output_dir),
    )
    started = time.perf_counter()
    spt.Manager(
        trainer=trainer,
        module=module,
        data=data,
        seed=args.seed,
        ckpt_path=str(args.resume.resolve()) if args.resume else None,
        weights_only=False,
    )()
    elapsed = time.perf_counter() - started
    if trainer.is_global_zero:
        summary = {
            "settings": settings,
            "dataset": "FakeData" if args.smoke_test else "CIFAR-10",
            "train_images": len(train),
            "evaluation_images": len(evaluation),
            "global_step": trainer.global_step,
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(
                trainer.strategy.root_device
            )
            if trainer.strategy.root_device.type == "cuda"
            else None,
            "metrics": {
                name: value.detach().cpu().item()
                for name, value in trainer.callback_metrics.items()
                if torch.is_tensor(value) and value.numel() == 1
            },
        }
        summary_path = Path(trainer.logger.log_dir) / "experiment_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(f"Saved experiment summary to {summary_path}")


if __name__ == "__main__":
    main()
