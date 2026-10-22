"""Train DirectCLR and matched linear-projector or no-projector controls.

Run ``python benchmarks/imagenet100/directclr.py --help`` for the staged CLI.
Data preparation is separate from training so no GPU is needed for downloads.
"""

import argparse
import hashlib
import json
import time
from datetime import timedelta
from pathlib import Path

import lightning as pl
import torch
from huggingface_hub import snapshot_download
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.classification import MulticlassAccuracy
from torchvision.datasets import CIFAR10, FakeData
from torchvision.transforms.functional import InterpolationMode

import stable_pretraining as spt


def _raw_datasets(args):
    if args.dataset == "fake":
        return (
            FakeData(32, (3, 32, 32), 2),
            FakeData(16, (3, 32, 32), 2, random_offset=1000),
            2,
        )
    if args.dataset == "cifar10":
        return (
            CIFAR10(args.data_dir, train=True, download=args.command == "prepare"),
            CIFAR10(args.data_dir, train=False, download=args.command == "prepare"),
            10,
        )
    # Compute nodes may have no internet. Only prepare can fetch source files;
    # training resolves the pinned snapshot locally and loads built-in Parquet.
    snapshot = Path(
        snapshot_download(
            "clane9/imagenet-100",
            repo_type="dataset",
            revision=args.data_revision,
            cache_dir=str(args.data_dir / "hub"),
            allow_patterns=["data/*.parquet", "README.md"],
            local_files_only=args.command != "prepare",
            max_workers=2,
        )
    )
    files = {
        split: [
            str(path) for path in sorted((snapshot / "data").glob(f"{split}-*.parquet"))
        ]
        for split in ("train", "validation")
    }
    if not all(files.values()):
        raise ValueError("dataset snapshot is incomplete; run prepare before training")
    splits = [
        spt.data.HFDataset(
            "parquet",
            data_files=files,
            split=split,
            cache_dir=str(args.data_dir / "arrow"),
        )
        for split in ("train", "validation")
    ]
    for split in splits:
        if split.dataset.features["label"].num_classes != 100:
            raise ValueError("expected the reference ImageNet-100 label mapping")
    if (
        splits[0].dataset.features["label"].names
        != splits[1].dataset.features["label"].names
    ):
        raise ValueError("training and validation class mappings disagree")
    return *splits, 100


def _wrap(raw, transform, dataset):
    if dataset == "imagenet100":
        raw.transform = transform
        return raw
    return spt.data.FromTorchDataset(raw, names=["image", "label"], transform=transform)


def _transforms(dataset, recipe="legacy"):
    if recipe not in ("legacy", "reference"):
        raise ValueError("unknown augmentation recipe")
    if recipe == "reference" and dataset != "imagenet100":
        raise ValueError("reference augmentation requires imagenet100")
    t = spt.data.transforms
    small = dataset != "imagenet100"
    size = 32 if small else 224
    statistics = spt.data.static.CIFAR10 if small else spt.data.static.ImageNet
    steps = [
        t.RGB(),
        t.RandomResizedCrop((size, size), scale=(0.08, 1.0)),
        t.RandomHorizontalFlip(),
        t.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        t.RandomGrayscale(p=0.2),
    ]
    if not small:
        steps.append(t.PILGaussianBlur(p=0.5))
    augment = t.Compose(*steps, t.ToImage(**statistics))
    evaluate = (
        t.Compose(t.RGB(), t.Resize((size, size)), t.ToImage(**statistics))
        if small
        else t.Compose(
            t.RGB(), t.Resize(256), t.CenterCrop((224, 224)), t.ToImage(**statistics)
        )
    )
    views = [augment, augment]
    if recipe == "reference":
        views = []
        for blur_probability, solar_probability in ((1.0, 0.0), (0.1, 0.2)):
            views.append(
                t.Compose(
                    t.RGB(),
                    t.RandomResizedCrop(
                        (224, 224), interpolation=InterpolationMode.BICUBIC
                    ),
                    t.RandomHorizontalFlip(),
                    t.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                    ),
                    t.RandomGrayscale(p=0.2),
                    t.PILGaussianBlur(p=blur_probability),
                    t.RandomSolarize(threshold=128, p=solar_probability),
                    t.ToImage(**statistics),
                )
            )
    return t.MultiViewTransform(views), evaluate


def _loader(dataset, args, train=False, batch_size=None, drop_last=None):
    devices = getattr(args, "devices", 1)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=devices,
            rank=spt.utils.get_rank(),
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        if train and devices > 1
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size or args.batch_size // devices,
        shuffle=train and sampler is None,
        sampler=sampler,
        drop_last=train if drop_last is None else drop_last,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=args.accelerator == "gpu",
    )


def _backbone(args):
    backbone = spt.backbone.from_torchvision(
        args.backbone,
        low_resolution=args.dataset != "imagenet100",
        weights=None,
        zero_init_residual=True,
    )
    backbone.fc = nn.Identity()
    return backbone, 2048 if args.backbone == "resnet50" else 512


def _linear_contrastive(self, batch: dict, stage: str) -> dict[str, torch.Tensor]:
    if "views" not in batch:
        return {"embedding": self.backbone(batch["image"]), "label": batch["label"]}
    embeddings = [self.backbone(view["image"]) for view in batch["views"]]
    projections = [self.projector(embedding) for embedding in embeddings]
    loss = self.directclr_loss(*projections)
    self.log(
        f"{stage}/loss",
        loss,
        on_step=True,
        on_epoch=True,
        batch_size=embeddings[0].shape[0],
    )
    return {
        "embedding": torch.cat(embeddings),
        "loss": loss,
        "label": torch.cat([view["label"] for view in batch["views"]]),
    }


def _configure(args):
    torch.set_num_threads(args.cpu_threads)
    pl.seed_everything(args.seed, workers=True)
    spt.set(
        cache_dir=str((args.output / "registry").resolve()),
        requeue_checkpoint=False,
        verbose="WARNING",
        default_callbacks={
            "env_dump": False,
            "trainer_info": False,
            "module_summary": False,
            "slurm_info": False,
            "hf_checkpoint": False,
        },
    )
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; refuse to silently run a GPU job on CPU"
        )


@spt.utils.rank_zero_only
def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _data_manifest(train_set, val_set, args):
    return {
        "dataset": args.dataset,
        "data_revision": args.data_revision,
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "train_fingerprint": getattr(
            getattr(train_set, "dataset", None), "_fingerprint", None
        ),
        "validation_fingerprint": getattr(
            getattr(val_set, "dataset", None), "_fingerprint", None
        ),
    }


def _reserve_output(args):
    if spt.utils.get_rank() != 0:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    path = args.output / "config.json"
    if path.exists():
        previous = json.loads(path.read_text())
        ignored = {"resume", "max_steps", "workers", "accelerator", "cpu_threads"}
        if not args.resume or any(
            previous.get(k, {"devices": 1, "augmentation": "legacy"}.get(k)) != v
            for k, v in config.items()
            if k not in ignored
        ):
            raise ValueError(
                "output already contains a run; use a new directory or resume with the same configuration"
            )
    else:
        _save_json(path, config)


class _Throughput(Callback):
    """Measure completed updates after a short startup warmup on every rank."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.start_time = None
        self.start_step = None
        self.initial_step = 0

    def on_train_start(self, trainer, pl_module):
        self.initial_step = trainer.global_step

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.start_time is None and trainer.global_step >= 20:
            if pl_module.device.type == "cuda":
                torch.cuda.synchronize(pl_module.device)
            self.start_time = time.perf_counter()
            self.start_step = trainer.global_step

    def on_train_end(self, trainer, pl_module):
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
        elapsed = time.perf_counter() - self.start_time if self.start_time else 0
        updates = trainer.global_step - self.start_step if self.start_time else 0
        peak = (
            torch.cuda.max_memory_allocated(pl_module.device)
            if pl_module.device.type == "cuda"
            else 0
        )
        values = torch.tensor(
            [elapsed, updates, peak], dtype=torch.float64, device=pl_module.device
        )
        gathered = spt.utils.all_gather(values.unsqueeze(0))
        rows = torch.cat(gathered).cpu().tolist()
        slowest, updates = max(row[0] for row in rows), min(int(row[1]) for row in rows)
        _save_json(
            self.args.output / "throughput.json",
            {
                "devices": getattr(self.args, "devices", 1),
                "global_batch_size": self.args.batch_size,
                "per_device_batch_size": self.args.batch_size
                // getattr(self.args, "devices", 1),
                "measured_updates": updates,
                "seconds_slowest_rank": slowest,
                "global_samples_per_second": updates * self.args.batch_size / slowest
                if slowest
                else None,
                "peak_gpu_memory_bytes_per_rank": [int(row[2]) for row in rows],
                "includes_validation_and_checkpoint_time": True,
            },
        )


def train(args: argparse.Namespace) -> pl.Trainer:
    """Run a fixed-schedule SSL experiment, optionally ending early for profiling.

    Args:
        args: Parsed benchmark options. ``max_steps`` limits a pilot without
            changing the epoch-based learning-rate schedule of the formal run.

    Returns:
        Trainer after Manager completes the requested steps.
    """
    devices = getattr(args, "devices", 1)
    if devices < 1 or args.batch_size % devices or args.batch_size // devices < 2:
        raise ValueError(
            "global batch size must divide evenly across devices, with at least two samples per device"
        )
    if devices > 1 and args.accelerator != "gpu":
        raise ValueError("multi-device training requires CUDA for SyncBatchNorm")
    augmentation = getattr(args, "augmentation", "legacy")
    if augmentation == "reference" and args.dataset != "imagenet100":
        raise ValueError("reference augmentation requires imagenet100")
    _reserve_output(args)
    _configure(args)
    raw_train, raw_val, classes = _raw_datasets(args)
    _save_json(
        args.output / "data_manifest.json", _data_manifest(raw_train, raw_val, args)
    )
    augment, evaluate = _transforms(args.dataset, augmentation)
    train_set, val_set = (
        _wrap(raw_train, augment, args.dataset),
        _wrap(raw_val, evaluate, args.dataset),
    )
    if len(train_set) < args.batch_size or args.batch_size < 2:
        raise ValueError(
            "training requires at least one full batch with two or more samples"
        )
    backbone, width = _backbone(args)
    if not 1 <= args.dim <= width:
        raise ValueError("dim must be between 1 and the backbone feature width")
    steps_per_epoch = len(train_set) // args.batch_size
    optim = {
        "optimizer": {
            "type": "LARS",
            "lr": args.lr * args.batch_size / 256,
            "weight_decay": 1e-6,
            "momentum": 0.9,
            "eta": 1e-3,
            "exclude_bias_n_norm": True,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealingLR",
            "warmup_steps": min(10, args.epochs // 10) * steps_per_epoch,
            "max_steps": args.epochs * steps_per_epoch,
        },
        "interval": "step",
    }
    metadata = {
        "dataset": args.dataset,
        "backbone_name": args.backbone,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "benchmark_method": args.method,
        "devices": devices,
        "augmentation": augmentation,
        "contrastive_loss_dim": width if args.method == "full" else args.dim,
        "trainable_projector": args.method == "linear",
    }
    if args.method == "linear":
        module = spt.Module(
            forward=_linear_contrastive,
            backbone=backbone,
            projector=nn.Linear(width, args.dim, bias=False),
            directclr_loss=spt.losses.DirectCLRLoss(args.temperature),
            optim=optim,
            hparams=metadata,
        )
    else:
        dimension = width if args.method == "full" else args.dim
        module = spt.Module(
            forward=spt.forward.directclr,
            backbone=backbone,
            subspace_dim=dimension,
            directclr_loss=spt.losses.DirectCLRLoss(args.temperature),
            optim=optim,
            hparams={
                **metadata,
                "method": "directclr",
                "subspace_dim": dimension,
                "temperature": args.temperature,
            },
        )
    probe = spt.OnlineProbe(
        module,
        name="probe",
        input="embedding",
        target="label",
        probe=nn.Linear(width, classes),
        loss=nn.CrossEntropyLoss(),
        optimizer={"type": "SGD", "lr": 0.1, "momentum": 0.9},
        metrics={"top1": MulticlassAccuracy(classes, average="micro")},
        verbose=False,
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(args.output / "checkpoints"),
        filename="epoch={epoch}-step={step}",
        auto_insert_metric_name=False,
        save_last=True,
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    throughput = _Throughput(args)
    planned_steps = args.epochs * steps_per_epoch
    # Restored epoch-end loop counters can advance through empty epochs.
    # The optimizer-update budget defines both training length and cosine decay.
    stop_step = (
        min(args.max_steps, planned_steps) if args.max_steps > 0 else planned_steps
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=devices,
        strategy=DDPStrategy(
            timeout=timedelta(seconds=180), gradient_as_bucket_view=True
        )
        if devices > 1
        else "auto",
        sync_batchnorm=devices > 1,
        max_epochs=-1,
        max_steps=stop_step,
        precision=args.precision,
        callbacks=[
            probe,
            checkpoint,
            throughput,
            EarlyStopping(
                monitor="fit/loss_epoch",
                mode="min",
                patience=args.epochs + 1,
                strict=False,
                check_finite=True,
                check_on_train_epoch_end=True,
            ),
        ],
        logger=CSVLogger(str(args.output), name="metrics", version="pretrain"),
        num_sanity_val_steps=0,
        check_val_every_n_epoch=args.val_every,
        log_every_n_steps=10,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(args.output),
    )
    start = time.perf_counter()
    spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(
            train=_loader(train_set, args, True), val=_loader(val_set, args)
        ),
        seed=args.seed,
        ckpt_path=str(args.resume.resolve()) if args.resume else None,
        weights_only=False,
    )()
    elapsed = time.perf_counter() - start
    steps_per_epoch = len(train_set) // args.batch_size
    _save_json(
        args.output / "training_summary.json",
        {
            "global_step": trainer.global_step,
            "elapsed_seconds": elapsed,
            "updates_this_invocation": trainer.global_step - throughput.initial_step,
            "seconds_per_step_including_setup_and_eval": elapsed
            / max(trainer.global_step - throughput.initial_step, 1),
            "steps_per_epoch": steps_per_epoch,
            "planned_total_steps": steps_per_epoch * args.epochs,
            "checkpoint": checkpoint.last_model_path,
            "complete_schedule": trainer.global_step >= steps_per_epoch * args.epochs,
            "devices": devices,
            "global_batch_size": args.batch_size,
            "per_device_batch_size": args.batch_size // devices,
            "augmentation": augmentation,
            "benchmark_method": args.method,
            "contrastive_loss_dim": metadata["contrastive_loss_dim"],
            "trainable_projector": metadata["trainable_projector"],
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
            if args.accelerator == "gpu"
            else None,
        },
    )
    return trainer


@torch.inference_mode()
def _extract(backbone, dataset, args):
    device = torch.device("cuda" if args.accelerator == "gpu" else "cpu")
    backbone.to(device).eval()
    features, labels = [], []
    for batch in _loader(dataset, args):
        features.append(backbone(batch["image"].to(device)).float().cpu())
        labels.append(batch["label"].cpu())
    return torch.cat(features), torch.cat(labels)


def _fit_linear_classifier(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    classes: int,
    args: argparse.Namespace,
    manifest: dict[str, object],
) -> spt.Module:
    """Fit a linear classifier using the shared frozen-feature protocol.

    Args:
        x_train: Frozen backbone features used for classifier fitting.
        y_train: Labels aligned with the training feature rows.
        classes: Number of classification targets.
        args: Probe optimization and runtime options.
        manifest: Encoder and dataset provenance for the run registry.

    Returns:
        Module containing the fitted classifier. No evaluation data is used.
    """
    width = x_train.shape[1]
    module = spt.Module(
        forward=spt.forward.supervised,
        backbone=nn.Identity(),
        classifier=nn.Linear(width, classes),
        supervised_loss=nn.CrossEntropyLoss(),
        optim={
            "optimizer": {"type": "SGD", "lr": args.probe_lr, "momentum": 0.9},
            "scheduler": {
                "type": "CosineAnnealingLR",
                "T_max": args.probe_epochs
                * ((len(x_train) + args.probe_batch_size - 1) // args.probe_batch_size),
            },
            "interval": "step",
        },
        hparams={
            **manifest,
            "protocol": "frozen_linear_l2_features",
            "probe_epochs": args.probe_epochs,
            "probe_lr": args.probe_lr,
        },
    )
    train_set = spt.data.FromTorchDataset(
        TensorDataset(nn.functional.normalize(x_train, dim=1), y_train),
        names=["image", "label"],
    )
    # No held-out set enters the probe training loop; final evaluation is once.
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        precision="32-true",
        max_epochs=args.probe_epochs,
        logger=CSVLogger(str(args.output), name="metrics", version="linear"),
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=20,
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(
            train=_loader(train_set, args, True, args.probe_batch_size, drop_last=False)
        ),
        seed=args.seed,
    )()
    return module


def evaluate(args: argparse.Namespace) -> None:
    """Fit a new linear classifier on cached frozen-backbone training features.

    Args:
        args: Evaluation options, including a trusted checkpoint from this
            benchmark or ``random`` for a random-encoder control. The recipe
            is fixed in advance; held-out labels do not select hyperparameters.
    """
    if getattr(args, "devices", 1) != 1:
        raise ValueError("frozen feature extraction and evaluation require devices=1")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "evaluation.json").exists():
        raise ValueError("evaluation already exists; use a new output directory")
    _configure(args)
    raw_train, raw_val, classes = _raw_datasets(args)
    _, transform = _transforms(args.dataset)
    backbone, width = _backbone(args)
    fingerprint = "random"
    if args.checkpoint:
        with args.checkpoint.open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = {
            k.removeprefix("backbone."): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("backbone.")
        }
        backbone.load_state_dict(state, strict=True)
    cache_file = args.output / "features.pt"
    manifest = {
        "checkpoint_sha256": fingerprint,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "seed": args.seed,
        **_data_manifest(raw_train, raw_val, args),
    }
    if cache_file.exists():
        if json.loads((args.output / "feature_manifest.json").read_text()) != manifest:
            raise ValueError("cached features do not match this encoder or dataset")
        features = torch.load(cache_file, weights_only=True)
    else:
        x_train, y_train = _extract(
            backbone, _wrap(raw_train, transform, args.dataset), args
        )
        x_val, y_val = _extract(backbone, _wrap(raw_val, transform, args.dataset), args)
        features = {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
        }
        torch.save(features, cache_file)
        _save_json(args.output / "feature_manifest.json", manifest)
    if getattr(args, "cache_only", False):
        return
    del backbone
    if args.accelerator == "gpu":
        torch.cuda.empty_cache()
    x_train, y_train, x_val, y_val = (
        features[k] for k in ("x_train", "y_train", "x_val", "y_val")
    )
    # Class-sorted sources bias a leading-row subset. Use one fixed random
    # subset across all encoders and save its indices with the spectra.
    spectral_indices = torch.randperm(
        len(x_train), generator=torch.Generator().manual_seed(0)
    )[: min(4096, len(x_train))]
    spectral = x_train[spectral_indices].double()
    singular = torch.linalg.svdvals(spectral)
    probabilities = singular / singular.sum().clamp_min(1e-12)
    rankme = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    ).item()
    torch.save(
        {
            "sample_indices": spectral_indices,
            "singular_values": singular,
            "centered_singular_values": torch.linalg.svdvals(
                spectral - spectral.mean(0)
            ),
        },
        args.output / "spectrum.pt",
    )
    module = _fit_linear_classifier(x_train, y_train, classes, args, manifest)
    device = module.device
    correct1 = correct5 = count = 0
    module.eval()
    with torch.inference_mode():
        for x, y in DataLoader(
            TensorDataset(x_val, y_val), batch_size=args.probe_batch_size
        ):
            logits = module.classifier(nn.functional.normalize(x.to(device), dim=1))
            prediction = logits.topk(min(5, classes), dim=1).indices.cpu()
            correct1 += (prediction[:, 0] == y).sum().item()
            correct5 += (prediction == y[:, None]).any(1).sum().item()
            count += y.numel()
    _save_json(
        args.output / "evaluation.json",
        {
            **manifest,
            "linear_top1": correct1 / count,
            "linear_top5": correct5 / count,
            "rankme": rankme,
            "spectral_samples": len(spectral),
            "train_samples": len(x_train),
            "eval_samples": count,
            "probe_epochs": args.probe_epochs,
            "probe_lr": args.probe_lr,
            "protocol": "frozen_encoder_l2_features_fixed_lr_final_epoch_no_eval_selection",
        },
    )
    torch.save(module.classifier.state_dict(), args.output / "linear_classifier.pt")


def main() -> None:
    """Dispatch data preparation, SSL training, or frozen linear evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "evaluate"])
    parser.add_argument(
        "--dataset", choices=["imagenet100", "cifar10", "fake"], default="imagenet100"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/directclr"))
    parser.add_argument(
        "--data-revision",
        default="main",
        help="Pin the dataset commit for recorded experiments",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/directclr"))
    parser.add_argument(
        "--method",
        choices=["directclr", "linear", "full"],
        default="directclr",
        help=(
            "directclr: fixed leading subspace; linear: trainable linear projector; "
            "full: no projector, contrastive loss on all encoder coordinates"
        ),
    )
    parser.add_argument(
        "--backbone", choices=["resnet50", "resnet18"], default="resnet50"
    )
    parser.add_argument("--dim", type=int, default=360)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Extract frozen features without fitting or scoring a classifier",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--lr", type=float, default=0.3, help="LARS learning rate at batch size 256"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Training GPU count; batch-size is global",
    )
    parser.add_argument(
        "--augmentation",
        choices=["legacy", "reference"],
        default="legacy",
        help="Preserve the initial recipe or use the reference's asymmetric ImageNet views",
    )
    parser.add_argument(
        "--precision", choices=["32-true", "16-mixed", "bf16-mixed"], default="16-mixed"
    )
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Trusted local checkpoint; omit for random encoder control",
    )
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-lr", type=float, default=1.0)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.probe_epochs < 1
        or args.workers < 0
        or args.cpu_threads < 1
        or args.val_every < 1
    ):
        parser.error(
            "epoch counts, cpu-threads and val-every must be positive; workers must be nonnegative"
        )
    if args.command == "prepare":
        train_set, val_set, classes = _raw_datasets(args)
        print(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "train": len(train_set),
                    "validation": len(val_set),
                    "classes": classes,
                }
            )
        )
    elif args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
