"""
All4One training and evaluation
========================================

All4One with existing data, optimization, and online evaluation utilities.

Run a small CPU training/evaluation smoke test::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/all4one.py

Train on an already downloaded CIFAR-100 dataset::

    python examples/all4one.py --dataset cifar100 --data-root ./datasets \
        --accelerator gpu --precision 16-mixed --num-workers 4

The training recipe follows the authors' released implementation. See ``docs/source/guides/all4one.rst`` for the audit.
"""

import argparse
import json
import math
from pathlib import Path

import lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets
from torchvision.transforms import InterpolationMode

import stable_pretraining as spt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the dataset and runtime options.

    Args:
        argv: Command-line arguments; None reads the process arguments.

    Returns:
        Validated options with dataset-dependent defaults resolved.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        choices=["fake", "cifar10", "cifar100", "imagenet100"],
        default="fake",
    )
    parser.add_argument("--stage", choices=["pretrain", "linear"], default="pretrain")
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--download", action="store_true", help="Allow torchvision to download CIFAR"
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument(
        "--lr",
        type=float,
        help="Base learning rate at batch size 256 (scaled by batch-size / 256)",
    )
    parser.add_argument(
        "--attention-lr",
        type=float,
        help="Attention learning rate (unscaled, as in the released method)",
    )
    parser.add_argument(
        "--cifar-stem-padding",
        type=int,
        choices=[1, 2],
        default=2,
        help="First CIFAR convolution padding; the authors' released recipe uses 2",
    )
    parser.add_argument(
        "--encoder-path", type=Path, help="Encoder export to load for linear evaluation"
    )
    parser.add_argument(
        "--export-encoder",
        type=Path,
        help="Save the final student encoder after pretraining",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--precision", choices=["32-true", "16-mixed", "bf16-mixed"], default="32-true"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/all4one"))
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        help="Resume a full checkpoint with the same configuration",
    )
    args = parser.parse_args(argv)
    synthetic = args.dataset == "fake"
    linear = args.stage == "linear"
    args.epochs = (
        args.epochs
        if args.epochs is not None
        else (
            2
            if synthetic
            else (100 if linear else (400 if args.dataset == "imagenet100" else 1000))
        )
    )
    args.batch_size = (
        args.batch_size if args.batch_size is not None else (8 if synthetic else 256)
    )
    args.warmup_epochs = (
        args.warmup_epochs
        if args.warmup_epochs is not None
        else (0 if synthetic or linear else min(10, args.epochs - 1))
    )
    args.lr = (
        args.lr
        if args.lr is not None
        else (0.1 if linear else (0.01 if synthetic else 1.0))
    )
    args.attention_lr = (
        args.attention_lr
        if args.attention_lr is not None
        else (0.01 if synthetic else 0.1)
    )
    if args.epochs < 1 or args.batch_size < 2 or args.num_workers < 0:
        parser.error("epochs must be positive, batch-size >= 2, and num-workers >= 0")
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error("warmup-epochs must be nonnegative and smaller than epochs")
    if args.accelerator == "cpu" and args.precision == "16-mixed":
        parser.error("use 32-true or bf16-mixed on CPU")
    if any(not math.isfinite(lr) or lr <= 0 for lr in (args.lr, args.attention_lr)):
        parser.error("learning rates must be finite and positive")
    if linear and args.encoder_path is None:
        parser.error(
            "--stage linear requires --encoder-path from completed pretraining"
        )
    if linear and args.export_encoder is not None:
        parser.error("--export-encoder is only supported for pretraining")
    if not linear and args.encoder_path is not None:
        parser.error(
            "use --ckpt-path to resume pretraining; --encoder-path is for linear evaluation"
        )
    if args.download and args.dataset not in ("cifar10", "cifar100"):
        parser.error(
            "--download only supports CIFAR; ImageNet-100 requires local train/val folders"
        )
    return args


def load_datasets(
    args: argparse.Namespace,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, dict]:
    """Load fixed train/held-out splits without attaching image transforms.

    Args:
        args: Validated command-line options.

    Returns:
        Training dataset, held-out dataset, and the recorded class-to-label mapping.

    Raises:
        ValueError: ImageNet-100 folders do not contain the same 100 classes.
    """
    if args.dataset == "fake":
        return (
            datasets.FakeData(
                size=4 * args.batch_size, image_size=(3, 32, 32), num_classes=10
            ),
            datasets.FakeData(
                size=2 * args.batch_size,
                image_size=(3, 32, 32),
                num_classes=10,
                random_offset=100000,
            ),
            {str(i): i for i in range(10)},
        )
    root = args.data_root.expanduser()
    if args.dataset == "imagenet100":
        train = datasets.ImageFolder(root / "train")
        val = datasets.ImageFolder(root / "val")
        if len(train.classes) != 100 or train.class_to_idx != val.class_to_idx:
            raise ValueError(
                "ImageNet-100 train/val must contain the same 100 class folders"
            )
    else:
        dataset = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
        train = dataset(str(root), train=True, download=args.download)
        val = dataset(str(root), train=False, download=args.download)
    return train, val, train.class_to_idx


def build_encoder(dataset: str) -> tuple[nn.Module, int]:
    """Construct the encoder with solo-learn's torchvision initialization.

    Args:
        dataset: Dataset identifier from the example CLI.

    Returns:
        Encoder and its feature width.
    """
    if dataset == "fake":
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        ), 16
    encoder = spt.backbone.from_torchvision(
        "resnet18",
        low_resolution=dataset != "imagenet100",
        weights=None,
        zero_init_residual=False,
    )
    return encoder, encoder.inplanes


def frozen_forward(
    self: spt.Module, batch: dict, stage: str
) -> dict[str, torch.Tensor]:
    """Extract fixed features for the library's trainable linear-probe callback.

    Args:
        self: Module containing the frozen backbone.
        batch: Images and labels from the fixed dataset splits.
        stage: Module stage supplied by the training loop.

    Returns:
        Detached embeddings and labels; OnlineProbe supplies the training loss.
    """
    # Module.train() also toggles its children; keep BatchNorm buffers fixed.
    self.backbone.eval()
    with torch.no_grad():
        embedding = self.backbone(batch["image"])
    return {"embedding": embedding, "label": batch["label"]}


def export_encoder(model: spt.All4One, config: dict, path: Path) -> None:
    """Save a compact final-epoch student encoder for frozen evaluation.

    Args:
        model: Completed pretraining model.
        config: Dataset and training configuration recorded with the run.
        path: Destination; atomically replaced after serialization succeeds.
    """
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "all4one-example-encoder-v1",
        "dataset": config["dataset"],
        "class_to_idx": config["class_to_idx"],
        "normalization": config["normalization"],
        "config": json.loads(json.dumps(config, default=str)),
        "state_dict": {
            k: v.detach().cpu() for k, v in model.backbone.student.state_dict().items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    print(f"Final student encoder: {path}")


def build_experiment(
    args: argparse.Namespace,
) -> tuple[spt.Module, spt.data.DataModule, list, dict]:
    """Build an experiment using library components, without running training.

    Args:
        args: Options returned by parse_args.

    Returns:
        Model, data module, evaluation callbacks, and an inspectable configuration.
    """
    spt.utils.seed_everything(args.seed, workers=True)
    transforms = spt.data.transforms
    imagenet = args.dataset == "imagenet100"
    linear = args.stage == "linear"
    size = 224 if imagenet else 32
    mean, std = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    }.get(args.dataset, ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    # The released classification loader uses these statistics for both CIFARs.
    eval_mean, eval_std = (
        ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        if args.dataset in ("cifar10", "cifar100")
        else (mean, std)
    )
    payload = None
    if linear:
        payload = torch.load(
            args.encoder_path.expanduser(), map_location="cpu", weights_only=True
        )
        if payload.get("format") != "all4one-example-encoder-v1":
            raise ValueError("encoder-path must be an encoder export from this example")
        # Preserve preprocessing for exports created before this recipe audit.
        normalization = payload["config"].get(
            "evaluation_normalization", payload["normalization"]
        )
        mean, std = tuple(normalization["mean"]), tuple(normalization["std"])
        eval_mean, eval_std = mean, std

    def view(solarize: float, blur: float) -> transforms.Compose:
        return transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(
                size=(size, size),
                scale=(0.08, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=blur),
            transforms.RandomSolarize(threshold=128, p=solarize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(mean=mean, std=std),
        )

    synthetic = args.dataset == "fake"
    train_set, val_set, class_to_idx = load_datasets(args)
    num_classes = len(class_to_idx)
    train_transform = (
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(
                (size, size),
                scale=(0.08, 1.0),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(mean=mean, std=std),
        )
        if linear
        else transforms.MultiViewTransform(
            [view(0.0, 1.0 if imagenet else 0.0), view(0.2, 0.1 if imagenet else 0.0)]
        )
    )
    val_transform = (
        transforms.Compose(
            transforms.RGB(),
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop((224, 224)),
            transforms.ToImage(mean=eval_mean, std=eval_std),
        )
        if imagenet
        else transforms.ToImage(mean=eval_mean, std=eval_std, rgb=True)
    )
    train_set = spt.data.FromTorchDataset(
        train_set,
        names=["image", "label"],
        transform=train_transform,
    )
    val_set = spt.data.FromTorchDataset(
        val_set,
        names=["image", "label"],
        transform=val_transform,
    )
    train = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=not linear,
        num_workers=args.num_workers,
        pin_memory=args.accelerator == "gpu",
    )
    val = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)
    if len(train) == 0:
        raise ValueError("batch-size exceeds the training dataset size")
    total_steps = len(train) * args.epochs
    # This example uses one device and no gradient accumulation.
    lr_scale = args.batch_size / 256 if not synthetic else 1.0
    representation_lr = args.lr * lr_scale
    # The released method fixes this parameter-group LR independently of batch size.
    attention_lr = args.attention_lr
    scheduler = {
        "type": "LinearWarmupCosineAnnealingLR",
        "warmup_steps": len(train) * args.warmup_epochs,
        "max_steps": total_steps,
        "warmup_start_lr": 3e-5,
        "eta_min": 0.0,
    }
    optimizer = (
        {"type": "SGD", "lr": representation_lr, "momentum": 0.9}
        if synthetic
        else {
            "type": "LARS",
            "lr": representation_lr,
            "momentum": 0.9,
            "weight_decay": 1e-5,
            "eta": 0.02,
            "clip_lr": True,
            "exclude_bias_n_norm": True,
            "exclude_bias_norm": True,
        }
    )
    optim = {
        "representation": {
            "modules": r"^(backbone\.student|projector\.student|predictor|centroid_predictor)$",
            "optimizer": optimizer,
            "scheduler": scheduler,
            "interval": "step",
        },
        "attention": {
            "modules": r"^attention$",
            "optimizer": {**optimizer, "lr": attention_lr},
            "scheduler": scheduler,
            "interval": "step",
        },
    }
    encoder, feature_dim = build_encoder(args.dataset)
    if args.dataset in ("cifar10", "cifar100"):
        encoder.conv1.padding = (args.cifar_stem_padding,) * 2
    model_options = dict(backbone=encoder, feature_dim=feature_dim)
    if synthetic:
        model_options.update(
            projection_dim=16,
            projector_hidden_dim=32,
            predictor_hidden_dim=32,
            queue_size=64,
            num_heads=2,
            num_layers=1,
        )
    probe_optimizer = {
        "type": "SGD",
        "lr": representation_lr if linear else 0.1 * lr_scale,
        "momentum": 0.9,
        "weight_decay": 0,
    }
    config = {
        **vars(args),
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "total_steps": total_steps,
        "recipe": "all4one-release-00eeb30-v1",
        "lr_scale": lr_scale,
        "effective_representation_lr": representation_lr,
        "effective_attention_lr": attention_lr,
        "optim": {"probe": probe_optimizer} if linear else optim,
        "class_to_idx": class_to_idx,
        "held_out_split": "validation" if imagenet or synthetic else "test",
        "backbone": "tiny-cnn" if synthetic else "resnet18",
        "crop_size": size,
        "low_resolution": not imagenet,
        "normalization": {"mean": mean, "std": std},
        "evaluation_normalization": {"mean": eval_mean, "std": eval_std},
        "reference_commit": "00eeb30c5e046e9fedf9a5024ba41ab4aaca4c08",
    }
    if linear:
        if (
            payload["dataset"] != args.dataset
            or payload["class_to_idx"] != class_to_idx
        ):
            raise ValueError(
                "Encoder dataset/classes do not match the evaluation dataset"
            )
        if payload["config"].get("cifar_stem_padding", 1) != args.cifar_stem_padding:
            raise ValueError(
                "Encoder stem padding does not match the evaluation encoder"
            )
        encoder.load_state_dict(payload["state_dict"], strict=True)
        encoder.requires_grad_(False).eval()
        config["pretraining"] = payload["config"]
    hparams = {"example": json.loads(json.dumps(config, default=str))}
    if linear:
        model = spt.Module(
            backbone=encoder, forward=frozen_forward, optim=None, hparams=hparams
        )
        model.feature_dim = feature_dim
    else:
        model = spt.All4One(**model_options, optim=optim, hparams=hparams)
    probe = spt.OnlineProbe(
        model,
        name="linear",
        input="embedding",
        target="label",
        probe=nn.Linear(model.feature_dim, num_classes),
        loss=nn.CrossEntropyLoss(),
        optimizer=probe_optimizer,
        scheduler=scheduler,
        metrics={
            "accuracy": MulticlassAccuracy(num_classes, average="micro"),
            "top5": MulticlassAccuracy(num_classes, average="micro", top_k=5),
        },
    )
    if linear:
        return model, spt.data.DataModule(train=train, val=val), [probe], config
    knn = spt.OnlineKNN(
        name="knn",
        input="embedding",
        target="label",
        queue_length=64 if synthetic else 20000,
        input_dim=model.feature_dim,
        num_classes=num_classes,
        k=5 if synthetic else 20,
        distance_metric="cosine",
        chunk_size=256,
        metrics={"accuracy": MulticlassAccuracy(num_classes, average="micro")},
    )
    return model, spt.data.DataModule(train=train, val=val), [probe, knn], config


def main() -> None:
    """Pretrain or fit a frozen-encoder linear probe."""
    args = parse_args()
    model, data, callbacks, config = build_experiment(args)
    print(json.dumps(config, indent=2, default=str))
    spt.set(cache_dir=str(args.cache_dir.expanduser().resolve()))
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        precision=args.precision,
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=False,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=model,
        data=data,
        seed=args.seed,
        ckpt_path=str(args.ckpt_path.expanduser().resolve())
        if args.ckpt_path
        else None,
        weights_only=False,
    )
    manager()
    keys = ["eval/linear_accuracy", "eval/linear_top5"]
    if args.stage == "pretrain":
        keys.append("eval/knn_accuracy")
    for key in keys:
        print(f"{key}: {float(trainer.callback_metrics[key]):.4f}")
    if args.export_encoder:
        if trainer.current_epoch < args.epochs:
            raise RuntimeError(
                "Pretraining ended early; refusing to export an incomplete final encoder"
            )
        export_encoder(model, config, args.export_encoder)
    if args.dataset == "fake":
        print(
            "Synthetic-data metrics verify the evaluation path; they are not benchmark results."
        )


if __name__ == "__main__":
    main()
