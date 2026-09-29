"""Run a small SSL experiment: python -m stable_pretraining.quickstart.

Uses synthetic images by default; this verifies the training pipeline, not
representation quality. Pass --data with train/ and val/ ImageFolder splits
to use real images. All training and checkpoint lifecycle belongs to Manager.
"""

import argparse
from pathlib import Path

import lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision.datasets import FakeData, ImageFolder

import stable_pretraining as spt


def _jet_entropy(self, batch: dict, stage: str) -> dict[str, torch.Tensor]:
    if "views" not in batch:
        tokens, logdet = self.backbone(batch["image"])
        return {"embedding": tokens.mean(1), "logdet": logdet}
    views = batch["views"]
    images = torch.cat([view["image"] for view in views])
    tokens, logdet = self.backbone(images)
    a, b = tokens.mean(1).chunk(2)
    prediction = nn.functional.mse_loss(a, b)
    entropy_gain = (logdet / images[0].numel()).mean()
    loss = prediction - self.entropy_weight * entropy_gain
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite Jet prediction/entropy loss")
    with torch.no_grad():
        energy = tokens.square().mean()
        semantic = tokens.mean(1).square().mean()
        residual = (tokens - tokens.mean(1, keepdim=True)).square().mean()
    self.log_dict(
        {
            "fit/loss": loss,
            "fit/mse": prediction,
            "flow/entropy_gain": entropy_gain,
            "tokens/energy": energy,
            "tokens/semantic_energy": semantic,
            "tokens/residual_energy": residual,
            **self.backbone.diagnostics,
        },
        batch_size=images.shape[0],
    )
    return {
        "loss": loss,
        "embedding": torch.cat((a, b)),
        "label": torch.cat([view["label"] for view in views]),
        "logdet": logdet,
    }


def _datasets(root: Path | None):
    if root is None:
        train = FakeData(size=32, image_size=(3, 16, 16), num_classes=2)
        val = FakeData(
            size=16, image_size=(3, 16, 16), num_classes=2, random_offset=1000
        )
        classes = 2
    else:
        train, val = ImageFolder(root / "train"), ImageFolder(root / "val")
        if train.class_to_idx != val.class_to_idx:
            raise ValueError("train/ and val/ must contain the same class directories")
        if len(train) < 8:
            raise ValueError("The quickstart needs at least eight training images")
        classes = len(train.classes)
        if classes < 2:
            raise ValueError("The classification probe needs at least two classes")
    transforms = spt.data.transforms
    augment = transforms.Compose(
        transforms.RandomResizedCrop((16, 16)),
        transforms.RandomHorizontalFlip(),
        transforms.ToImage(scale=True),
    )
    train = spt.data.FromTorchDataset(
        train,
        names=["image", "label"],
        transform=transforms.MultiViewTransform([augment, augment]),
    )
    val = spt.data.FromTorchDataset(
        val,
        names=["image", "label"],
        transform=transforms.Compose(
            transforms.Resize((16, 16)), transforms.ToImage(scale=True)
        ),
    )
    return train, val, classes


def _run(args: argparse.Namespace) -> pl.Trainer:
    if args.epochs < 1 or args.steps == 0 or args.steps < -1:
        raise ValueError("epochs must be positive; steps must be positive or -1")
    torch.set_num_threads(1)
    pl.seed_everything(7, workers=True)
    spt.set(
        cache_dir=str(args.cache_dir.resolve()),
        requeue_checkpoint=False,
        default_callbacks={
            "env_dump": False,
            "trainer_info": False,
            "module_summary": False,
            "slurm_info": False,
        },
    )
    train, val, classes = _datasets(args.data)
    if args.method == "jet-entropy":
        backbone = spt.backbone.Jet(
            image_size=16,
            patch_size=2,
            coupling_layers=2,
            hidden_dim=16,
            depth=1,
            num_heads=2,
            capture_stats=True,
            scale_parameterization="exp_floor",
            scale_eps=1e-4,
        )
        module = spt.Module(
            forward=_jet_entropy,
            backbone=backbone,
            entropy_weight=0.01,
            hparams={
                "method": "jet-entropy",
                "scale_parameterization": "exp_floor",
                "scale_eps": 1e-4,
                "entropy_weight": 0.01,
            },
            optim={"optimizer": {"type": "AdamW", "lr": 1e-3}},
        )
        embedding_dim = backbone.patch_dim
    else:
        embedding_dim = 16
        backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        module = spt.Module(
            forward=spt.forward.simclr,
            hparams={"method": "simclr", "temperature": 0.5},
            backbone=backbone,
            projector=nn.Linear(16, 8),
            simclr_loss=spt.losses.NTXEntLoss(temperature=0.5),
            optim={"optimizer": {"type": "AdamW", "lr": 1e-3}},
        )
    probe = spt.OnlineProbe(
        module,
        name="probe",
        input="embedding",
        target="label",
        probe=nn.Linear(embedding_dim, classes),
        loss=nn.CrossEntropyLoss(),
        optimizer={"type": "AdamW", "lr": 1e-3},
        metrics={"accuracy": MulticlassAccuracy(classes)},
        verbose=False,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=args.epochs,
        max_steps=args.steps,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        callbacks=[probe],
    )
    data = spt.data.DataModule(
        train=DataLoader(train, batch_size=8, drop_last=True),
        val=DataLoader(val, batch_size=8),
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=data,
        seed=7,
        ckpt_path=str(args.resume.resolve()) if args.resume else None,
        weights_only=False,
    )()
    print(
        f"Completed {trainer.global_step} optimizer steps. Inspect: spt web {args.cache_dir.resolve() / 'runs'}"
    )
    return trainer


def main() -> None:
    """Parse quickstart options and run through stable_pretraining.Manager."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("simclr", "jet-entropy"), default="simclr")
    parser.add_argument(
        "--data", type=Path, help="ImageFolder root with train/ and val/ splits"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("./spt-quickstart"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--steps", type=int, default=-1, help="Optional optimizer-step limit"
    )
    parser.add_argument("--resume", type=Path, help="Trusted full-state checkpoint")
    _run(parser.parse_args())


if __name__ == "__main__":
    main()
