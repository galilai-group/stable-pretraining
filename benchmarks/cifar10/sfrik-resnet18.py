"""Reproducible small-data SFRIK validation, not an ImageNet reproduction.

Run ``python benchmarks/cifar10/sfrik-resnet18.py --help`` for options.
Uses the public CIFAR-10 training split only: disjoint, seeded subsets for
SSL/probe training and held-out evaluation. The test split is not used.
"""

import argparse
import json
import time
from pathlib import Path

import datasets
import lightning as pl
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import stable_pretraining as spt


class CIFARViews(Dataset):
    """Produce independently augmented views or deterministic probe inputs."""

    def __init__(self, records, paired: bool):
        self.records = records
        self.paired = paired
        operations = []
        if paired:
            operations = [
                T.RandomResizedCrop(32, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
            ]
        self.transform = T.Compose(
            operations
            + [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616)),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        item = self.records[index]
        if self.paired:
            return {"views": [{"image": self.transform(item["img"])} for _ in range(2)]}
        return {"image": self.transform(item["img"]), "label": item["label"]}


class FiniteTraining(pl.Callback):
    """Reject nonfinite loss/gradients and record epoch-level progress."""

    def __init__(self):
        self.history = []
        self.losses = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.losses = []

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        for parameter in pl_module.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError("Nonfinite training gradient")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"].detach().item()
        if not torch.isfinite(torch.tensor(loss)):
            raise RuntimeError("Nonfinite training loss")
        self.losses.append(loss)

    def on_train_epoch_end(self, trainer, pl_module):
        row = {
            "epoch": trainer.current_epoch + 1,
            "loss": sum(self.losses) / len(self.losses),
        }
        self.history.append(row)
        print(json.dumps(row), flush=True)


@torch.no_grad()
def extract(
    backbone: nn.Module, loader: DataLoader, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract frozen backbone features with identical preprocessing each time."""
    backbone.to(device).eval()
    features, labels = [], []
    for batch in loader:
        features.append(backbone(batch["image"].to(device)).cpu())
        labels.append(batch["label"])
    return torch.cat(features), torch.cat(labels)


def evaluate(
    backbone: nn.Module, reference: DataLoader, validation: DataLoader, device: str
) -> dict:
    """Fit a fresh linear probe and kNN on frozen representations."""
    x, y = extract(backbone, reference, device)
    v, target = extract(backbone, validation, device)
    x, v = nn.functional.normalize(x, dim=1), nn.functional.normalize(v, dim=1)
    probe = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    probe.fit(x.numpy(), y.numpy())
    predictions = []
    for chunk in v.split(256):
        neighbors = (chunk @ x.T).topk(min(20, len(x)), dim=1).indices
        predictions.append(nn.functional.one_hot(y[neighbors], 10).sum(1).argmax(1))
    spectrum = torch.linalg.svdvals(x - x.mean(0)).double()
    probabilities = spectrum / spectrum.sum().clamp_min(1e-12)
    rank = (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).exp()
    return {
        "linear_accuracy": float(probe.score(v.numpy(), target.numpy())),
        "knn20_accuracy": float((torch.cat(predictions) == target).float().mean()),
        "effective_rank": float(rank),
        "mean_feature_std": float(x.std(0).mean()),
    }


def main() -> None:
    """Train via Manager and save reproducible before/after measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/huggingface"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accelerator", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--method", choices=["sfrik", "simclr"], default="sfrik")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--projector-dim", type=int, default=512)
    parser.add_argument("--sim-coeff", type=float, default=256.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.train_size < args.batch_size or args.val_size < 1:
        parser.error(
            "Need positive epochs/validation size and at least one full training batch"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(4)
    spt.set(cache_dir=str(args.output_dir.resolve() / "registry"))
    records = datasets.load_dataset(
        "uoft-cs/cifar10", split="train", cache_dir=str(args.data_dir)
    )
    if args.train_size + args.val_size > len(records):
        parser.error(
            "Training and validation subsets must fit disjointly in the training split"
        )
    indices = torch.randperm(len(records)).tolist()
    train_indices = indices[: args.train_size]
    val_indices = indices[args.train_size : args.train_size + args.val_size]
    train_records, val_records = (
        records.select(train_indices),
        records.select(val_indices),
    )
    loader_options = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    train = DataLoader(
        CIFARViews(train_records, True), shuffle=True, drop_last=True, **loader_options
    )
    reference = DataLoader(CIFARViews(train_records, False), **loader_options)
    validation = DataLoader(CIFARViews(val_records, False), **loader_options)
    backbone = spt.backbone.from_torchvision(
        "resnet18", low_resolution=True, weights=None
    )
    backbone.fc = nn.Identity()
    projector = nn.Sequential(
        nn.Linear(512, args.projector_dim),
        nn.BatchNorm1d(args.projector_dim),
        nn.ReLU(),
        nn.Linear(args.projector_dim, args.projector_dim),
        nn.BatchNorm1d(args.projector_dim),
        nn.ReLU(),
        nn.Linear(args.projector_dim, args.projector_dim, bias=False),
    )
    optim = {
        "optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": 1e-4},
        "scheduler": {"type": "CosineAnnealingLR", "T_max": args.epochs},
        "interval": "epoch",
    }
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    if args.method == "sfrik":
        module = spt.SFRIK(
            backbone=backbone,
            projector=projector,
            sim_coeff=args.sim_coeff,
            optim=optim,
            hparams=config,
        )
    else:
        module = spt.Module(
            backbone=backbone,
            projector=projector,
            forward=spt.forward.simclr,
            simclr_loss=spt.losses.NTXEntLoss(temperature=0.5),
            optim=optim,
            hparams=config,
        )
    report = {
        "config": config,
        "torch": torch.__version__,
        "lightning": pl.__version__,
        "train_indices": train_indices,
        "validation_indices": val_indices,
    }
    report["initial"] = evaluate(backbone, reference, validation, args.accelerator)
    print("Initial evaluation: " + json.dumps(report["initial"]), flush=True)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2))
    check = FiniteTraining()
    trainer = pl.Trainer(
        accelerator="gpu" if args.accelerator == "cuda" else args.accelerator,
        devices=1,
        precision="32-true",
        max_epochs=args.epochs,
        callbacks=[check],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        log_every_n_steps=10,
        default_root_dir=str(args.output_dir),
    )
    start = time.perf_counter()
    spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train),
        seed=args.seed,
    )()
    report["training_seconds"] = time.perf_counter() - start
    report["steps"] = trainer.global_step
    report["history"] = check.history
    trainer.save_checkpoint(args.output_dir / "last.ckpt")
    report["final"] = evaluate(backbone, reference, validation, args.accelerator)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2))
    print("Final evaluation: " + json.dumps(report["final"]), flush=True)


if __name__ == "__main__":
    main()
