"""
LFR on Handwritten Digits
=========================

Small CPU validation of LFR on scikit-learn's real handwritten digits. This is
an independent known-data check, not a reproduction of the paper's CIFAR-10
schedule::

    python examples/lfr_digits.py --output-dir ./lfr-digits-run
"""

import argparse
import copy
import json
from functools import partial
from pathlib import Path

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

import stable_pretraining as spt


class Digits(Dataset):
    """Unlabeled images for LFR pretraining."""

    def __init__(self, data: torch.Tensor) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self.data[index]}


@torch.no_grad()
def encode(encoder: nn.Module, data: torch.Tensor) -> np.ndarray:
    """Get frozen features for linear evaluation."""
    encoder.eval()
    return encoder(data).detach().numpy()


def probe(
    encoder: nn.Module,
    train_data: torch.Tensor,
    test_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    seed: int,
) -> float:
    """Train a logistic-regression probe and return held-out accuracy."""
    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(encode(encoder, train_data), train_labels.numpy())
    return float(classifier.score(encode(encoder, test_data), test_labels.numpy()))


def run(args: argparse.Namespace, seed: int, inputs, labels) -> dict[str, float]:
    """Pretrain on one split without labels and probe both encoders."""
    pl.seed_everything(seed, workers=True)
    train_idx, test_idx = train_test_split(
        np.arange(len(inputs)),
        test_size=0.2,
        random_state=seed,
        stratify=labels.numpy(),
    )
    encoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32))
    initial_encoder = copy.deepcopy(encoder)
    model = spt.LFR(
        encoder_name=encoder,
        projectors=[
            nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32))
            for _ in range(3)
        ],
        predictors=[nn.Linear(32, 32) for _ in range(3)],
        predictor_epochs=args.predictor_epochs,
        optim={"optimizer": partial(torch.optim.Adam, lr=1e-3)},
    )
    loader = DataLoader(
        Digits(inputs[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    trainer = pl.Trainer(
        max_epochs=model.max_epochs_for(args.cycles),
        accelerator="cpu",
        devices=1,
        logger=CSVLogger(str(args.output_dir), name=f"lfr_digits_seed{seed}"),
        default_root_dir=str(args.output_dir),
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        log_every_n_steps=1,
    )
    trainer.fit(model, loader)
    split = (inputs[train_idx], inputs[test_idx], labels[train_idx], labels[test_idx])
    return {
        "seed": seed,
        "random_test_accuracy": probe(initial_encoder, *split, seed),
        "lfr_test_accuracy": probe(model.backbone, *split, seed),
    }


def main() -> None:
    """Pretrain on digits without labels and compare to the initial encoder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--predictor-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    if min(args.cycles, args.predictor_epochs) < 1 or args.batch_size < 2:
        parser.error("counts must be positive and batch size at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spt.set(cache_dir=str(args.output_dir.resolve()))
    data = load_digits()
    inputs = torch.tensor(data.data, dtype=torch.float32) / 16
    labels = torch.tensor(data.target, dtype=torch.long)
    runs = [run(args, seed, inputs, labels) for seed in args.seeds]
    summary = {
        "dataset": "scikit-learn digits (1,437 train / 360 test per split)",
        "cycles": args.cycles,
        "predictor_epochs_per_cycle": args.predictor_epochs,
        "batch_size": args.batch_size,
        "runs": runs,
    }
    for key in ("random_test_accuracy", "lfr_test_accuracy"):
        values = np.array([r[key] for r in runs])
        summary[key] = {"mean": values.mean(), "std": values.std(ddof=1)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
