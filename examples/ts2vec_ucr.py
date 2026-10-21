"""
TS2Vec on UCR time series
=========================

Train on fixed-length UCR train/test TSV files and evaluate an SVM.

Each row contains an integer label followed by finite time-series values::

    python examples/ts2vec_ucr.py ECG200_TRAIN.tsv ECG200_TEST.tsv --steps 200

Training uses AdamW (lr=0.001, weight_decay=0.01) and batch size 16. Evaluation
max-pools timestamp representations from the cumulative averaged encoder.
The SVM's C is selected using training data only; test labels score accuracy.
Grid search runs serially and is skipped when a class has fewer than five
training examples. No forecasting or sliding-window inference is included.

For a matched comparison, use zhihanyue/ts2vec at commit
b0088e14a99706c05451316dc6db8d3da9351163 with identical data and settings::

    python train.py ECG200 matched --loader UCR --seed 0 --iters 200 \
        --batch-size 16 --repr-dims 320 --eval

Reproducible CPU comparisons are provided in ``benchmarks/ts2vec/README.md``.
``--checkpoint`` saves model weights and the averaging counter; use Lightning
checkpoints for optimizer resume. Instance negatives are local to each rank;
this example trains on one device.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from functools import partial

import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset


class UCRSeries(Dataset):
    """Fixed-length UCR rows with label in column zero."""

    def __init__(self, path: Path) -> None:
        rows = np.loadtxt(path, delimiter="\t", dtype=np.float32, ndmin=2)
        if rows.shape[1] < 2:
            raise ValueError(f"{path} must contain a label and at least one timestep")
        if not np.isfinite(rows).all():
            raise ValueError(f"{path} must contain finite labels and timesteps")
        if not np.equal(rows[:, 0], np.floor(rows[:, 0])).all():
            raise ValueError(f"{path} labels must be integer class IDs")
        self.labels = torch.as_tensor(rows[:, 0], dtype=torch.long)
        self.series = torch.as_tensor(rows[:, 1:, None], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"series": self.series[index], "label": self.labels[index]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_tsv", type=Path)
    parser.add_argument("test_tsv", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dims", type=int, default=320)
    parser.add_argument("--hidden-dims", type=int, default=64)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    import stable_pretraining as spt
    import lightning as pl

    train_set, test_set = UCRSeries(args.train_tsv), UCRSeries(args.test_tsv)
    batch_size = min(args.batch_size, len(train_set))
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True
    )
    data = spt.data.DataModule(train=train_loader)
    model = spt.TS2Vec(
        input_dims=1,
        output_dims=args.output_dims,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        temporal_unit=0,
        optim={"optimizer": partial(torch.optim.AdamW, lr=0.001, weight_decay=0.01)},
    )
    trainer = pl.Trainer(
        max_steps=args.steps,
        max_epochs=-1,
        accelerator=args.accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    spt.Manager(trainer=trainer, module=model, data=data, seed=args.seed)()

    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)

    device = next(model.parameters()).device
    model.eval()

    @torch.no_grad()
    def embeddings(dataset):
        values, labels = [], []
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            result = model(batch, stage="test")
            values.append(result["embedding"].detach().cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
        return np.concatenate(values), np.concatenate(labels)

    x_train, y_train = embeddings(train_set)
    x_test, y_test = embeddings(test_set)
    n_classes, counts = np.unique(y_train, return_counts=True)
    if n_classes.size < 2:
        raise ValueError("SVM evaluation requires at least two training classes")
    svm = SVC(C=np.inf, gamma="scale")
    if counts.min() < 5 or len(y_train) < 50:
        classifier = svm.fit(x_train, y_train)
    else:
        classifier = (
            GridSearchCV(
                SVC(gamma="scale", max_iter=10000000),
                {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf]},
                cv=5,
                n_jobs=1,
            )
            .fit(x_train, y_train)
            .best_estimator_
        )
    accuracy = accuracy_score(y_test, classifier.predict(x_test))
    result = {
        "accuracy": float(accuracy),
        "train_size": len(train_set),
        "test_size": len(test_set),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": batch_size,
        "output_dims": args.output_dims,
        "hidden_dims": args.hidden_dims,
        "depth": args.depth,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "stable_pretraining": spt.__version__,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
