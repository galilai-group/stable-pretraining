"""CPU adapter that trains author TS2Vec and applies the example's SVM protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def read_tsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = np.loadtxt(path, delimiter="\t", dtype=np.float32, ndmin=2)
    return rows[:, 1:, None], rows[:, 0].astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_tsv", type=Path)
    parser.add_argument("test_tsv", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output-dims", type=int, required=True)
    parser.add_argument("--hidden-dims", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference))
    from ts2vec import TS2Vec

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train, y_train = read_tsv(args.train_tsv)
    test, y_test = read_tsv(args.test_tsv)
    model = TS2Vec(
        input_dims=train.shape[-1],
        output_dims=args.output_dims,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        temporal_unit=0,
        device="cpu",
        batch_size=args.batch_size,
        lr=0.001,
    )
    model.fit(train, n_iters=args.steps, verbose=False)
    if model.n_iters != args.steps:
        raise RuntimeError(
            f"Reference completed {model.n_iters} updates; expected {args.steps}"
        )
    x_train = model.encode(
        train, encoding_window="full_series", batch_size=args.batch_size
    )
    x_test = model.encode(
        test, encoding_window="full_series", batch_size=args.batch_size
    )
    _, counts = np.unique(y_train, return_counts=True)
    if counts.min() < 5 or len(y_train) < 50:
        classifier = SVC(C=np.inf, gamma="scale").fit(x_train, y_train)
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
    print(
        json.dumps(
            {
                "accuracy": float(accuracy_score(y_test, classifier.predict(x_test))),
                "train_size": len(y_train),
                "test_size": len(y_test),
                "seed": args.seed,
                "steps": args.steps,
                "batch_size": min(args.batch_size, len(y_train)),
                "completed_updates": model.n_iters,
                "averaged_encoder_count": int(model.net.n_averaged),
                "output_dims": args.output_dims,
                "hidden_dims": args.hidden_dims,
                "depth": args.depth,
                "optimizer": "AdamW",
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "sklearn": __import__("sklearn").__version__,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
