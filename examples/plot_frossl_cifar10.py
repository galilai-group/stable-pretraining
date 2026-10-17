"""Plot FroSSL training losses and online-probe validation accuracy.

Example::

    python examples/plot_frossl_cifar10.py /path/to/run
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_METRICS = {
    "fit/loss_epoch": "frossl_loss",
    "train/linear_probe_loss_epoch": "probe_train_loss",
    "eval/linear_probe_MulticlassAccuracy_epoch": "probe_val_accuracy",
}


def read_epoch_metrics(path: Path) -> list[dict[str, float]]:
    """Collect finite epoch metrics from the registry CSV.

    Args:
        path: Path to a FroSSL run's ``metrics.csv``.

    Returns:
        One row per epoch, in ascending order. Repeated epochs from a resumed
        run use their latest recorded value for each metric.

    Raises:
        ValueError: No epoch metrics were found.
    """
    by_epoch: dict[int, dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            raw_epoch = row.get("epoch")
            if not raw_epoch:
                continue
            epoch = int(float(raw_epoch)) + 1
            values = by_epoch.setdefault(epoch, {"epoch": float(epoch)})
            for source, target in _METRICS.items():
                raw_value = row.get(source)
                if raw_value:
                    value = float(raw_value)
                    if math.isfinite(value):
                        values[target] = value
    rows = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    if not any("frossl_loss" in row for row in rows):
        raise ValueError(f"No epoch-level FroSSL loss found in {path}")
    return rows


def save_training_curves(run_dir: Path) -> tuple[Path, Path]:
    """Save a plot and epoch-level CSV for one completed FroSSL run.

    Args:
        run_dir: Registry run directory containing ``metrics.csv``.

    Returns:
        Paths to the PNG plot and tidy epoch-level CSV.

    Note:
        FroSSL is self-supervised, so classification accuracy comes from the
        online linear probe on the held-out validation split.
    """
    rows = read_epoch_metrics(run_dir / "metrics.csv")
    table_path = run_dir / "training_curves.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", *_METRICS.values()])
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout="constrained")
    series = (
        ("frossl_loss", "FroSSL training loss", "#1f77b4"),
        ("probe_train_loss", "Online-probe training loss", "#ff7f0e"),
        ("probe_val_accuracy", "Online-probe validation accuracy (%)", "#2ca02c"),
    )
    for axis, (key, label, color) in zip(axes, series, strict=True):
        points = [(row["epoch"], row[key]) for row in rows if key in row]
        if key == "probe_val_accuracy":
            points = [(epoch, value * 100) for epoch, value in points]
            axis.set_ylim(0, 100)
        if points:
            axis.plot(*zip(*points, strict=True), color=color, linewidth=1.5)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Epoch")
    fig.suptitle("FroSSL on CIFAR-10")
    plot_path = run_dir / "training_curves.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path, table_path


def main() -> None:
    """Read one run directory and write its curve artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    for artifact in save_training_curves(args.run_dir):
        print(artifact)


if __name__ == "__main__":
    main()
