"""Render the recorded projector comparison under both classifier protocols."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main() -> None:
    """Plot historical scalar results without loading models or image data."""
    root = Path(__file__).resolve().parent
    with (root / "data/projector-ablation.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    methods = ["directclr", "linear", "full"]
    colors = ["#327ba8", "#248977", "#d69236"]
    labels = [
        "DirectCLR\nfixed 360",
        "Trainable linear\n2048 to 360",
        "No projector\nfull 2048",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), layout="constrained")
    for axis, protocol, title in zip(
        axes,
        ["fixed-lr1", "dev-selected-lr10"],
        ["Fixed classifier LR = 1", "Classifier LR selected on dev (1 / 3 / 10)"],
    ):
        values = [
            float(
                next(
                    r
                    for r in rows
                    if r["method"] == method and r["protocol"] == protocol
                )["val_top1_percent"]
            )
            for method in methods
        ]
        bars = axis.bar(labels, values, color=colors)
        axis.bar_label(bars, fmt="%.2f%%", padding=4)
        axis.set(title=title, ylabel="Final validation Top-1 (%)", ylim=(0, 75))
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "ImageNet-100: matched 100-epoch encoder comparison", fontweight="bold"
    )
    fig.supxlabel(
        "One encoder seed | same cross-view loss, training budget and frozen-feature protocol\n"
        "The DirectCLR / no-projector ordering changes with classifier evaluation."
    )
    fig.savefig(root / "figures/06-projector-ablation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
