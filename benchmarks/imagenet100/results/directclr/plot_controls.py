"""Regenerate the scalar CIFAR-control figure; no models or data downloads."""

from pathlib import Path
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Read committed scalar CSVs and regenerate the three-panel control plot."""
    root = Path(__file__).resolve().parent
    with (root / "data/cifar-controls-training.csv").open() as stream:
        epochs = list(csv.DictReader(stream))
    with (root / "data/cifar-controls-comparison.csv").open() as stream:
        scores = list(csv.DictReader(stream))
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    bars = axs[0].bar(
        [
            "Pretrained\nfinetune",
            "Random-init\nfull training",
            "Pretrained\nfrozen encoder",
        ],
        [float(r["top1"]) for r in scores],
        color=["#228870", "#db9234", "#7264ad"],
    )
    axs[0].set(title="Same 5,000-image dev split", ylabel="Top-1 (%)", ylim=(0, 100))
    axs[0].bar_label(bars, fmt="%.2f", padding=4)
    for kind, color in [("random-init", "#db9234"), ("frozen-encoder", "#7264ad")]:
        rows = [r for r in epochs if r["control"] == kind]
        axs[1].plot(
            [int(r["epoch"]) for r in rows],
            [float(r["mean_loss"]) for r in rows],
            label=kind,
            color=color,
        )
    axs[1].set(
        title="New control training losses",
        xlabel="Epoch",
        ylabel="Mean fit cross-entropy",
    )
    axs[1].legend()
    rows = [r for r in epochs if r["control"] == "random-init"]
    for field, label, color in [
        ("last_encoder_lr", "Encoder LR", "#2782b4"),
        ("last_head_lr", "Head LR", "#b85541"),
    ]:
        axs[2].plot(
            [int(r["epoch"]) for r in rows],
            [float(r[field]) for r in rows],
            label=label,
            color=color,
        )
    axs[2].set(
        title="Shared prescribed schedule",
        xlabel="Epoch",
        ylabel="End-of-epoch learning rate",
    )
    axs[2].legend()
    for ax in axs:
        ax.grid(axis="y", alpha=0.18)
    fig.savefig(root / "figures/05-cifar-controls.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
