"""Regenerate DirectCLR training figures from the included scalar inputs."""

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/directclr-plot-cache")
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

PUBLIC = Path(__file__).resolve().parent


lr_rows = []
with (PUBLIC / "data/cifar-learning-rates.csv").open() as f:
    for row in csv.DictReader(f):
        lr_rows.append(
            {
                "epoch": int(row["epoch"]),
                "encoder_last_lr": float(row["encoder_last_lr"]),
                "head_last_lr": float(row["head_last_lr"]),
            }
        )


def csv_rows(name: str) -> list[dict]:
    """Read the original exported numeric CSV without smoothing."""
    with (PUBLIC / "data" / name).open() as f:
        return list(csv.DictReader(f))


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "path",
    }
)
colors = ["#327ba8", "#d69236", "#7865a8", "#248977"]


def figure(fig, name: str) -> None:
    """Save an English PNG figure for the experiment report."""
    fig.tight_layout()
    fig.savefig(PUBLIC / "figures" / (name + ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


x = csv_rows("ssl-loss.csv")
d = csv_rows("ssl-dev-selection.csv")
assert len(x) == 800 and len(d) == 4
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for key, label, col in zip(
    ["temp-0.05", "temp-0.2", "budget-400"],
    ["T=0.05, 200 epochs", "T=0.2, 200 epochs", "T=0.2, fresh 400 epochs"],
    colors,
):
    rows = [e for e in x if e["candidate"] == key]
    assert len(rows) == (400 if key == "budget-400" else 200)
    axes[0].plot(
        [int(e["epoch"]) for e in rows],
        [float(e["logged_loss_mean"]) for e in rows],
        label=label,
        color=col,
    )
axes[0].axvline(10128 / 247, color="#888888", ls=":", lw=1)
axes[0].set(
    xlabel="Encoder epoch",
    ylabel="Mean of logged SSL losses",
    title="Actual logged training; no smoothing",
)
axes[0].legend(fontsize=8)
values = [float(e["top1_percent"]) for e in d]
bars = axes[1].bar(
    ["0.1 / 200", "0.05 / 200", "0.2 / 200", "0.2 / 400"], values, color=colors
)
for bar, v in zip(bars, values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2, v + 0.7, f"{v:.2f}", ha="center", fontsize=9
    )
axes[1].set(
    ylim=(0, 100),
    xlabel="Temperature / encoder epochs",
    ylabel="Final dev Top-1 (%)",
    title="Dev-only temperature and budget selection",
)
for ax in axes:
    ax.grid(axis="y", alpha=0.2)
figure(fig, "01-ssl-training-and-selection")
x = csv_rows("frozen-classifier-loss.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
for key, label, col in zip(
    ["temp-0.05", "temp-0.2", "budget-400", "selected-fulltrain-refit"],
    ["T=0.05 / 200", "T=0.2 / 200", "T=0.2 / 400", "Selected full-train refit"],
    colors,
):
    rows = [e for e in x if e["candidate"] == key]
    assert len(rows) == 100
    axes[0].plot(
        [int(e["epoch"]) for e in rows],
        [float(e["mean_loss"]) for e in rows],
        label=label,
        color=col,
    )
axes[0].set(
    xlabel="Classifier epoch",
    ylabel="Fit cross-entropy",
    title="Fresh 100-epoch frozen classifiers",
)
axes[0].legend(fontsize=8)
bars = axes[1].bar(["Top-1", "Top-5"], [79.8, 94.98], color=[colors[3], colors[0]])
for bar, v in zip(bars, [79.8, 94.98]):
    axes[1].text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.2f}%", ha="center")
axes[1].set(
    ylim=(0, 105),
    ylabel="Single final val accuracy (%)",
    title="ImageNet100: 5,000 final val images",
)
for ax in axes:
    ax.grid(axis="y", alpha=0.2)
figure(fig, "02-frozen-evaluation")
x = csv_rows("cifar-training.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
axes[0].plot(
    [int(e["epoch"]) for e in x], [float(e["mean_loss"]) for e in x], color=colors[3]
)
axes[0].axvline(45.5, ls="--", color="#888888", label="Resume after retained epoch 45")
axes[0].set(
    xlabel="Original epoch index",
    ylabel="Supervised fit cross-entropy",
    title="CIFAR transfer: 100 epochs, 17,600 updates",
)
axes[0].legend(fontsize=8)
for key, label, col in [
    ("encoder_last_lr", "Encoder", colors[0]),
    ("head_last_lr", "New head", colors[1]),
]:
    axes[1].plot(
        [e["epoch"] for e in lr_rows], [e[key] for e in lr_rows], label=label, color=col
    )
axes[1].axvline(45.5, ls="--", color="#888888")
axes[1].set(
    xlabel="Original epoch index",
    ylabel="Last successful-update LR",
    title="Original warmup/cosine schedule preserved",
)
axes[1].legend()
for ax in axes:
    ax.grid(alpha=0.2)
figure(fig, "03-cifar-transfer-and-resume")
