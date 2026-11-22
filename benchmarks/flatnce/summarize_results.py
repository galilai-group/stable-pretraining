"""Validate matched CIFAR-10 runs and export per-epoch results and accuracy plots."""

import argparse
import csv
import json
import math
from pathlib import Path


_METRICS = {
    "eval/linear_probe_top1_epoch": "linear_probe_macro_top1",
    "eval/knn_probe_accuracy": "knn_macro_top1",
    "fit/loss_epoch": "contrastive_loss",
}
_MATCHED_SETTINGS = (
    "batch_size",
    "temperature",
    "epochs",
    "lr",
    "seed",
    "num_workers",
    "accelerator",
    "devices",
    "precision",
    "validation_size",
    "eval_split",
    "train_samples",
    "smoke_test",
    "max_steps",
)


def _read_epochs(path: Path, count: int) -> list[dict]:
    epochs = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            for key, value in row.items():
                if value and not math.isfinite(float(value)):
                    raise ValueError(f"Nonfinite metric {key} in {path}")
            if not row.get("epoch"):
                continue
            epoch = int(float(row["epoch"])) + 1
            entry = epochs.setdefault(epoch, {"epoch": epoch})
            for source, target in _METRICS.items():
                if row.get(source):
                    entry[target] = float(row[source])
            for source, target in (
                ("diagnostics/gradient_norm", "gradient_norm"),
                ("diagnostics/embedding_std", "embedding_std"),
            ):
                if row.get(source):
                    value = float(row[source])
                    if value <= 0:
                        raise ValueError(f"Nonpositive {source} at epoch {epoch}")
                    entry[target + "_min"] = min(
                        value, entry.get(target + "_min", value)
                    )
                    entry[target + "_max"] = max(
                        value, entry.get(target + "_max", value)
                    )
                    if target == "gradient_norm":
                        entry["updates_logged"] = entry.get("updates_logged", 0) + 1
    if sorted(epochs) != list(range(1, count + 1)):
        raise ValueError(f"Expected {count} complete epochs in {path}")
    required = set(_METRICS.values()) | {"gradient_norm_min", "embedding_std_min"}
    for epoch, entry in epochs.items():
        if not required.issubset(entry):
            raise ValueError(f"Missing validation or diagnostics at epoch {epoch}")
        if not all(
            0 <= entry[key] <= 1
            for key in ("linear_probe_macro_top1", "knn_macro_top1")
        ):
            raise ValueError(f"Accuracy outside [0, 1] at epoch {epoch}")
    return [epochs[epoch] for epoch in sorted(epochs)]


def _load_run(path: Path, method: str) -> tuple[dict, list[dict]]:
    summary = json.loads((path / "experiment_summary.json").read_text())
    settings = summary["settings"]
    if summary["dataset"] != "CIFAR-10" or settings["smoke_test"]:
        raise ValueError("The comparison requires real CIFAR-10 runs")
    if settings["method"] != method or settings["devices"] != 1:
        raise ValueError(f"Expected a single-device {method} run")
    expected_steps = (
        summary["train_images"] // settings["batch_size"] * settings["epochs"]
    )
    if summary["global_step"] != expected_steps:
        raise ValueError(f"Incomplete {method} run: expected {expected_steps} updates")
    epochs = _read_epochs(path / "metrics.csv", settings["epochs"])
    if any(
        row["updates_logged"] != expected_steps // settings["epochs"] for row in epochs
    ):
        raise ValueError(f"Missing training diagnostics in {method} run")
    if method == "flatnce" and any(
        abs(row["contrastive_loss"] - 1) > 1e-6 for row in epochs
    ):
        raise ValueError("FlatNCE's objective should remain one")
    return summary, epochs


def _validate_pair(first: dict, second: dict) -> None:
    for key in _MATCHED_SETTINGS:
        if first["settings"][key] != second["settings"][key]:
            raise ValueError(f"Experiment settings differ: {key}")
    for key in ("train_images", "evaluation_images"):
        if first[key] != second[key]:
            raise ValueError(f"Experiment datasets differ: {key}")


def main() -> None:
    """Export checked experiment data and standalone PNG/PDF accuracy figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flatnce-run", type=Path, required=True)
    parser.add_argument("--simclr-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = {
        method: _load_run(path, method)
        for method, path in (("flatnce", args.flatnce_run), ("simclr", args.simclr_run))
    }
    _validate_pair(runs["flatnce"][0], runs["simclr"][0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "protocol": {
            key: runs["flatnce"][0]["settings"][key] for key in _MATCHED_SETTINGS
        },
        "runs": {},
    }
    for method, (summary, epochs) in runs.items():
        with (args.output_dir / f"{method}-epochs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(epochs[0]))
            writer.writeheader()
            writer.writerows(epochs)
        final = epochs[-1]
        result = {
            "global_step": summary["global_step"],
            "train_images": summary["train_images"],
            "evaluation_images": summary["evaluation_images"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "final": final,
        }
        for metric in ("linear_probe_macro_top1", "knn_macro_top1"):
            best = max(epochs, key=lambda row: row[metric])
            result["best_" + metric] = {"epoch": best["epoch"], "value": best[metric]}
        comparison["runs"][method] = result
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for axis, metric, title in zip(
        axes,
        ("linear_probe_macro_top1", "knn_macro_top1"),
        ("Online linear probe", "Online k-NN (k=10)"),
    ):
        for method, (_, epochs) in runs.items():
            axis.plot(
                [row["epoch"] for row in epochs],
                [100 * row[metric] for row in epochs],
                marker="o",
                markersize=4,
                label="FlatNCE" if method == "flatnce" else "SimCLR",
            )
        axis.axhline(10, color="gray", linestyle=":", linewidth=1, label="10% chance")
        axis.set(title=title, xlabel="Epoch", ylim=(0, 100))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Validation macro top-1 accuracy (%)")
    settings = comparison["protocol"]
    fig.suptitle(
        f"CIFAR-10 / ResNet-18 / batch {settings['batch_size']} / "
        f"temperature {settings['temperature']} / seed {settings['seed']}"
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "accuracy.png", dpi=200)
    fig.savefig(args.output_dir / "accuracy.pdf")
    plt.close(fig)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
