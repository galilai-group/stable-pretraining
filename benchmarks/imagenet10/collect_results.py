"""Print best per-method top1 from CSV logs.

Looks under both ``benchmarks/imagenet10/logs`` (CSVLogger) and
``~/.cache/stable-pretraining/runs`` (SPT Manager auto-runs).
"""

from pathlib import Path
import sys

import pandas as pd

LOG_DIRS = [
    Path(__file__).parent / "logs",
    Path.home() / ".cache" / "stable-pretraining" / "runs",
]


def _summarize(csv_path: Path, label: str):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"method": label, "error": str(e)}
    cols = set(df.columns)
    knn = df["eval/knn_probe_top1"].dropna() if "eval/knn_probe_top1" in cols else pd.Series([], dtype=float)
    lin = (
        df["eval/linear_probe_top1_epoch"].dropna()
        if "eval/linear_probe_top1_epoch" in cols
        else (df["eval/linear_probe_top1"].dropna() if "eval/linear_probe_top1" in cols else pd.Series([], dtype=float))
    )
    loss = df["fit/loss_epoch"].dropna() if "fit/loss_epoch" in cols else pd.Series([], dtype=float)
    return {
        "method": label,
        "epochs": int(df["epoch"].max()) + 1 if "epoch" in cols and not df["epoch"].dropna().empty else 0,
        "best_knn": float(knn.max()) if len(knn) else float("nan"),
        "best_linear": float(lin.max()) if len(lin) else float("nan"),
        "last_loss": float(loss.iloc[-1]) if len(loss) else float("nan"),
    }


def main():
    rows = []
    for base in LOG_DIRS:
        if not base.exists():
            continue
        # CSVLogger layout: <base>/<name>/version_*/metrics.csv
        for csv in base.rglob("metrics.csv"):
            # Skip empty files
            if csv.stat().st_size == 0:
                continue
            label = csv.parent.parent.name if "version_" in csv.parent.name else csv.parent.name
            rows.append(_summarize(csv, label))

    if not rows:
        print("no metrics found")
        sys.exit(0)
    out = pd.DataFrame(rows).sort_values(["method", "epochs"], ascending=[True, False]).drop_duplicates("method")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
