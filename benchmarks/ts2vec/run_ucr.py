#!/usr/bin/env python3
"""Run matched short UCR benchmarks for stable-pretraining and reference TS2Vec."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import requests


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REFERENCE_COMMIT = "b0088e14a99706c05451316dc6db8d3da9351163"
ARCHIVE_BASE = "https://www.timeseriesclassification.com/aeon-toolkit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(name: str, data_dir: Path) -> tuple[Path, Path, str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / f"{name}.zip"
    url = f"{ARCHIVE_BASE}/{name}.zip"
    if not archive.exists():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        archive.write_bytes(response.content)
    checksum = sha256(archive)
    with zipfile.ZipFile(archive) as zipped:
        for split in ("TRAIN", "TEST"):
            matches = [
                item
                for item in zipped.namelist()
                if Path(item).name.lower() == f"{name}_{split.lower()}.ts".lower()
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one {name}_{split}.ts in {archive}, got {matches}"
                )
            target = data_dir / f"{name}_{split}.ts"
            target.write_bytes(zipped.read(matches[0]))
    return data_dir / f"{name}_TRAIN.ts", data_dir / f"{name}_TEST.ts", checksum


def convert_ts(source: Path, destination: Path) -> None:
    """Convert the equal-length univariate archive format to example TSV input."""
    rows: list[str] = []
    with source.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("@") or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) < 2:
                continue
            values = parts[0].split(",")
            label = parts[-1]
            rows.append("\t".join([label, *values]))
    if not rows:
        raise ValueError(f"No records found in {source}")
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run(command: list[str], env: dict[str, str], cwd: Path) -> dict:
    print("$ " + " ".join(command), flush=True)
    proc = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    if proc.returncode:
        raise subprocess.CalledProcessError(
            proc.returncode, command, proc.stdout, proc.stderr
        )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError(f"No JSON result in output from: {command}")
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["ECG200", "GunPoint"],
        default=["ECG200", "GunPoint"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--results", type=Path, default=HERE / "results.jsonl")
    parser.add_argument(
        "--reference", type=Path, default=ROOT.parent / "ts2vec-reference"
    )
    args = parser.parse_args()
    if args.steps != 200:
        print(
            "Note: the recorded benchmark default is 200 updates; this run overrides it."
        )
    actual_commit = subprocess.check_output(
        ["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != REFERENCE_COMMIT:
        raise RuntimeError(
            f"Reference checkout must be {REFERENCE_COMMIT}; found {actual_commit}"
        )
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "lightning", "scikit-learn", "numpy")
    }

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    args.results.parent.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        train_raw, test_raw, archive_hash = download_dataset(name, args.data_dir)
        train_tsv, test_tsv = (
            args.data_dir / f"{name}_TRAIN.tsv",
            args.data_dir / f"{name}_TEST.tsv",
        )
        convert_ts(train_raw, train_tsv)
        convert_ts(test_raw, test_tsv)
        for seed in args.seeds:
            common = [
                "--seed",
                str(seed),
                "--steps",
                str(args.steps),
                "--batch-size",
                "16",
                "--output-dims",
                "320",
                "--hidden-dims",
                "64",
                "--depth",
                "10",
            ]
            current_command = [
                sys.executable,
                str(ROOT / "examples/ts2vec_ucr.py"),
                str(train_tsv),
                str(test_tsv),
                *common,
                "--accelerator",
                "cpu",
            ]
            current = run(current_command, env, ROOT)
            ref_env = env.copy()
            ref_env["PYTHONPATH"] = (
                str(args.reference) + os.pathsep + ref_env.get("PYTHONPATH", "")
            )
            reference_command = [
                sys.executable,
                str(HERE / "reference_ucr.py"),
                str(train_tsv),
                str(test_tsv),
                *common,
                "--reference",
                str(args.reference),
            ]
            reference = run(reference_command, ref_env, ROOT)
            record = {
                "dataset": name,
                "seed": seed,
                "archive_url": f"{ARCHIVE_BASE}/{name}.zip",
                "archive_sha256": archive_hash,
                "train_source_sha256": sha256(train_raw),
                "test_source_sha256": sha256(test_raw),
                "train_tsv_sha256": sha256(train_tsv),
                "test_tsv_sha256": sha256(test_tsv),
                "reference_commit": actual_commit,
                "project_commit": subprocess.check_output(
                    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
                ).strip(),
                "python": sys.version.split()[0],
                "versions": versions,
                "commands": {
                    "current": current_command,
                    "reference": reference_command,
                },
                "current": current,
                "reference": reference,
            }
            with args.results.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
