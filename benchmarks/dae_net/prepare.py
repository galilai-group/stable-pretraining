"""Prepare full-category training and test-only scoring manifests from author data."""

import argparse
import io
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from protocol import CATEGORIES, DEFAULT_CONFIG, digest, load_config, relative_source


def _shape_ids(path: Path) -> list[str]:
    values = [line.strip().split("/")[-1] for line in path.read_text().splitlines()]
    if not values or any(
        not value or value in (".", "..") or "\\" in value for value in values
    ):
        raise ValueError(f"Invalid shape IDs in {path}")
    return values


def prepare(
    root: Path, category_dir: Path, output: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Create manifests preserving full-file row order and test-list membership.

    Args:
        root: Dataset root for all stored relative paths.
        category_dir: Extracted author category directory under root.
        output: New or empty manifest directory.
        config: Selected category recipe from load_config.

    Returns:
        Preparation summary including coverage and input identity.

    Raises:
        ValueError: If IDs, array schema, coverage or surface labels are invalid.
        FileExistsError: If output already contains files.

    Note:
        Test geometry remains in training. Duplicate test-list IDs are retained
        once, in first-occurrence order, matching the existing scoring protocol.
        Training labels are never read; only test surface files are inspected.
    """
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new or empty manifest directory")
    root = root.resolve()
    folder = (root / category_dir).resolve()
    synset = config["synset"]
    source = folder / f"{synset}_vox.hdf5"
    identities = folder / f"{synset}_vox.txt"
    tests = folder / f"{synset}_test_vox.txt"
    paths = {
        key: relative_source(root, path)
        for key, path in (("hdf5", source), ("ids", identities), ("tests", tests))
    }
    ids, test_ids = _shape_ids(identities), _shape_ids(tests)
    if len(ids) != config["train_shapes"] or len(set(ids)) != len(ids):
        raise ValueError(
            "Full-category count or unique shape identities disagree with config"
        )
    if set(test_ids) - set(ids):
        raise ValueError("Test identities are missing from the full-category file")
    unique_tests = list(dict.fromkeys(test_ids))
    resolution = config["query_resolution"]
    queries = 32768 if resolution == 64 else 8192
    with h5py.File(source, "r") as handle:
        for name, shape in {
            "voxels": (len(ids), 64, 64, 64, 1),
            f"points_{resolution}": (len(ids), queries, 3),
            f"values_{resolution}": (len(ids), queries, 1),
        }.items():
            if handle[name].shape != shape or handle[name].dtype != np.uint8:
                raise ValueError(f"Unexpected HDF5 schema: {name}")
    source_hashes = {path: digest(root / path) for path in paths.values()}
    records = [
        {
            "shape_id": shape_id,
            "synset": synset,
            "source_hdf5": paths["hdf5"],
            "source_hdf5_sha256": source_hashes[paths["hdf5"]],
            "row_index": row,
            "source_txt": paths["ids"],
            "clean_split": "train",
            "query_resolution": resolution,
            "protocol": config["protocol"],
        }
        for row, shape_id in enumerate(ids)
    ]
    by_id = {row["shape_id"]: row for row in records}
    selected = [{**by_id[shape_id], "clean_split": "test"} for shape_id in unique_tests]
    label_ids = {
        "mug": {36, 37},
        "chair": {12, 13, 14, 15},
        "lamp": {24, 25, 26, 27},
        "skateboard": {44, 45, 46},
    }[config["category"]]
    surfaces = []
    for shape_id in unique_tests:
        path = folder / "points" / f"{shape_id}.txt"
        relative = relative_source(root, path)
        content = path.read_bytes()
        values = np.loadtxt(io.BytesIO(content), dtype=np.float64, ndmin=2)
        if values.shape[1] != 7 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Expected finite surface [N,7]: {path}")
        labels = values[:, 6]
        if not np.array_equal(labels, np.floor(labels)) or set(labels) - label_ids:
            raise ValueError(f"Unexpected semantic labels: {path}")
        surfaces.append(
            {
                "shape_id": shape_id,
                "synset": synset,
                "clean_split": "test",
                "source_points": relative,
                "source_points_sha256": digest(path),
                "point_count": len(values),
                "raw_labels": sorted(set(labels.astype(int).tolist())),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    manifests = {
        "train.jsonl": records,
        "test.jsonl": selected,
        "evaluation_labels.jsonl": surfaces,
    }
    for name, rows in manifests.items():
        (output / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {
        "category": config["category"],
        "synset": synset,
        "protocol": config["protocol"],
        "train_shapes": len(records),
        "test_shapes": len(selected),
        "query_resolution": resolution,
        "test_geometry_seen_during_training": True,
        "test_duplicates_removed": len(test_ids) - len(unique_tests),
        "source_sha256": source_hashes,
        "manifest_sha256": {name: digest(output / name) for name in manifests},
    }
    (output / "prepared.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    """Prepare manifests for an extracted author category.

    Args:
        argv: CLI arguments, or None for process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=CATEGORIES, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--category-dir",
        type=Path,
        help="Relative to data-root; defaults to data/SYNSET_CATEGORY",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.category, args.config)
    folder = args.category_dir or Path("data") / f"{config['synset']}_{args.category}"
    print(
        json.dumps(prepare(args.data_root, folder, args.output_dir, config), indent=2)
    )


if __name__ == "__main__":
    main()
