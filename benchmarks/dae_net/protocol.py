"""Configuration and input identity for the full-category DAE-Net benchmark."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).with_name("config.json")
CATEGORIES = ("mug", "chair", "lamp", "skateboard")


def load_config(category: str, path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read one category's released training recipe.

    Args:
        category: Shape category in the configuration.
        path: JSON containing common settings and per-category overrides.

    Returns:
        Merged configuration, with the selected category recorded.

    Raises:
        ValueError: If a setting is outside the supported FP32 protocol.
    """
    config = json.loads(path.read_text())
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}")
    result = {
        **config["common"],
        **config["categories"][category],
        "category": category,
    }
    for key in (
        "train_shapes",
        "branch_num",
        "batch_size",
        "epochs",
        "era_length",
        "shapes_per_epoch",
    ):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(result["seed"]) is not int or not 0 <= result["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if result["era_length"] < 2 or result["query_resolution"] not in (32, 64):
        raise ValueError("Unsupported era length or query resolution")
    if result["precision"] != "32-true" or result["queries_per_shape"] != 8192:
        raise ValueError("This benchmark uses FP32 and 8192 queries per shape")
    if result["protocol"] != "official_release_full_category":
        raise ValueError("This benchmark trains on full-category geometry")
    for key in ("lr", "sparse_weight"):
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"Invalid {key}")
    if result["lr"] == 0:
        raise ValueError("lr must be positive")
    return result


def digest(path: Path) -> str:
    """Return a bounded-memory SHA256 for a source or manifest.

    Args:
        path: File to read.

    Returns:
        Hexadecimal digest, used only for input identity and recovery checks.
    """
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def relative_source(root: Path, path: Path) -> str:
    """Resolve a source inside the portable data root.

    Args:
        root: User-provided dataset directory.
        path: Source file, absolute or relative to root.

    Returns:
        Root-relative POSIX path.

    Raises:
        ValueError: If the resolved source is outside the data root.
    """
    return (root / path).resolve(strict=True).relative_to(root.resolve()).as_posix()


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL manifest.

    Args:
        path: Manifest path.

    Returns:
        Nonblank JSON records in source order.
    """
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
