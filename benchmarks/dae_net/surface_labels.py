"""Surface-label adapter for the DAE-Net benchmark.

Source and license: THIRD_PARTY_NOTICES.md (DAE-Net).
"""

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


_LABEL_IDS = {
    "03797390": (36, 37),
    "03001627": (12, 13, 14, 15),
    "03636649": (24, 25, 26, 27),
    "04225987": (44, 45, 46),
}


class DAENetSurfaceLabels:
    """Join audited surface labels to voxel records in voxel-manifest order.

    Args:
        root: Data root containing the source point files.
        manifest: Label JSONL containing shape IDs, synsets, splits, point
            counts, raw label sets, source paths and source SHA256 values.
        voxel_records: Records from the selected DAENetDataset split.
        split: Expected train, val or test split.
        label_ids: Optional explicit sorted category vocabulary. Defaults to
            the fixed ShapeNet vocabulary for Mug, Chair, Lamp or Skateboard.

    Note:
        This adapter does not normalize or subsample surface points. It uses
        the upstream (x,y,z) -> (z,y,-x) coordinate convention. Matching is exact
        within the selected split; other splits in the label manifest are not
        evaluated. Labels remain separate from the training dataset.
    """

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        voxel_records: Sequence[dict[str, Any]],
        *,
        split: str,
        label_ids: Sequence[int] | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError("Invalid surface-label split")
        self.root = Path(root).resolve()
        self.manifest = Path(manifest)
        content = self.manifest.read_bytes()
        self.manifest_sha256 = hashlib.sha256(content).hexdigest()
        records = [json.loads(line) for line in content.splitlines() if line.strip()]
        if not records or not voxel_records:
            raise ValueError("Surface and voxel manifests must be nonempty")
        required = {
            "shape_id",
            "synset",
            "clean_split",
            "source_points",
            "source_points_sha256",
            "point_count",
            "raw_labels",
        }
        by_id = {}
        categories = set()
        for record in records:
            if not isinstance(record, dict) or required - record.keys():
                raise ValueError("Missing surface-label manifest fields")
            shape_id = record["shape_id"]
            if not isinstance(shape_id, str) or not shape_id or shape_id in by_id:
                raise ValueError("Invalid or duplicate surface shape ID")
            if record["clean_split"] not in ("train", "val", "test"):
                raise ValueError(f"Invalid split for {shape_id}")
            if not isinstance(record["synset"], str):
                raise ValueError("synset must be a string")
            categories.add(record["synset"])
            if type(record["point_count"]) is not int or record["point_count"] <= 0:
                raise ValueError(f"Invalid point_count for {shape_id}")
            raw_labels = record["raw_labels"]
            if (
                not isinstance(raw_labels, list)
                or not raw_labels
                or any(type(label) is not int or label < 0 for label in raw_labels)
                or raw_labels != sorted(set(raw_labels))
            ):
                raise ValueError(f"Invalid raw_labels for {shape_id}")
            digest = record["source_points_sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError(f"Invalid point SHA256 for {shape_id}")
            if not isinstance(record["source_points"], str):
                raise ValueError("source_points must be a path string")
            path = (self.root / record["source_points"]).resolve()
            if not path.is_relative_to(self.root) or path.stem != shape_id:
                raise ValueError(
                    f"Point path does not match data root/shape ID: {shape_id}"
                )
            by_id[shape_id] = record
        if len(categories) != 1:
            raise ValueError("Surface manifest must contain exactly one category")
        self.synset = categories.pop()
        vocabulary = label_ids if label_ids is not None else _LABEL_IDS.get(self.synset)
        if vocabulary is None:
            raise ValueError(
                "Unknown category; supply an explicit fixed label_ids vocabulary"
            )
        self.label_ids = tuple(vocabulary)
        if (
            not self.label_ids
            or any(type(label) is not int or label < 0 for label in self.label_ids)
            or self.label_ids != tuple(sorted(set(self.label_ids)))
        ):
            raise ValueError("label_ids must be sorted unique nonnegative integers")
        if any(set(r["raw_labels"]) - set(self.label_ids) for r in records):
            raise ValueError("Surface labels are outside the category vocabulary")
        voxel_ids = [record["shape_id"] for record in voxel_records]
        if len(set(voxel_ids)) != len(voxel_ids):
            raise ValueError("Duplicate voxel shape ID")
        if any(
            record["clean_split"] != split
            or record.get("synset", self.synset) != self.synset
            for record in voxel_records
        ):
            raise ValueError("Voxel split/category does not match surface labels")
        selected_ids = {key for key, r in by_id.items() if r["clean_split"] == split}
        if set(voxel_ids) != selected_ids:
            raise ValueError(
                "Voxel and surface shape IDs must match exactly in the split"
            )
        self.records = [by_id[shape_id] for shape_id in voxel_ids]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Read and authenticate all surface points for one matched shape.

        Args:
            index: Nonnegative position in the voxel manifest.

        Returns:
            Shape ID, float32 points [N,3], integer semantic indices [N] and
            the verified source SHA256. The normals in the source are unused.

        Raises:
            IndexError: If the index is invalid.
            ValueError: If source hash, dimensions, values or labels disagree
                with the manifest or the fixed category vocabulary.
        """
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        record = self.records[index]
        content = (self.root / record["source_points"]).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != record["source_points_sha256"]:
            raise ValueError(f"Surface source hash mismatch: {record['shape_id']}")
        values = np.loadtxt(io.BytesIO(content), dtype=np.float64, ndmin=2)
        if values.shape != (record["point_count"], 7) or not np.isfinite(values).all():
            raise ValueError("Expected finite [point_count,7] surface point data")
        raw_labels = values[:, 6]
        if not np.array_equal(raw_labels, np.floor(raw_labels)):
            raise ValueError("Surface label values must be integers")
        if np.unique(raw_labels).tolist() != record["raw_labels"]:
            raise ValueError("Surface label values disagree with manifest raw_labels")
        labels = np.searchsorted(self.label_ids, raw_labels).astype(np.int64)
        points = np.ascontiguousarray(values[:, [2, 1, 0]], dtype=np.float32)
        points[:, 2] *= -1
        if not np.isfinite(points).all():
            raise ValueError("Surface coordinates exceed float32 range")
        return {
            "shape_id": record["shape_id"],
            "points": torch.from_numpy(points),
            "labels": torch.from_numpy(labels),
            "source_points_sha256": digest,
        }
