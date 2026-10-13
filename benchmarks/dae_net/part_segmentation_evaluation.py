"""CPU metrics and global branch mappings for semantic part segmentation.

Inputs are integer branch/label counts, independent of models and data formats.
Semantic labels must be mapped to a fixed category vocabulary before counting.

Source and license: THIRD_PARTY_NOTICES.md (DAE-Net).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PartIoUResult:
    """Semantic IoUs with equal weight for every shape and category part.

    Attributes:
        intersection: Intersection counts, shaped [shapes, parts].
        union: Union counts, shaped [shapes, parts].
        shape_part_iou: Per-shape, per-part IoUs; double-empty parts score one.
        shape_miou: Mean across the fixed part vocabulary for each shape.
        per_part_iou: Mean across all shapes for each semantic part.
        category_miou: Mean of shape_miou, also equal to mean(per_part_iou).
        pooled_part_iou: Per-part IoU after pooling counts across shapes.
        gt_present_shapes: Number of shapes containing each ground-truth part.
        empty_union_shapes: Number of double-empty shapes for each part.
    """

    intersection: NDArray[np.int64]
    union: NDArray[np.int64]
    shape_part_iou: NDArray[np.float64]
    shape_miou: NDArray[np.float64]
    per_part_iou: NDArray[np.float64]
    category_miou: float
    pooled_part_iou: NDArray[np.float64]
    gt_present_shapes: NDArray[np.int64]
    empty_union_shapes: NDArray[np.int64]


@dataclass(frozen=True)
class BranchMappingResult:
    """A category-wide many-to-one mapping and scores recomputed from it.

    Attributes:
        mapping: Semantic label index for each branch, shaped [branches].
        protocol: Mapping algorithm identifier.
        unused_branches: Branch indices with no points in the fitting set.
        scores: Scores of the actual returned mapping on the fitting set.
        reference_reported_miou: Legacy rounded score tracked by the released
            DAE-Net search, or None for majority voting. This can disagree with
            scores.category_miou; it is not substituted for the actual score.
    """

    mapping: NDArray[np.int64]
    protocol: str
    unused_branches: tuple[int, ...]
    scores: PartIoUResult
    reference_reported_miou: float | None


def _integer_array(value: ArrayLike, name: str) -> NDArray[np.int64]:
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    if array.size and (np.any(array < 0) or np.max(array) > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must contain nonnegative int64 values")
    return array.astype(np.int64, copy=False)


def _counts(value: ArrayLike) -> NDArray[np.int64]:
    counts = _integer_array(value, "counts")
    if counts.ndim != 3 or any(size == 0 for size in counts.shape):
        raise ValueError("counts must have nonempty [shapes, branches, parts] axes")
    # Bound all sums, including union construction, before integer arithmetic.
    if int(counts.max()) * counts.size > np.iinfo(np.int64).max // 2:
        raise ValueError("counts are too large for safe int64 aggregation")
    if np.any(counts.sum(axis=(1, 2)) == 0):
        raise ValueError("Each shape must contain at least one labeled point")
    return counts


def branch_label_counts(
    branches: ArrayLike,
    labels: ArrayLike,
    *,
    num_branches: int,
    num_parts: int,
) -> NDArray[np.int64]:
    """Count branch/semantic-label pairs for one shape or a query chunk.

    Args:
        branches: One-dimensional integer branch IDs in [0, num_branches).
        labels: Matching semantic indices in [0, num_parts).
        num_branches: Positive number of predicted branches.
        num_parts: Fixed positive category vocabulary size, including parts
            absent from this shape or chunk.

    Returns:
        Integer counts shaped [num_branches, num_parts]. Sum chunk counts
        within each shape before stacking shapes and computing IoUs.

    Raises:
        ValueError: If dimensions, types, or indices are invalid.

    Note:
        Empty integer chunks return zero counts; a completely empty shape is
        rejected by score_part_iou. Original dataset label IDs need explicit
        remapping by the data adapter.
    """
    if any(type(n) is not int or n <= 0 for n in (num_branches, num_parts)):
        raise ValueError("num_branches and num_parts must be positive integers")
    branches = _integer_array(branches, "branches")
    labels = _integer_array(labels, "labels")
    if branches.ndim != 1 or branches.shape != labels.shape:
        raise ValueError("branches and labels must be matching one-dimensional arrays")
    if np.any(branches >= num_branches) or np.any(labels >= num_parts):
        raise ValueError("Branch or semantic label index is outside the vocabulary")
    return np.bincount(
        branches * num_parts + labels, minlength=num_branches * num_parts
    ).reshape(num_branches, num_parts)


def _score(counts: NDArray[np.int64], mapping: NDArray[np.int64]) -> PartIoUResult:
    shapes, _, parts = counts.shape
    confusion = np.zeros((shapes, parts, parts), dtype=np.int64)
    for branch, part in enumerate(mapping):
        confusion[:, part, :] += counts[:, branch, :]
    intersection = np.diagonal(confusion, axis1=1, axis2=2).copy()
    target = confusion.sum(axis=1)
    union = target + confusion.sum(axis=2) - intersection
    iou = np.ones((shapes, parts), dtype=np.float64)
    np.divide(intersection, union, out=iou, where=union != 0)
    pooled = np.ones(parts, dtype=np.float64)
    pooled_union = union.sum(axis=0)
    np.divide(
        intersection.sum(axis=0), pooled_union, out=pooled, where=pooled_union != 0
    )
    shape_miou = iou.mean(axis=1)
    return PartIoUResult(
        intersection=intersection,
        union=union,
        shape_part_iou=iou,
        shape_miou=shape_miou,
        per_part_iou=iou.mean(axis=0),
        category_miou=float(shape_miou.mean()),
        pooled_part_iou=pooled,
        gt_present_shapes=(target > 0).sum(axis=0),
        empty_union_shapes=(union == 0).sum(axis=0),
    )


def score_part_iou(counts: ArrayLike, mapping: ArrayLike) -> PartIoUResult:
    """Score a fixed global many-to-one mapping on labeled shapes.

    Args:
        counts: Nonnegative integer counts shaped [shapes, branches, parts].
            Each shape must contain at least one point.
        mapping: Integer semantic index for each branch, shaped [branches].

    Returns:
        Full-precision per-shape, per-part and category metrics in [0, 1].

    Raises:
        ValueError: If inputs have invalid shapes, indices or counts.

    Note:
        All vocabulary parts participate in each shape's mean. A part absent
        from both prediction and ground truth has IoU=1; a false-positive part
        has IoU=0. Shapes are equally weighted, regardless of their point count.
        Inputs are not modified. Mapping may have been fitted on another set.
    """
    counts = _counts(counts)
    mapping = _integer_array(mapping, "mapping")
    if mapping.shape != (counts.shape[1],) or np.any(mapping >= counts.shape[2]):
        raise ValueError("mapping must assign one valid semantic index to every branch")
    return _score(counts, mapping)


def fit_branch_mapping(
    counts: ArrayLike,
    *,
    protocol: Literal["majority", "dae_net_release_v1"] = "majority",
) -> BranchMappingResult:
    """Fit one many-to-one branch mapping across the supplied labeled shapes.

    Args:
        counts: Integer counts shaped [shapes, branches, fixed category parts].
        protocol: Global point-count majority voting, or compatibility with
            the pinned DAE-Net release's rounded, single-pass branch search.

    Returns:
        Mapping, unused branch IDs, actual fitting-set scores and, for the
        release protocol, the separately tracked reference reported score.

    Raises:
        ValueError: If counts or the protocol identifier are invalid.

    Note:
        Voting ties and unused branches initially select semantic index zero
        (the first maximum). Unused branches remain mapped to index zero.
        This fits using ground truth; callers must record the fitting split.
        The release resets a rejected candidate to the branch's original label,
        even after an earlier accepted candidate, so its tracked rounded score
        can disagree with its final mapping. Compatibility preserves both
        quantities explicitly. The pinned source is recorded in
        stable_pretraining/tests/unit/_dae_net_reference.py.
    """
    if protocol not in ("majority", "dae_net_release_v1"):
        raise ValueError(f"Unknown mapping protocol: {protocol}")
    counts = _counts(counts)
    votes = counts.sum(axis=0)
    mapping = votes.argmax(axis=1).astype(np.int64)
    unused = tuple(int(i) for i in np.flatnonzero(votes.sum(axis=1) == 0))
    reported = None
    if protocol == "dae_net_release_v1":
        reported_percent = np.round(_score(counts, mapping).category_miou * 1000) / 10
        for branch in range(counts.shape[1]):
            original = mapping[branch]
            for part in range(counts.shape[2]):
                if part == original:
                    continue
                mapping[branch] = part
                candidate = np.round(_score(counts, mapping).category_miou * 1000) / 10
                if candidate > reported_percent:
                    reported_percent = candidate
                else:
                    mapping[branch] = original
        reported = float(reported_percent / 100)
    return BranchMappingResult(
        mapping, protocol, unused, _score(counts, mapping), reported
    )


def propagate_confident_labels(
    points: ArrayLike,
    branches: ArrayLike,
    confidence: ArrayLike,
    *,
    threshold: float = 0.4,
) -> tuple[NDArray[np.int64], bool]:
    """Propagate the nearest confident point's branch to every surface point.

    Args:
        points: Finite coordinates shaped [points, 3].
        branches: Nonnegative integer branch indices shaped [points].
        confidence: Maximum existence-weighted occupancy per point in [0,1].
        threshold: Strict confidence threshold for the reference points.

    Returns:
        Propagated branch indices and whether no confident points were found.
        In that case the original branch labels are copied unchanged.

    Raises:
        ValueError: If input dimensions or values are invalid.

    Note:
        Uses sklearn KDTree with leaf_size=8, matching the DAE-Net release.
        Propagation spans the entire shape, not individual query chunks.
    """
    points = np.asarray(points)
    branches = _integer_array(branches, "branches")
    confidence = np.asarray(confidence)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) == 0
        or branches.shape != (len(points),)
        or confidence.shape != (len(points),)
        or points.dtype.kind not in "fiu"
        or confidence.dtype.kind not in "fiu"
        or not np.isfinite(points).all()
        or not np.isfinite(confidence).all()
        or np.any(confidence < 0)
        or np.any(confidence > 1)
        or not 0 <= threshold < 1
    ):
        raise ValueError("Invalid surface points, confidence, labels or threshold")
    valid = confidence > threshold
    if not valid.any():
        return branches.copy(), True
    from sklearn.neighbors import KDTree

    tree = KDTree(points[valid], leaf_size=8)
    _, nearest = tree.query(points)
    return branches[valid][nearest[:, 0]], False
