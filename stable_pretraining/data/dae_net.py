"""Manifest-backed occupancy data for DAE-Net."""

import json
import math
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .module import DataModule
from ..utils.distributed import get_rank, get_world_size


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "DAENetDataset requires the optional dependency 'h5py'"
        ) from exc
    return h5py


class DAENetExposureSampler(Sampler[tuple[int, int]]):
    """Repeat ordered dataset passes with reproducible query offsets.

    Args:
        dataset: Nonempty DAE-Net dataset with query resolution metadata.
        shapes_per_epoch: Minimum number of shape exposures requested per epoch.
            The actual count rounds up to a whole number of dataset passes.
        seed: Nonnegative seed for query-offset sampling.

    Raises:
        ValueError: If the dataset is empty, the quota is not a positive integer,
            or the seed is not a nonnegative integer.

    Note:
        Each iteration starts at record zero and emits ``(record_index, offset)``.
        Resolution-64 offsets are sampled from 0 through 3; resolution-32
        offsets are always zero. Set the epoch before iterating to change the
        reproducible offset sequence. This does not update the epoch automatically.
        Use ``drop_last=False`` in the DataLoader to retain every exposure.
    """

    def __init__(
        self, dataset: "DAENetDataset", *, shapes_per_epoch: int, seed: int = 0
    ) -> None:
        if type(shapes_per_epoch) is not int or shapes_per_epoch < 1:
            raise ValueError("shapes_per_epoch must be a positive integer")

        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")

        self.num_shapes = len(dataset)
        if self.num_shapes == 0:
            raise ValueError("dataset must contain at least one shape")

        self.shapes_per_epoch = shapes_per_epoch
        self.repeat_times = math.ceil(shapes_per_epoch / self.num_shapes)

        self.seed = seed
        self.epoch = 0

        self.query_resolutions = tuple(
            record["query_resolution"] for record in dataset.records
        )

    def __len__(self) -> int:
        """Return the actual number of exposures after whole-pass rounding."""
        return self.num_shapes * self.repeat_times

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch used to generate query offsets.

        Args:
            epoch: Nonnegative, zero-based epoch number.

        Raises:
            ValueError: If epoch is not a nonnegative integer.
        """
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")

        self.epoch = epoch

    def __iter__(self) -> Iterator[tuple[int, int]]:
        """Yield ordered record indices and reproducible query offsets."""
        # Rebuilding a local generator makes repeated iteration reproducible.
        seed_sequence = np.random.SeedSequence([self.seed, self.epoch])
        rng = np.random.default_rng(seed_sequence)

        for _ in range(self.repeat_times):
            for index, resolution in enumerate(self.query_resolutions):
                if resolution == 64:
                    offset = int(rng.integers(0, 4))
                else:
                    offset = 0

                yield index, offset


class DAENetDistributedBatchSampler(Sampler[list[tuple[int, int]]]):
    """Shard complete global exposure batches without padding or dropping.

    Args:
        sampler: The full, deterministic exposure sequence.
        batch_size: Global number of shapes per optimizer update.
        rank: Explicit rank for testing; otherwise resolved at iteration time.
        world_size: Explicit group size; otherwise resolved at iteration time.

    Note:
        Rank-order concatenation reconstructs each original global batch.
        Unequal nonempty shards are supported. A smaller-than-world-size final
        batch fails before yielding anything. Disable Lightning's automatic
        distributed sampler replacement when using this batch sampler.
    """

    def __init__(
        self,
        sampler: DAENetExposureSampler,
        batch_size: int,
        *,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.sampler = sampler
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.drop_last = False

    def __len__(self) -> int:
        """Return the number of global optimizer updates."""
        return math.ceil(len(self.sampler) / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        """Select the exposure sequence for an epoch.

        Args:
            epoch: Nonnegative training epoch.
        """
        self.sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        """Yield contiguous rank shards of each global batch."""
        rank = get_rank() if self.rank is None else self.rank
        world = get_world_size() if self.world_size is None else self.world_size
        if world < 1 or not 0 <= rank < world:
            raise ValueError("Invalid rank/world_size")
        tail = len(self.sampler) % self.batch_size
        if self.batch_size < world or (tail and tail < world):
            raise ValueError(
                "Every global batch must contain at least world_size shapes"
            )
        sequence = iter(self.sampler)
        while batch := list(islice(sequence, self.batch_size)):
            size, extra = divmod(len(batch), world)
            start = rank * size + min(rank, extra)
            yield batch[start : start + size + (rank < extra)]


class DAENetDataset(Dataset):
    """Read audited DAE-Net voxel and occupancy-query records.

    Args:
        root: Dataset root used to resolve source paths in the manifest.
        manifest: Path to a JSONL manifest.
        split: Expected split: ``train``, ``val``, or ``test``.
        load_queries: Whether to read occupancy queries and targets. If False,
            read only voxels and return them with the shape ID.

    Note:
        Integer indices use a fixed query offset of zero. A sampler may
        supply ``(record_index, query_offset)`` to select another offset.
        This dataset performs neither repetition nor random sampling.
    """

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        *,
        split: str,
        load_queries: bool = True,
    ) -> None:
        if split not in ["train", "val", "test"]:
            raise ValueError(
                f"Split '{split}' is not valid. Choose from 'train', 'val', 'test'."
            )
        if type(load_queries) is not bool:
            raise ValueError("load_queries must be a boolean")

        self.root = Path(root)
        self.manifest = Path(manifest)
        self.split = split
        self.load_queries = load_queries

        with self.manifest.open(encoding="utf-8") as stream:
            self.records = [json.loads(line) for line in stream if line.strip()]

        if not self.records:
            raise ValueError("The manifest is empty")

        required = {
            "shape_id",
            "source_hdf5",
            "row_index",
            "clean_split",
            "query_resolution",
        }
        seen_ids: set[str] = set()

        for record in self.records:
            missing = required - record.keys()
            if missing:
                raise ValueError(f"Missing manifest fields: {sorted(missing)}")

            shape_id = record["shape_id"]
            if not isinstance(shape_id, str) or not shape_id:
                raise ValueError("shape_id must be a nonempty string")
            if shape_id in seen_ids:
                raise ValueError(f"Duplicate shape_id: {shape_id}")
            seen_ids.add(shape_id)

            if record["clean_split"] != split:
                raise ValueError(f"Unexpected split for {shape_id}")

            row = record["row_index"]
            if type(row) is not int or row < 0:
                raise ValueError("row_index must be a nonnegative integer")

            resolution = record["query_resolution"]
            if type(resolution) is not int or resolution not in {32, 64}:
                raise ValueError("query_resolution must be 32 or 64")

    def __len__(self) -> int:
        """Return the number of manifest records."""
        return len(self.records)

    def __getitem__(
        self,
        index: int | tuple[int, int],
    ) -> dict[str, Any]:
        """Read a shape and optionally its paired occupancy queries.

        Args:
            index: Record index, or a tuple of record index and query offset.
                Resolution 64 accepts offsets 0 through 3. Resolution 32
                accepts only offset 0. Valid offsets have no effect when
                ``load_queries=False``.

        Returns:
            Dictionary containing float32 voxels and the original shape ID.
            Includes float32 points and occupancy targets if queries are enabled.

        Raises:
            ImportError: If the optional HDF5 dependency is unavailable.
            IndexError: If the record index is outside the manifest.
            ValueError: If the offset or source arrays are invalid.
        """
        record_index, offset = index if isinstance(index, tuple) else (index, 0)

        if type(record_index) is not int:
            raise TypeError("record_index must be an integer")

        if not 0 <= record_index < len(self.records):
            raise IndexError(record_index)

        record = self.records[record_index]
        resolution = record["query_resolution"]
        allowed_offsets = range(4) if resolution == 64 else range(1)
        if type(offset) is not int or offset not in allowed_offsets:
            raise ValueError(f"Invalid offset for resolution {resolution}: {offset}")

        source = self.root / record["source_hdf5"]
        row = record["row_index"]

        h5py = _import_h5py()
        # No live HDF handle is shared or serialized between workers.
        with h5py.File(source, "r") as data:
            voxels = data["voxels"][row]
            if self.load_queries:
                points = data[f"points_{resolution}"][row]
                occupancy = data[f"values_{resolution}"][row]

        if voxels.shape != (64, 64, 64, 1):
            raise ValueError(f"Unexpected voxel shape in {source}: {voxels.shape}")
        if voxels.dtype != np.uint8:
            raise ValueError("Expected uint8 arrays from the preprocessed dataset")
        if np.any(voxels > 1):
            raise ValueError("Voxels must be binary")

        voxels = np.ascontiguousarray(voxels.transpose(3, 0, 1, 2), dtype=np.float32)
        sample = {
            "voxels": torch.from_numpy(voxels),
            "shape_id": record["shape_id"],
        }
        if not self.load_queries:
            return sample

        query_count = 32768 if resolution == 64 else 8192
        if points.shape != (query_count, 3):
            raise ValueError(f"Unexpected query shape in {source}: {points.shape}")
        if occupancy.shape != (query_count, 1):
            raise ValueError(f"Unexpected target shape in {source}: {occupancy.shape}")

        if any(array.dtype != np.uint8 for array in [points, occupancy]):
            raise ValueError("Expected uint8 arrays from the preprocessed dataset")
        if np.any(points >= resolution):
            raise ValueError("Query coordinates exceed the sampling grid")
        if np.any(occupancy > 1):
            raise ValueError("Occupancy values must be binary")

        if resolution == 64:
            points = points[offset::4]
            occupancy = occupancy[offset::4]

        points = (points.astype(np.float32) + 0.5) / resolution - 0.5
        occupancy = np.ascontiguousarray(occupancy, dtype=np.float32)

        sample["points"] = torch.from_numpy(points)
        sample["occupancy"] = torch.from_numpy(occupancy)
        return sample


class DAENetDataModule(DataModule):
    """Assemble manifest-backed DAE-Net training and evaluation loaders.

    Args:
        root: Dataset root used to resolve manifest source paths.
        manifest_dir: Directory containing train, val, and test JSONL files.
        batch_size: Number of shapes per batch.
        shapes_per_epoch: Minimum requested training exposures per epoch.
        seed: Seed used by the training query-offset sampler.
        num_workers: Number of DataLoader worker processes.
        distributed: Shard global training batches across DDP ranks. Requires
            ``Trainer(use_distributed_sampler=False)``. Evaluation and usage
            loaders remain full traversals.

    Note:
        Lightning updates the training sampler's epoch. The separate
        ``usage_dataloader()`` visits training records once with voxels only;
        callers must invoke it explicitly for branch-usage checks.
    """

    def __init__(
        self,
        root: str | Path,
        manifest_dir: str | Path,
        *,
        batch_size: int = 16,
        shapes_per_epoch: int = 12500,
        seed: int = 0,
        num_workers: int = 0,
        distributed: bool = False,
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        if type(num_workers) is not int or num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        root = Path(root)
        manifest_dir = Path(manifest_dir)

        train_dataset = DAENetDataset(
            root,
            manifest_dir / "train.jsonl",
            split="train",
        )
        val_dataset = DAENetDataset(
            root,
            manifest_dir / "val.jsonl",
            split="val",
        )
        test_dataset = DAENetDataset(
            root,
            manifest_dir / "test.jsonl",
            split="test",
        )
        usage_dataset = DAENetDataset(
            root, manifest_dir / "train.jsonl", split="train", load_queries=False
        )

        train_sampler = DAENetExposureSampler(
            train_dataset,
            shapes_per_epoch=shapes_per_epoch,
            seed=seed,
        )

        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": False,
            "drop_last": False,
        }

        if distributed:
            self.train_loader = DataLoader(
                train_dataset,
                batch_sampler=DAENetDistributedBatchSampler(train_sampler, batch_size),
                generator=torch.Generator().manual_seed(seed),
                num_workers=num_workers,
            )
        else:
            self.train_loader = DataLoader(
                train_dataset,
                sampler=train_sampler,
                generator=torch.Generator().manual_seed(seed),
                **loader_kwargs,
            )

        self.val_loader = DataLoader(
            val_dataset,
            generator=torch.Generator().manual_seed(seed + 1),
            **loader_kwargs,
        )

        self.test_loader = DataLoader(
            test_dataset,
            generator=torch.Generator().manual_seed(seed + 2),
            **loader_kwargs,
        )

        self.usage_loader = DataLoader(
            usage_dataset,
            generator=torch.Generator().manual_seed(seed + 3),
            **loader_kwargs,
        )

        super().__init__(
            train=self.train_loader,
            val=self.val_loader,
            test=self.test_loader,
        )

        self.train_sampler = train_sampler

    def usage_dataloader(self) -> DataLoader:
        """Return a single-pass voxel loader over the training records.

        Returns:
            Sequential loader yielding float32 voxels and shape IDs, including
            the final partial batch. Query arrays are not loaded.
        """
        return self.usage_loader
