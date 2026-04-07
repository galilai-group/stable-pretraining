"""Utility functions for data manipulation and processing.

This module provides utility functions for working with datasets, including
view folding for contrastive learning and dataset splitting.
"""

import itertools
import math
import warnings
from collections.abc import Sequence
from typing import Optional, Union, cast

import torch
from torch import Generator, default_generator, randperm

from .datasets import Dataset, Subset
import os
from loguru import logger


def get_num_workers():
    """Automatically determine the optimal number of DataLoader workers.

    This function computes the ideal number of worker processes for PyTorch
    DataLoaders based on available CPU resources and distributed training
    configuration. It provides a zero-configuration approach that works
    reliably across different environments.

    The calculation logic:
        1. Detect CPUs available to this process (respects affinity/cgroups)
        2. Divide by world_size if using DDP (each rank spawns its own workers)
        3. Return the result (always >= 1)

    Returns:
        int: Number of DataLoader workers to use. Minimum value is 1.

    Notes:
        - Uses `os.sched_getaffinity(0)` on Linux to respect CPU affinity masks
          set by job schedulers (SLURM), containers (Docker), or taskset.
        - Falls back to `os.cpu_count()` on macOS/Windows.
        - In DDP mode, automatically divides by world_size since each process
          independently spawns workers.
        - Should be called AFTER distributed initialization for accurate results.

    Examples:
        >>> # Simple usage
        >>> num_workers = get_num_workers()
        >>> loader = DataLoader(dataset, num_workers=num_workers)

        >>> # In a Lightning DataModule
        >>> class MyDataModule(L.LightningDataModule):
        ...     def train_dataloader(self):
        ...         return DataLoader(
        ...             self.train_dataset,
        ...             num_workers=get_num_workers(),
        ...             shuffle=True,
        ...         )

        >>> # Example outputs:
        >>> # - 16 CPUs, single GPU: returns 16
        >>> # - 32 CPUs, 4 GPUs (DDP): returns 8 per GPU
        >>> # - 8 CPUs, 8 GPUs (DDP): returns 1 per GPU

    See Also:
        - PyTorch DataLoader: https://pytorch.org/docs/stable/data.html
        - CPU affinity: https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html
    """
    logger.debug("Starting automatic num_workers detection")

    # Step 1: Detect available CPUs
    try:
        num_cpus = len(os.sched_getaffinity(0))
        logger.info(
            f"  Detected {num_cpus} CPUs via sched_getaffinity (respects affinity mask)"
        )
        logger.debug(
            "Using sched_getaffinity ensures we respect SLURM/Docker/cgroup limits"
        )
    except AttributeError:
        # Fallback for systems without sched_getaffinity (macOS, Windows)
        num_cpus = os.cpu_count()
        if num_cpus is None:
            logger.warning("! os.cpu_count() returned None, defaulting to 1 CPU")
            num_cpus = 1
        else:
            logger.info(
                f"  Detected {num_cpus} CPUs via os.cpu_count() (fallback for non-Linux)"
            )

    logger.debug(f"Total CPUs available to this process: {num_cpus}")

    # Step 2: Check for distributed training
    world_size = 1
    rank = 0
    is_distributed = False

    if torch.distributed.is_available():
        logger.debug(
            "torch.distributed is available, checking initialization status..."
        )

        if torch.distributed.is_initialized():
            is_distributed = True
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            logger.info(f"  DDP detected: world_size={world_size}, rank={rank}")
            logger.debug(
                f"Each of {world_size} processes will spawn its own DataLoader workers"
            )
        else:
            logger.debug(
                "torch.distributed available but not initialized (single-process training)"
            )
    else:
        logger.debug("torch.distributed not available (CPU-only PyTorch build)")

    # Step 3: Calculate workers
    if is_distributed and world_size > 1:
        num_workers = num_cpus // world_size
        logger.info(
            f"  DDP mode: {num_cpus} CPUs / {world_size} ranks = {num_workers} workers per rank"
        )

        if num_workers == 0:
            logger.warning(
                f"! Only {num_cpus} CPUs for {world_size} ranks results in 0 workers/rank. "
                f"Setting to 1 worker minimum."
            )
            num_workers = 1
    else:
        num_workers = num_cpus
        logger.debug(
            f"Single-process mode: using all {num_cpus} CPUs for DataLoader workers"
        )

    # Step 4: Ensure minimum of 1
    num_workers = max(1, num_workers)

    # Final summary
    if is_distributed:
        total_workers = num_workers * world_size
        logger.success(
            f"✓ Final configuration: {num_workers} workers/rank x {world_size} ranks = {total_workers} total workers"
        )
    else:
        logger.success(f"✓ Final configuration: {num_workers} DataLoader workers")

    logger.debug(f"Returning num_workers={num_workers}")
    logger.debug("num_workers detection complete\n")

    return num_workers


def fold_views(tensor, idx):
    """Fold a tensor containing multiple views back into separate views.

    Args:
        tensor: Tensor containing concatenated views
        idx: Sample indices to determine view boundaries

    Returns:
        Tuple of tensors, one for each view
    """
    sidx = torch.argsort(idx, stable=True)

    _, counts = torch.unique_consecutive(idx[sidx], return_counts=True)
    if not counts.min().eq(counts.max()):
        raise RuntimeError(
            "counts are not the same for all samples!\n"
            "This typically occurs when batch size and number of views\n"
            "are not divisible"
        )
    n_views = counts[0].item()
    fold_shape = (tensor.size(0) // n_views, n_views)
    t = tensor[sidx].view(*fold_shape, *tensor.shape[1:])
    return t.unbind(dim=1)


def random_split(
    dataset: Dataset,
    lengths: Sequence[Union[int, float]],
    generator: Optional[Generator] = default_generator,
) -> list[Subset]:
    r"""Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> generator1 = torch.Generator().manual_seed(42)
        >>> generator2 = torch.Generator().manual_seed(42)
        >>> random_split(range(10), [3, 7], generator=generator1)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
    lengths = cast(Sequence[int], lengths)
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(itertools.accumulate(lengths), lengths)
    ]


def apply_masks(x: torch.Tensor, *masks: torch.Tensor) -> torch.Tensor:
    r"""Apply one or more masks to a batch of patched images.

    This function is generalized to accept any number of mask tensors.
    If a single mask is provided, the output shape is `[B, K, D]`. If `M`
    masks are provided, the function creates `M` masked views
    and concatenates them along the batch dimension, resulting in an
    output of shape `[B*M, K, D]`.

    Example:
        >>> # xdoctest: +SKIP
        >>> x = torch.randn(4, 196, 128)
        >>> mask1 = torch.randint(0, 196, (4, 50))
        >>> mask2 = torch.randint(0, 196, (4, 50))
        >>> # Single mask case
        >>> single_view = apply_masks(x, mask1)
        >>> single_view.shape
        torch.Size([4, 50, 128])
        >>> # Multi-mask case
        >>> multi_view = apply_masks(x, mask1, mask2)
        >>> multi_view.shape
        torch.Size([8, 50, 128])

    Args:
        x (torch.Tensor): Input tensor of patches with shape `[B, N, D]`.
        *masks (torch.Tensor): A variable number of mask tensors, each a
            tensor of indices with shape `[B, K]`.

    Returns:
        torch.Tensor: The tensor of selected patches. The shape will be
            `[B, K, D]` for a single mask, or `[B*M, K, D]` for `M` masks.

    Raises:
        ValueError: If no masks are provided.
    """
    if not masks:
        raise ValueError("At least one mask tensor must be provided.")

    B, N, D = x.shape
    M = len(masks)

    idx = torch.stack([m.to(x.device, dtype=torch.long) for m in masks], dim=1)
    K = idx.size(-1)

    x_expanded = x.unsqueeze(1).expand(-1, M, -1, -1)
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, D)
    out = x_expanded.gather(2, idx_expanded)

    return out.reshape(B * M, K, D)
