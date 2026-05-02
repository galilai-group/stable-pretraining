"""Dataset classes for real data sources.

This module provides dataset wrappers and utilities for working with real data sources
including PyTorch datasets, HuggingFace datasets (both map-style and streaming),
and dataset subsets. All wrappers produce dict-based samples with support for
transforms, automatic ``sample_idx`` injection, and PyTorch Lightning trainer
integration (``global_step``, ``current_epoch``).

Typical usage::

    from stable_pretraining.data.datasets import HFDataset

    # Map-style (downloads / caches locally)
    ds = HFDataset("imagenet-1k", split="train", transform=my_transform)

    # Streaming (no disk usage)
    ds = HFDataset("imagenet-1k", split="train", streaming=True, transform=my_transform)
    ds.shuffle(seed=42, buffer_size=10_000)

Both return objects that PyTorch ``DataLoader`` and Lightning ``Trainer`` handle
correctly without any special flags.
"""

from pathlib import Path
import time
from collections.abc import Sequence

import lightning as pl
import torch
from loguru import logger as logging
from datasets import config as hf_config
from ..utils import with_hf_retry_ratelimit


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class DatasetMixin:
    """Shared logic for all dataset types.

    Provides transform application, PyTorch Lightning trainer integration
    (injecting ``global_step`` and ``current_epoch`` into every sample), and
    a unified ``process_sample`` pipeline.
    """

    def __init__(self, transform=None):
        self.transform = transform
        self._trainer = None

    def set_pl_trainer(self, trainer: pl.Trainer):
        """Attach a Lightning trainer so its state is injected into samples."""
        self._trainer = trainer

    def __getstate__(self):
        # Drop the trainer back-reference. Pickle-walking it reaches
        # `trainer.train_dataloader._iterator`, a
        # `_MultiProcessingDataLoaderIter`, which raises on __getstate__.
        # Spawn-mode DataLoader workers therefore can't serialise any
        # dataset that has had `set_pl_trainer` called on it.
        # Workers see only a snapshot of trainer state at spawn time
        # anyway; `process_sample` already handles `_trainer is None`.
        state = self.__dict__.copy()
        state["_trainer"] = None
        return state

    def process_sample(self, sample, **kwargs):
        """Run a raw sample dict through trainer-injection and transforms.

        Args:
            sample: Dict-like sample from the underlying dataset.
            **kwargs: Extra key/value pairs merged into *sample* before
                transforms (e.g. ``view_idx``).

        Returns:
            The (possibly transformed) sample dict.
        """
        for k, v in kwargs.items():
            sample[k] = v
        if self._trainer is not None:
            if "global_step" in sample:
                raise ValueError("'global_step' is a reserved key")
            if "current_epoch" in sample:
                raise ValueError("'current_epoch' is a reserved key")
            sample["global_step"] = self._trainer.global_step
            sample["current_epoch"] = self._trainer.current_epoch
        if self.transform:
            sample = self.transform(sample)
        return sample


class Dataset(DatasetMixin, torch.utils.data.Dataset):
    """Base map-style dataset with transform and trainer support."""

    def __init__(self, transform=None):
        DatasetMixin.__init__(self, transform)

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class IterableDataset(DatasetMixin, torch.utils.data.IterableDataset):
    """Base iterable (streaming) dataset with transform and trainer support."""

    def __init__(self, transform=None):
        DatasetMixin.__init__(self, transform)

    def __iter__(self):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Subset
# ---------------------------------------------------------------------------


class Subset(Dataset):
    """Subset of a dataset at specified indices.

    All attributes and methods of the wrapped dataset are accessible directly
    on the subset via attribute proxying. For example, if the underlying dataset
    has a ``column_names`` property or a ``custom_method()``, they can be called
    as ``subset.column_names`` or ``subset.custom_method()`` respectively.

    Args:
        dataset: The whole dataset.
        indices: Indices in the whole set selected for the subset.
    """

    dataset: Dataset
    indices: Sequence[int]

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: list[int]) -> list:
        """Batched sampling support (see ``torch.utils.data._utils.fetch``)."""
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])
        else:
            return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        # Don't proxy dunders to the wrapped dataset. Pickle/copy/etc. on
        # the Subset must use Subset's own machinery, not the inner ds's.
        # On Python <3.11 ``object.__getstate__`` doesn't exist, so a bare
        # proxy makes pickle read the inner dataset's ``__getstate__`` and
        # serialize the wrong state (inner's ``__dict__`` under Subset's
        # class), breaking spawn-mode DataLoader workers.
        if name == "dataset" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self.dataset, name)


# ---------------------------------------------------------------------------
# PyTorch dataset wrapper
# ---------------------------------------------------------------------------


class FromTorchDataset(Dataset):
    """Wrapper that converts a positional-return PyTorch dataset into dict samples.

    Args:
        dataset: PyTorch dataset to wrap.
        names: List of names for each element returned by the dataset.
        transform: Optional transform applied to the dict sample.
        add_sample_idx: If ``True``, adds a ``sample_idx`` field to each sample.
    """

    def __init__(self, dataset, names, transform=None, add_sample_idx=True):
        super().__init__(transform)
        self.dataset = dataset
        self.names = names
        self.add_sample_idx = add_sample_idx

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample = dict(zip(self.names, sample))
        if self.add_sample_idx:
            sample["sample_idx"] = idx
        return self.process_sample(sample)

    def __len__(self):
        return len(self.dataset)

    @property
    def column_names(self):
        columns = list(self.names)
        if self.add_sample_idx and "sample_idx" not in columns:
            columns.append("sample_idx")
        return columns


# ---------------------------------------------------------------------------
# HuggingFace dataset wrappers
# ---------------------------------------------------------------------------


class HFMapDataset(Dataset):
    """Map-style wrapper around a HuggingFace ``datasets.Dataset``.

    Supports random access, ``len()``, and sampler-based shuffling.
    A ``sample_idx`` column is added automatically.

    This class is not intended to be instantiated directly; use the
    :func:`HFDataset` factory instead.
    """

    def __init__(
        self, dataset, transform=None, rename_columns=None, remove_columns=None
    ):
        super().__init__(transform)
        dataset = dataset.add_column("sample_idx", list(range(dataset.num_rows)))
        if rename_columns:
            for k, v in rename_columns.items():
                dataset = dataset.rename_column(k, v)
        if remove_columns:
            dataset = dataset.remove_columns(remove_columns)
        self.dataset = dataset

    def __getitem__(self, idx):
        extra = {}
        if type(idx) is tuple:
            extra["view_idx"] = idx[1]
            idx = idx[0]
        return self.process_sample(self.dataset[idx], **extra)

    def __len__(self):
        return self.dataset.num_rows

    def shuffle(self, seed=42, **kwargs):
        """Shuffle the dataset in-place.

        Args:
            seed: Random seed for reproducibility.
            **kwargs: Ignored (accepted for API compatibility with the
                iterable variant).

        Returns:
            ``self`` for chaining.
        """
        self.dataset = self.dataset.shuffle(seed=seed)
        return self

    @property
    def column_names(self):
        return self.dataset.column_names


class HFIterableDataset(IterableDataset):
    """Streaming wrapper around a HuggingFace ``datasets.IterableDataset``.

    Supports lazy iteration with buffer-based shuffling. Because it inherits
    from :class:`torch.utils.data.IterableDataset`, PyTorch ``DataLoader`` and
    Lightning ``Trainer`` handle it correctly without a ``DistributedSampler``
    or ``__len__``.

    A ``sample_idx`` field is injected via ``.map(with_indices=True)``.

    This class is not intended to be instantiated directly; use the
    :func:`HFDataset` factory instead.
    """

    def __init__(
        self, dataset, transform=None, rename_columns=None, remove_columns=None
    ):
        super().__init__(transform)
        dataset = dataset.map(
            lambda sample, idx: {**sample, "sample_idx": idx}, with_indices=True
        )
        if rename_columns:
            for k, v in rename_columns.items():
                dataset = dataset.rename_column(k, v)
        if remove_columns:
            dataset = dataset.remove_columns(remove_columns)
        self.dataset = dataset

    def __iter__(self):
        for sample in self.dataset:
            yield self.process_sample(sample)

    def shuffle(self, seed=42, buffer_size=10_000):
        """Buffer-shuffle the streaming dataset in-place.

        Args:
            seed: Random seed for reproducibility.
            buffer_size: Number of samples to buffer for shuffling. Larger
                values give better randomness at the cost of memory.

        Returns:
            ``self`` for chaining.
        """
        self.dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)
        return self

    @property
    def column_names(self):
        return self.dataset.column_names


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def HFDataset(
    *args, transform=None, rename_columns=None, remove_columns=None, **kwargs
):
    """Create a HuggingFace dataset wrapper.

    Automatically chooses map-style or streaming based on
    ``streaming=True/False`` in *kwargs*.

    The returned object is either an :class:`HFMapDataset` (subclass of
    :class:`torch.utils.data.Dataset`) or an :class:`HFIterableDataset`
    (subclass of :class:`torch.utils.data.IterableDataset`), so PyTorch
    ``DataLoader`` and Lightning ``Trainer`` handle both correctly out of
    the box.

    Args:
        *args: Positional arguments forwarded to ``datasets.load_dataset``
            (typically the dataset name/path).
        transform: Optional transform applied to every sample dict.
        rename_columns: Optional ``{old: new}`` mapping of columns to rename.
        remove_columns: Optional list of column names to drop.
        **kwargs: Keyword arguments forwarded to ``datasets.load_dataset``
            (e.g. ``split``, ``streaming``, ``data_dir``).

    Returns:
        An :class:`HFMapDataset` or :class:`HFIterableDataset` instance.

    Example::

        # Map-style
        ds = HFDataset("imagenet-1k", split="train")
        print(len(ds))  # works

        # Streaming
        ds = HFDataset("imagenet-1k", split="train", streaming=True)
        ds.shuffle(seed=42, buffer_size=10_000)
        for sample in ds:
            ...
    """
    import datasets

    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        s = int(torch.distributed.get_rank()) * 2
        logging.info(
            f"Sleeping for {s}s to avoid race condition of dataset cache"
            " (see https://github.com/huggingface/transformers/issues/15976)"
        )
        time.sleep(s)

    if "storage_options" not in kwargs:
        logging.warning(
            "No storage_options provided — adding a default timeout to avoid hanging"
        )
        from aiohttp import ClientTimeout

        kwargs["storage_options"] = {
            "client_kwargs": {"timeout": ClientTimeout(total=3600)}
        }

    hf_path = kwargs.get("path", args[0] if len(args) > 0 else None)
    if not isinstance(hf_path, str):
        raise ValueError("Only string dataset path/name is supported")

    load_dataset_fn = datasets.load_dataset
    if Path(hf_path, hf_config.DATASET_STATE_JSON_FILENAME).exists():
        logging.info(f"Loading dataset with load_from_disk: {hf_path}")
        load_dataset_fn = datasets.load_from_disk

    dataset = with_hf_retry_ratelimit(load_dataset_fn, *args, **kwargs)

    if isinstance(dataset, datasets.IterableDataset):
        return HFIterableDataset(dataset, transform, rename_columns, remove_columns)
    return HFMapDataset(dataset, transform, rename_columns, remove_columns)
