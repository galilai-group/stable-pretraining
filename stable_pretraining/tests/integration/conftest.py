"""Shared, versioned inputs for integration tests."""

from collections.abc import Callable
from functools import partial

import pytest
from torch.utils.data import Dataset

import stable_pretraining as spt


@pytest.fixture(scope="session")
def imagenette_dataset() -> Callable[..., Dataset]:
    """Return an HFDataset factory using a fixed Imagenette 160px snapshot.

    The Hub's generated ``refs/convert/parquet`` branch was deleted. Load
    committed Parquet files directly to avoid that branch and the legacy
    dataset script. Explicit paths also prevent mixing all three resolutions.
    """
    revision = "4b23ffb92a8029db9958bdfdbd6978427d09b1a0"
    base = f"https://huggingface.co/datasets/frgfm/imagenette/resolve/{revision}"
    return partial(
        spt.data.HFDataset,
        "parquet",
        data_files={
            split: f"{base}/160px/{split}-00000-of-00001.parquet"
            for split in ("train", "validation")
        },
    )
