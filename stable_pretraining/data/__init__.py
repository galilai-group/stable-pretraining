"""Data module for stable-pretraining.

This module provides dataset utilities, data loading, transformations,
and other data-related functionality for the stable-pretraining framework.
"""

from . import dataset_stats, gpu_transforms, sampler, synthetic_data, transforms
from .collate import Collator
from .datasets import Dataset, FromTorchDataset, HFDataset, Subset
from .gpu_transforms import GPUCompose, MultiView, StackedMultiView, ToDevice
from .download import bulk_download, download
from .module import DataModule
from .sampler import RandomBatchSampler, RepeatedRandomSampler, SupervisedBatchSampler
from .synthetic_data import (
    Categorical,
    ExponentialMixtureNoiseModel,
    ExponentialNormalNoiseModel,
    GMM,
    MinariEpisodeDataset,
    MinariStepsDataset,
    generate_perlin_noise_2d,
    perlin_noise_3d,
    swiss_roll,
)
from .utils import fold_views, random_split

# Video (Lance-backed) — guard against environments where the cv2 native lib
# can't load (e.g. missing libGL on a headless server with the GUI wheel
# installed). The library still works for everything else.
try:
    from .video import LanceVideoSegments, build_lance_video_dataset  # noqa: F401

    _VIDEO_AVAILABLE = True
except ImportError:
    _VIDEO_AVAILABLE = False

# Backward compatibility
static = dataset_stats
# Legacy imports - these modules are now consolidated
noise = synthetic_data  # noise generators are in synthetic_data
manifold = synthetic_data  # manifold datasets are in synthetic_data

__all__ = [
    # Modules
    "dataset_stats",
    "static",  # Backward compatibility
    "transforms",
    "gpu_transforms",
    "sampler",
    "synthetic_data",
    # Core classes
    "DataModule",
    "Collator",
    "Dataset",
    # GPU-side transforms (ToDevice + GPUCompose + view fan-out wrappers
    # are kornia-free; the kornia-backed ops live in ``gpu_transforms``
    # submodule)
    "ToDevice",
    "GPUCompose",
    "StackedMultiView",
    "MultiView",
    # Real data wrappers
    "FromTorchDataset",
    "HFDataset",
    "Subset",
    # Synthetic data generators
    "GMM",
    "MinariStepsDataset",
    "MinariEpisodeDataset",
    "swiss_roll",
    "generate_perlin_noise_2d",
    "perlin_noise_3d",
    # Noise models
    "Categorical",
    "ExponentialMixtureNoiseModel",
    "ExponentialNormalNoiseModel",
    # Samplers
    "SupervisedBatchSampler",
    "RepeatedRandomSampler",
    "RandomBatchSampler",
    # Utilities
    "fold_views",
    "random_split",
    "download",
    "bulk_download",
    # Legacy compatibility
    "noise",
    "manifold",
]

if _VIDEO_AVAILABLE:
    __all__.extend(["LanceVideoSegments", "build_lance_video_dataset"])
