"""LeVJEPA's token loader: every view's kept patches, built from only the JPEG MCUs those patches read, on the GPU.

    from stable_pretraining.data.levjepa import SharedTokenLoader, ViewRecipe

    loader = SharedTokenLoader("/vol/frames.lance", batch=64, index="shared")  # workers: from the node
    global_patches, global_idx, local_patches, local_idx = next(loader)

The store is a converted frame store (stable_datasets.framestore.convert). Modules:
    recipe     ViewRecipe: v2's views (sizes, scales, jitter, kept share), V2
    sampling   view parameters and each kept patch's footprint in the stored frame (CPU)
    splice     the MCUs under the kept patches -> one packed JPEG per clip (CPU, numba)
    views      TokenViews: packed JPEGs -> kept patches (GPU)
    kernels    the fused GPU stage (Triton): MCU scatter, window squash + roi_align, jitter
    loader     TokenLoader (CPU workers), SharedTokenLoader (+ GPU-stage server processes and a CUDA-IPC ring)
"""

__all__ = ["V2", "SharedTokenLoader", "TokenLoader", "TokenViews", "ViewRecipe"]

_WHERE = {
    "V2": "recipe",
    "ViewRecipe": "recipe",
    "SharedTokenLoader": "loader",
    "TokenLoader": "loader",
    "TokenViews": "views",
}


def __getattr__(name):  # lazy: importing the package does not import torch
    if name not in _WHERE:
        raise AttributeError(
            f"module 'stable_pretraining.data.levjepa' has no attribute {name!r}"
        )
    import importlib

    return getattr(importlib.import_module(f".{_WHERE[name]}", __name__), name)
