"""Small models and example loading shared by All4One tests."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from torch import nn

import stable_pretraining as spt


def tiny_all4one(**kwargs: Any) -> spt.All4One:
    """Build a small real All4One model with deterministic attention.

    Args:
        **kwargs: Constructor options overriding the small test defaults.

    Returns:
        A model with BatchNorm, an eleven-entry queue, and no attention dropout.
    """
    options = dict(
        backbone=nn.Sequential(nn.Linear(6, 8), nn.BatchNorm1d(8), nn.ReLU()),
        feature_dim=8,
        projection_dim=8,
        projector_hidden_dim=12,
        predictor_hidden_dim=12,
        queue_size=11,
        num_heads=2,
        num_layers=1,
        dropout=0,
    )
    options.update(kwargs)
    return spt.All4One(**options)


def load_all4one_example() -> ModuleType:
    """Load the repository example without running its CLI entry point.

    Returns:
        Module exposing the example's configuration and experiment builders.
    """
    path = Path(__file__).resolve().parents[2] / "examples" / "all4one.py"
    spec = importlib.util.spec_from_file_location("all4one_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
