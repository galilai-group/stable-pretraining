"""Synthetic microvolt patches and a dict bridge for the reference sampler."""

from dataclasses import dataclass
from typing import Any

import torch

from .config import Config


@dataclass
class Collate:
    """Preserve N-MET sampling and expose the dictionary expected by Module."""

    sampler: Any

    def __call__(self, samples: list) -> dict[str, torch.Tensor]:
        """Collate reference-format patches with optional valid-token grids."""
        result = self.sampler(samples)
        x, context, target = result[:3]
        valid = (
            result[3] if len(result) == 4 else torch.ones(x.shape[:3], dtype=torch.bool)
        )
        return dict(
            eeg=x,
            context_indices=context[0],
            target_indices=target[0],
            valid_grid=valid,
        )


def synthetic_samples(config: Config, seed: int = 0) -> list:
    """Return seeded microvolt inputs; no dataset reader or preprocessing policy."""
    generator = torch.Generator().manual_seed(seed)
    values = (
        torch.randn(
            config.total_steps * config.batch_size,
            config.channels,
            config.patches,
            200,
            generator=generator,
        )
        * 20
    )
    valid = torch.ones(values.shape[:3], dtype=torch.bool)
    valid[:, -1, -1] = False
    return list(zip(values, valid))
