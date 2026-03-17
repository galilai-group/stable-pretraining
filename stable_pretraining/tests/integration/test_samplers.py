"""Integration tests for samplers with actual datasets and data loading."""

import pytest
import torch
from omegaconf import OmegaConf

import stable_pretraining as spt


@pytest.mark.v1
@pytest.mark.integration
@pytest.mark.download
@pytest.mark.parametrize("n_views", [1, 2, 4])
def test_repeated_sampler_with_dataloader(n_views):
    """Test RepeatedRandomSampler with full data loading pipeline."""
    import logging

    logging.basicConfig(level=logging.INFO)

    # Configuration for train loader
    train = OmegaConf.create(
        {
            "dataset": {
                "_target_": "stable_pretraining.data.HFDataset",
                "path": "ylecun/mnist",
                "split": "train[:128]",
                "transform": {
                    "_target_": "stable_pretraining.data.transforms.ToImage",
                },
            },
            "sampler": {
                "_target_": "stable_pretraining.data.sampler.RepeatedRandomSampler",
                "n_views": n_views,
                "data_source_or_len": 128,
            },
            "batch_size": 128,
        }
    )

    # Test loader creation
    loader = spt.data.instantiate_from_dataloader_config(train, shuffle=True)

    # Verify one batch
    batch = next(iter(loader))
    assert batch["image"].shape[0] == 128
    assert batch["label"].shape[0] == 128

    # Verify repeated indices
    indices = batch["index"]
    unique_indices = torch.unique(indices)
    assert len(indices) == 128
    if n_views > 1:
        # With n_views > 1, we should have repeated indices
        assert len(unique_indices) < len(indices)


@pytest.mark.v1
@pytest.mark.integration
@pytest.mark.download
def test_samplers_with_partial_config():
    """Test samplers with partial configuration."""
    train = OmegaConf.create(
        {
            "dataset": {
                "_target_": "stable_pretraining.data.HFDataset",
                "path": "ylecun/mnist",
                "split": "train[:64]",
                "transform": {
                    "_target_": "stable_pretraining.data.transforms.ToImage",
                },
            },
            "sampler": {
                "_target_": "stable_pretraining.data.sampler.RepeatedRandomSampler",
                "_partial_": True,
                "n_views": 2,
            },
            "batch_size": 32,
        }
    )

    loader = spt.data.instantiate_from_dataloader_config(train, shuffle=True)
    batch = next(iter(loader))
    assert batch["image"].shape[0] == 32
